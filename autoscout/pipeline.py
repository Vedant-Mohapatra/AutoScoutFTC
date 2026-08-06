"""End-to-end scouting run: video in, per-robot shooting statistics out."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import detect, overlay, shots as shots_mod
from .cache import Cache
from .config import (
    ARTIFACT_COLORS,
    AUTO_CLOCK_START,
    BALL_AREA,
    BALL_DIM,
    Config,
    N_ROBOTS,
    TELEOP_CLOCK_START,
)
from .detect import Blob, RobotCandidate
from .geometry import FieldGeometry, calibrate
from .overlay import AUTO, MatchLabel, TELEOP, TRANSITION, OverlaySample, PeriodSpan
from .tracking import ArtifactTracker, RobotTracker

OVERLAY_STRIDE = 8          # frames between scoreboard reads (~0.13 s at 60 fps)
#: First row kept from each sampled frame.  High enough to include the match
#: caption above the bar, which is what names the output directory.
PLATE_TOP = 810
#: Frames between kept scoreboard strips (~6 s at 60 fps).
PLATE_STRIDE = OVERLAY_STRIDE * 48


def _pack_plate(frame: np.ndarray) -> bytes:
    """PNG-encode the scoreboard bar so the cache stays small and lossless."""
    ok, buf = cv2.imencode(".png", frame[PLATE_TOP:])
    if not ok:
        raise RuntimeError("failed to encode scoreboard strip")
    return buf.tobytes()


def _restore_plate(packed: bytes, width: int = 1920, height: int = 1080) -> np.ndarray:
    """Re-embed a cached scoreboard strip in a full-size frame."""
    strip = cv2.imdecode(np.frombuffer(packed, np.uint8), cv2.IMREAD_COLOR)
    frame = np.zeros((height, width, 3), np.uint8)
    frame[PLATE_TOP: PLATE_TOP + strip.shape[0], : strip.shape[1]] = strip
    return frame


@dataclass
class RobotStats:
    robot_id: int
    alliance: str
    team: Optional[int] = None
    attempts: Dict[str, int] = field(default_factory=lambda: {AUTO: 0, TELEOP: 0})
    made: Dict[str, int] = field(default_factory=lambda: {AUTO: 0, TELEOP: 0})
    #: Share of match frames where this identity sat on a real detection rather
    #: than coasting.  A low value means the row below it is guesswork.
    tracked: float = 1.0

    @property
    def total_attempts(self) -> int:
        return sum(self.attempts.values())

    @property
    def total_made(self) -> int:
        return sum(self.made.values())

    @property
    def accuracy(self) -> Optional[float]:
        return self.total_made / self.total_attempts if self.total_attempts else None


@dataclass
class MatchResult:
    video: str
    fps: float
    label: MatchLabel
    geometry: FieldGeometry
    spans: List[PeriodSpan]
    teams: Dict[str, List[int]]
    #: Alliance artifact totals straight off the scoreboard, per period.
    scoreboard: Dict[str, Dict[str, int]]
    final_score: Dict[str, Optional[int]]
    robots: List[RobotStats]
    shots: List[shots_mod.Shot]
    alliance_goal: Dict[str, str]
    robot_tracks: Dict[int, Dict[int, Tuple[float, float]]]
    warnings: List[str] = field(default_factory=list)

    def clock_at(self, frame_index: int) -> Optional[float]:
        """Seconds remaining on the on-screen match clock."""
        for span in self.spans:
            if not span.contains(frame_index):
                continue
            elapsed = (frame_index - span.start_frame) / self.fps
            if span.name == AUTO:
                return max(0.0, AUTO_CLOCK_START - elapsed)
            if span.name == TELEOP:
                return max(0.0, TELEOP_CLOCK_START - elapsed)
            total = (span.end_frame - span.start_frame) / self.fps
            return max(0.0, total - elapsed)
        return None


ProgressFn = Callable[[str, float], None]


def _noop(_stage: str, _frac: float) -> None:
    pass


# --------------------------------------------------------------------------- #
# Pass 1: scoreboard
# --------------------------------------------------------------------------- #
def _scan_overlay(
    cap: "cv2.VideoCapture", total: int, progress: ProgressFn
) -> Tuple[List[OverlaySample], List[Tuple[int, bytes]]]:
    samples: List[OverlaySample] = []
    plates: List[Tuple[int, bytes]] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % OVERLAY_STRIDE == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            samples.append(
                OverlaySample(
                    frame_index=idx,
                    icon=overlay.read_period_icon(frame),
                    clock=overlay.read_clock(frame),
                    counters=overlay.read_counters(frame),
                )
            )
            if idx % PLATE_STRIDE == 0:
                # Keep only the scoreboard strip, and only every few seconds of
                # it: the teams, the caption and the final score never change
                # within a match, so a few dozen samples is plenty to vote on
                # and whole frames would balloon the cache for no benefit.
                # ``_restore_plate`` puts the strip back at the right offset so
                # the configured boxes still apply.
                plates.append((idx, _pack_plate(frame)))
            if total:
                progress("scoreboard", idx / total)
        idx += 1
    return samples, plates


def _counter_series(
    samples: Sequence[OverlaySample],
) -> Dict[str, List[Tuple[int, int]]]:
    """Each alliance's artifact counter over time, with OCR dropouts removed."""
    series: Dict[str, List[Tuple[int, int]]] = {"red": [], "blue": []}
    for alliance in ("red", "blue"):
        raw = [
            (s.frame_index, s.counters.scored(alliance) if s.counters else None)
            for s in samples
        ]
        values = overlay.enforce_monotonic([v for _, v in raw])
        series[alliance] = [(f, v) for (f, _), v in zip(raw, values) if v is not None]
    return series


def _scoreboard_totals(
    series: Dict[str, List[Tuple[int, int]]], spans: Sequence[PeriodSpan]
) -> Dict[str, Dict[str, int]]:
    """Artifacts scored per alliance per period, read off the counter series."""
    out: Dict[str, Dict[str, int]] = {}
    span_by_name = {s.name: s for s in spans}
    for alliance, points in series.items():
        totals: Dict[str, int] = {}
        if not points:
            out[alliance] = totals
            continue

        def value_at(frame_index: int) -> int:
            best = 0
            for f, v in points:
                if f <= frame_index:
                    best = v
                else:
                    break
            return best

        auto = span_by_name.get(AUTO)
        teleop = span_by_name.get(TELEOP)
        if auto:
            # The scorer keeps crediting autonomous artifacts through the
            # transition, so take the value at the end of the hands-off period.
            end = span_by_name[TRANSITION].end_frame if TRANSITION in span_by_name else auto.end_frame
            totals[AUTO] = value_at(end)
        if teleop:
            start = totals.get(AUTO, value_at(teleop.start_frame))
            totals[TELEOP] = max(0, points[-1][1] - start)
        totals["total"] = points[-1][1]
        out[alliance] = totals
    return out


# --------------------------------------------------------------------------- #
# Pass 2: vision
# --------------------------------------------------------------------------- #
#: Per-frame detections, stored as compact arrays:
#: frame_index -> (robots (n, 3+SIGNATURE_BINS) = x,y,score,colour signature ;
#:                 artifacts (m,4) = x,y,area,colour_index)
Detections = Dict[int, Tuple[np.ndarray, np.ndarray]]


def _scan_detections(
    cap: "cv2.VideoCapture",
    background: np.ndarray,
    geom: FieldGeometry,
    first: int,
    last: int,
    cfg: Config,
    progress: ProgressFn,
) -> Detections:
    """Decode the match once and record what is where in every frame.

    Detection is the expensive half and depends only on the video and the
    appearance thresholds, so it is separated from tracking -- which is cheap,
    and which we want to be able to re-run with different parameters.
    """
    inside = detect.field_mask(geom)
    allowed = detect.artifact_mask_region(geom)
    bg_work = detect.crop(background, geom)
    colour_index = {c.name: i for i, c in enumerate(ARTIFACT_COLORS)}

    out: Detections = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    span = max(1, last - first)
    for idx in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        work = detect.crop(frame, geom)
        fg = detect.foreground(work, bg_work, inside, cfg.robot)
        cands = detect.detect_robots(fg, work, inside, geom, cfg.robot)
        blobs = detect.detect_artifacts(work, allowed, geom)
        out[idx] = (
            np.array(
                [[c.x, c.y, c.score, *c.signature] for c in cands], np.float32
            ).reshape(-1, 3 + detect.SIGNATURE_BINS),
            np.array(
                [[b.x, b.y, b.area, colour_index[b.color]] for b in blobs], np.float32
            ).reshape(-1, 4),
        )
        if (idx - first) % 60 == 0:
            progress("detecting", (idx - first) / span)
    return out


def _track(
    detections: Detections, cfg: Config
) -> Tuple[List, List, Dict[int, List[Tuple[int, float, float]]]]:
    colours = [c.name for c in ARTIFACT_COLORS]
    robots = RobotTracker(cfg.robot)
    artifacts = ArtifactTracker(cfg.shot)
    history: Dict[int, List[Tuple[int, float, float]]] = {}

    frames = sorted(detections)
    first_frame = frames[0] if frames else 0
    for idx in frames:
        rob, art = detections[idx]
        cands = [
            RobotCandidate(float(r[0]), float(r[1]), float(r[2]), r[3:].astype(np.float32))
            for r in rob
        ]
        if not robots.tracks:
            waited = idx - first_frame
            robots.seed(
                cands, idx, strict=waited < cfg.robot.seed_patience_frames
            )
        else:
            robots.update(cands, idx)
        history[idx] = robots.positions()

        blobs = [
            Blob(float(a[0]), float(a[1]), int(a[2]), colours[int(a[3])]) for a in art
        ]
        artifacts.update(blobs, idx)

    return robots.tracks, artifacts.finish(), history


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def analyse(
    video_path: str,
    cfg: Optional[Config] = None,
    progress: ProgressFn = _noop,
    cache: Optional[Cache] = None,
) -> MatchResult:
    cfg = cfg or Config()
    cache = cache or Cache(None)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    warnings: List[str] = []

    cached = cache.load(video_path, "overlay")
    if cached is None:
        cached = _scan_overlay(cap, total, progress)
        cache.store(video_path, "overlay", cached)
    samples, plates = cached

    spans = overlay.segment_periods(samples, fps)
    if not any(s.name == AUTO for s in spans) or not any(s.name == TELEOP for s in spans):
        raise RuntimeError(
            "could not find both an autonomous and a driver-controlled period; "
            "is this a full match recording with the FTC scoreboard visible?"
        )
    _check_period_order(spans, fps, warnings)

    series = _counter_series(samples)
    scoreboard = _scoreboard_totals(series, spans)

    first = min(s.start_frame for s in spans)
    last = max(s.end_frame for s in spans)

    # Team plates are only rendered while a match is loaded, so vote on frames
    # from inside the match itself.
    in_match = [_restore_plate(p) for i, p in plates if first <= i <= last]
    teams = dict(cfg.teams) if cfg.teams else overlay.read_teams(in_match)
    label = overlay.vote_match_label(in_match) if in_match else MatchLabel(None, None)
    final_score = _read_final_score(plates, last)

    progress("background", 0.0)
    bg_indices = np.linspace(first, last, cfg.robot.bg_samples).astype(int)
    background = detect.build_background(cap, bg_indices)
    geom = cfg.geometry or calibrate(background, cfg.proportions)

    salt = _detection_salt(cfg, geom)
    detections = cache.load(video_path, "detections", salt=salt)
    if detections is None:
        detections = _scan_detections(
            cap, background, geom, first, last, cfg, progress
        )
        cache.store(video_path, "detections", detections, salt=salt)
    cap.release()

    progress("tracking", 0.0)
    robot_tracks, artifact_tracks, history = _track(detections, cfg)
    progress("tracking", 1.0)

    if len(robot_tracks) < N_ROBOTS:
        warnings.append(
            f"only seeded {len(robot_tracks)} of {N_ROBOTS} robots; "
            "per-robot numbers will be incomplete"
        )

    all_shots = shots_mod.extract_shots(
        artifact_tracks, history, spans, cfg.shot, geom
    )

    totals = {a: scoreboard.get(a, {}).get("total", 0) for a in ("red", "blue")}
    measured = shots_mod.infer_goal_alliance_map(all_shots, totals)
    alliance_goal = dict(cfg.goal_of_alliance)
    if measured and measured != alliance_goal:
        alliance_goal = measured
        warnings.append(
            f"goal ownership measured from the scoreboard ({measured}) differs from "
            f"the configured default ({cfg.goal_of_alliance}); using the measurement"
        )
    elif not measured:
        warnings.append(
            "the two alliances finished too close together to verify which alliance "
            "owns which goal from the scoreboard; using the colour-matched default"
        )

    robot_ids = [t.rid for t in robot_tracks]
    start_x = {t.rid: t.history.get(first, (t.x, t.y))[0] for t in robot_tracks}
    centre_x = (geom.floor[0] + geom.floor[2]) / 2.0
    alliances = shots_mod.infer_robot_alliances(
        all_shots, alliance_goal, robot_ids, start_x, centre_x
    )

    coverage = _track_coverage(detections, history)
    stats = {
        rid: RobotStats(
            robot_id=rid, alliance=alliances[rid], tracked=coverage.get(rid, 1.0)
        )
        for rid in robot_ids
    }
    for shot in all_shots:
        if shot.robot_id is None or shot.period not in (AUTO, TELEOP):
            continue
        s = stats[shot.robot_id]
        s.attempts[shot.period] += 1
        if shot.made:
            s.made[shot.period] += 1

    _assign_teams(stats, teams, cfg.robot_teams)
    _reconcile(all_shots, alliance_goal, scoreboard, warnings)

    unattributed = sum(1 for s in all_shots if s.robot_id is None)
    if unattributed:
        warnings.append(
            f"{unattributed} of {len(all_shots)} detected shots could not be traced "
            "back to a robot and are excluded from the per-robot table"
        )

    return MatchResult(
        video=video_path,
        fps=fps,
        label=label,
        geometry=geom,
        spans=spans,
        teams=teams,
        scoreboard=scoreboard,
        final_score=final_score,
        robots=sorted(stats.values(), key=lambda s: (s.alliance, s.robot_id)),
        shots=all_shots,
        alliance_goal=alliance_goal,
        robot_tracks={t.rid: t.history for t in robot_tracks},
        warnings=warnings,
    )


def _track_coverage(
    detections: Detections,
    history: Dict[int, List[Tuple[int, float, float]]],
    tolerance: float = 25.0,
) -> Dict[int, float]:
    """How often each identity was actually on a detection.

    A track with no detection keeps its last position, which is right for a
    parked robot and wrong for a lost one -- and nothing downstream can tell
    the difference.  Reporting the rate lets a reader discount a row instead of
    trusting it blindly.
    """
    hits: Dict[int, int] = {}
    total: Dict[int, int] = {}
    for frame_index, positions in history.items():
        det = detections.get(frame_index)
        robots = det[0] if det is not None else None
        for rid, x, y in positions:
            total[rid] = total.get(rid, 0) + 1
            if robots is not None and len(robots):
                d = np.min(np.hypot(robots[:, 0] - x, robots[:, 1] - y))
                if d <= tolerance:
                    hits[rid] = hits.get(rid, 0) + 1
    return {rid: hits.get(rid, 0) / n for rid, n in total.items() if n}


def _read_final_score(
    plates: Sequence[Tuple[int, bytes]], last_match_frame: int
) -> Dict[str, Optional[int]]:
    """Settled alliance point totals.

    The scoreboard animates the total upwards for a second or two after the
    buzzer, and then the broadcast cuts to a results or awards screen where the
    panel is gone entirely.  So walk backwards from the end of the recording and
    take the last frame that still shows both numbers.
    """
    for _, packed in sorted(plates, reverse=True):
        scores = overlay.read_scores(_restore_plate(packed))
        if all(v is not None for v in scores.values()):
            return scores
    for idx, packed in sorted(plates, reverse=True):
        if idx <= last_match_frame:
            return overlay.read_scores(_restore_plate(packed))
    return {}


def _check_period_order(
    spans: Sequence[PeriodSpan], fps: float, warnings: List[str]
) -> None:
    order = [s.name for s in spans]
    expected = [n for n in (AUTO, TRANSITION, TELEOP) if n in order]
    if order != expected:
        warnings.append(f"periods came out in an unexpected order: {order}")
    for span in spans:
        length = (span.end_frame - span.start_frame) / fps
        nominal = {AUTO: 30.0, TELEOP: 120.0}.get(span.name)
        if nominal and abs(length - nominal) > 6.0:
            warnings.append(
                f"{span.name} measured {length:.1f}s but should be about "
                f"{nominal:.0f}s; the clock may have been misread"
            )


def _assign_teams(
    stats: Dict[int, RobotStats],
    teams: Dict[str, List[int]],
    manual: Dict[int, int],
) -> None:
    """Attach team numbers to tracked robots where that is actually knowable.

    The scoreboard lists two teams per alliance but nothing on screen says which
    plate belongs to which chassis, so a guess would be a coin flip dressed up
    as data.  A pairing is claimed only when it is forced -- one robot, one
    number -- or when the caller supplied it by hand.
    """
    for rid, number in manual.items():
        if rid in stats:
            stats[rid].team = number

    for alliance in ("red", "blue"):
        members = [
            s for s in stats.values() if s.alliance == alliance and s.team is None
        ]
        numbers = [
            n for n in teams.get(alliance, [])
            if n not in {s.team for s in stats.values()}
        ]
        if len(members) == 1 and len(numbers) == 1:
            members[0].team = numbers[0]


def _reconcile(
    shots: Sequence[shots_mod.Shot],
    alliance_goal: Dict[str, str],
    scoreboard: Dict[str, Dict[str, int]],
    warnings: List[str],
) -> None:
    """Compare vision-detected makes against the authoritative scoreboard.

    Counted per goal rather than per robot, so a mis-attributed shot does not
    show up here -- this measures the detector, not the attribution.
    """
    seen = shots_mod.makes_per_goal(shots)
    for alliance, goal in alliance_goal.items():
        expected = scoreboard.get(alliance, {}).get("total")
        if not expected:
            continue
        got = seen.get(goal, 0)
        if abs(got - expected) / expected > 0.2:
            warnings.append(
                f"{alliance}: vision saw {got} artifacts land in the {goal} goal "
                f"but the scoreboard counted {expected} -- read the per-robot "
                "table as a relative split, not an exact count"
            )



def _detection_salt(cfg: Config, geom: FieldGeometry) -> str:
    """Everything that changes what the detectors output.

    Anything omitted here would silently serve a stale cache after a tweak, so
    the appearance thresholds and the field geometry are all folded in.  The
    association and recovery settings are deliberately *not* here: they belong
    to tracking, which runs outside the cache, and including them would throw
    away a ten-minute decode every time one of them moved.
    """
    parts = [
        f"{cfg.robot.fg_threshold}|{cfg.robot.min_fill}|{cfg.robot.bg_samples}",
        repr(ARTIFACT_COLORS),
        repr(BALL_AREA),
        repr(BALL_DIM),
        repr(geom.ball_roi),
        repr(geom.work_roi),
        repr(geom.chutes),
        geom.field_poly.tobytes().hex(),
        str(geom.robot_size),
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
