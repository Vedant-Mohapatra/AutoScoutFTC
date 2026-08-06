"""Turning artifact trajectories into attributed shot attempts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Box, ShotParams
from .geometry import FieldGeometry
from .overlay import OFF, PeriodSpan, period_at
from .tracking import ArtifactTrack


@dataclass
class Shot:
    track_id: int
    color: str
    launch_frame: int
    end_frame: int
    period: str
    launch_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    goal: Optional[str]                 # which goal the artifact landed in
    made: bool
    robot_id: Optional[int] = None
    attribution_px: float = float("inf")


def _in_box(x: float, y: float, box: Box, pad: float = 0.0) -> bool:
    x0, y0, x1, y1 = box
    return (x0 - pad) <= x <= (x1 + pad) and (y0 - pad) <= y <= (y1 + pad)


Point = Tuple[int, float, float]          # frame index, x, y


@dataclass
class Flight:
    """A contiguous stretch of one artifact track during which it was in the air."""

    color: str
    points: List[Point]

    @property
    def start_frame(self) -> int:
        return self.points[0][0]

    @property
    def end_frame(self) -> int:
        return self.points[-1][0]

    @property
    def displacement(self) -> float:
        return float(
            np.hypot(
                self.points[-1][1] - self.points[0][1],
                self.points[-1][2] - self.points[0][2],
            )
        )

    @property
    def rise(self) -> float:
        return self.points[0][2] - self.points[-1][2]

    def velocity(self, at_start: bool, n: int = 4) -> Tuple[float, float]:
        pts = self.points[:n] if at_start else self.points[-n:]
        (f0, x0, y0), (f1, x1, y1) = pts[0], pts[-1]
        span = max(1, f1 - f0)
        return (x1 - x0) / span, (y1 - y0) / span


def _point_speeds(points: Sequence[Point]) -> List[float]:
    """Instantaneous speed at each sample, in px per frame."""
    speeds: List[float] = []
    for i in range(len(points)):
        a = points[max(0, i - 1)]
        b = points[min(len(points) - 1, i + 1)]
        gap = b[0] - a[0]
        if gap <= 0:
            speeds.append(0.0)
            continue
        speeds.append(float(np.hypot(b[1] - a[1], b[2] - a[2])) / gap)
    return speeds


def flights(
    track: ArtifactTrack, params: ShotParams, scale: float = 1.0
) -> List[Flight]:
    """Cut an artifact track into the stretches where it was actually flying.

    One track routinely covers an artifact sitting on the floor, being picked
    up, launched, and rolling after it lands -- and occasionally an identity
    swap between two artifacts.  Classifying whole tracks therefore throws away
    real shots and keeps junk.  Slicing on instantaneous speed isolates the
    ballistic part, which is the only part a shot is made of.
    """
    pts = track.points
    if len(pts) < params.min_points:
        return []

    min_speed = params.min_speed_px_per_frame * scale
    min_disp = params.min_displacement_px * scale
    min_rise = params.min_rise_px * scale

    speeds = _point_speeds(pts)
    out: List[Flight] = []
    i = 0
    while i < len(pts):
        if speeds[i] < min_speed:
            i += 1
            continue
        j = i
        while j + 1 < len(pts) and speeds[j + 1] >= min_speed:
            j += 1
        segment = pts[i: j + 1]
        i = j + 1
        if len(segment) < params.min_points:
            continue
        flight = Flight(track.color, segment)
        if flight.displacement >= min_disp and flight.rise >= min_rise:
            out.append(flight)
    return out


def _goal_hit(
    flight: Flight, goals: Dict[str, Box], lookahead: int = 6
) -> Optional[str]:
    """Which goal, if any, the flight terminates in.

    Checks the observed tail first, then a short constant-velocity
    extrapolation, because the artifact is often lost a frame or two before it
    disappears into the goal.
    """
    for _, x, y in reversed(flight.points[-3:]):
        for name, box in goals.items():
            if _in_box(x, y, box):
                return name

    _, x, y = flight.points[-1]
    vx, vy = flight.velocity(at_start=False)
    for k in range(1, lookahead + 1):
        px, py = x + vx * k, y + vy * k
        for name, box in goals.items():
            if _in_box(px, py, box):
                return name
    return None


def _attribute(
    flight: Flight,
    robot_history: Dict[int, List[Tuple[int, float, float]]],
    params: ShotParams,
    scale: float = 1.0,
) -> Tuple[Optional[int], float]:
    """Find the robot that launched this artifact.

    By the first frame an artifact is visible it has usually already cleared the
    shooter, so we walk the trajectory backwards along its launch velocity and
    take the closest approach to any robot centre -- following the robots back
    in time as we go, since they are moving too.
    """
    f0, x0, y0 = flight.points[0]
    vx, vy = flight.velocity(at_start=True)
    best_rid, best_d = None, float("inf")
    for k in range(0, params.backtrack_frames + 1):
        px, py = x0 - vx * k, y0 - vy * k
        robots = robot_history.get(f0 - k) or _nearest_history(robot_history, f0 - k)
        for rid, rx, ry in robots:
            d = float(np.hypot(px - rx, py - ry))
            if d < best_d:
                best_d, best_rid = d, rid
    if best_d > params.attribution_radius_px * scale:
        return None, best_d
    return best_rid, best_d


def extract_shots(
    tracks: Sequence[ArtifactTrack],
    robot_history: Dict[int, List[Tuple[int, float, float]]],
    spans: Sequence[PeriodSpan],
    params: ShotParams,
    geom: FieldGeometry,
) -> List[Shot]:
    scale = geom.scale
    shots: List[Shot] = []
    for track in tracks:
        for flight in flights(track, params, scale):
            period = period_at(spans, flight.start_frame)
            if period == OFF:
                continue
            goal = _goal_hit(flight, geom.goals, params.goal_lookahead_frames)
            rid, dist = _attribute(flight, robot_history, params, scale)
            _, x0, y0 = flight.points[0]
            _, x1, y1 = flight.points[-1]
            shots.append(
                Shot(
                    track_id=track.tid,
                    color=flight.color,
                    launch_frame=flight.start_frame,
                    end_frame=flight.end_frame,
                    period=period,
                    launch_xy=(x0, y0),
                    end_xy=(x1, y1),
                    goal=goal,
                    made=goal is not None,
                    robot_id=rid,
                    attribution_px=dist,
                )
            )
    return dedupe(shots, px_tol=90.0 * scale)


def _nearest_history(
    robot_history: Dict[int, List[Tuple[int, float, float]]], frame: int
) -> List[Tuple[int, float, float]]:
    if not robot_history:
        return []
    key = min(robot_history, key=lambda f: abs(f - frame))
    return robot_history[key] if abs(key - frame) <= 5 else []


def _close(a: Tuple[float, float], b: Tuple[float, float], tol: float) -> bool:
    return float(np.hypot(a[0] - b[0], a[1] - b[1])) <= tol


def dedupe(shots: Sequence[Shot], frame_tol: int = 10, px_tol: float = 90.0) -> List[Shot]:
    """Collapse the several trajectories one physical shot can produce.

    Two things split a single shot in the tracker: a tumbling artifact can
    momentarily break into two colour blobs, and a long trajectory across the
    whole field can drop below the speed threshold in the middle and come back
    as two flights.  Either way the pieces overlap in time and share a landing
    point, so shots that arrive at the same goal at the same moment -- or that
    leave from the same place at the same moment -- are folded together, keeping
    the earliest launch so attribution still points at the shooter.
    """
    out: List[Shot] = []
    for shot in sorted(shots, key=lambda s: s.launch_frame):
        merged = False
        for kept in out:
            if kept.goal != shot.goal:
                continue
            same_launch = (
                abs(kept.launch_frame - shot.launch_frame) <= frame_tol
                and _close(kept.launch_xy, shot.launch_xy, px_tol)
            )
            same_landing = (
                shot.goal is not None
                and abs(kept.end_frame - shot.end_frame) <= frame_tol
                and _close(kept.end_xy, shot.end_xy, px_tol)
            )
            # A fragment of an already-kept flight: overlapping in time and
            # continuing along it rather than being a fresh launch.
            continuation = (
                shot.launch_frame <= kept.end_frame + frame_tol
                and shot.launch_frame >= kept.launch_frame
                and _close(shot.launch_xy, kept.end_xy, px_tol * 2)
            )
            if same_launch or same_landing or continuation:
                kept.end_frame = max(kept.end_frame, shot.end_frame)
                if shot.end_frame >= kept.end_frame:
                    kept.end_xy = shot.end_xy
                kept.made = kept.made or shot.made
                merged = True
                break
        if not merged:
            out.append(shot)
    return out


# --------------------------------------------------------------------------- #
# Alliance inference
# --------------------------------------------------------------------------- #
def makes_per_goal(shots: Sequence[Shot], goals: Sequence[str] = ("red", "blue")) -> Dict[str, int]:
    """How many artifacts vision saw land in each goal."""
    out = {goal: 0 for goal in goals}
    for shot in shots:
        if shot.made and shot.goal in out:
            out[shot.goal] += 1
    return out


def infer_goal_alliance_map(
    shots: Sequence[Shot],
    totals: Dict[str, int],
    margin: float = 0.15,
) -> Dict[str, str]:
    """Decide which alliance owns which goal, from the scoreboard itself.

    Matching individual makes to individual counter ticks does not work here:
    both alliances score more or less continuously, so within any plausible
    scoring-latency window there is nearly always a tick from *either* alliance
    and the "nearest" one is a coin flip.

    What does discriminate is the totals.  Over a whole match the goal that
    received more artifacts must belong to the alliance the scoreboard credits
    with more artifacts.  That survives a mirrored camera or a swapped overlay,
    and when the two alliances finish too close together to call, it says so
    instead of guessing.
    """
    seen = makes_per_goal(shots)
    goals = sorted(seen, key=lambda g: -seen[g])
    alliances = sorted(totals, key=lambda a: -totals[a])
    if len(goals) != 2 or len(alliances) != 2:
        return {}

    def separated(values: Sequence[int]) -> bool:
        hi, lo = values
        return hi > 0 and (hi - lo) / hi >= margin

    if not separated([seen[g] for g in goals]):
        return {}
    if not separated([totals[a] for a in alliances]):
        return {}
    return {alliances[i]: goals[i] for i in range(2)}


def infer_robot_alliances(
    shots: Sequence[Shot],
    alliance_goal: Dict[str, str],
    robot_ids: Sequence[int],
    robot_start_x: Dict[int, float],
    field_centre_x: float,
) -> Dict[int, str]:
    """Assign each tracked robot to an alliance.

    Primary evidence is which goal the robot shoots at.  A robot that never
    takes a shot falls back to the side of the field it started on, and the
    result is balanced to two robots per alliance.
    """
    goal_alliance = {goal: alliance for alliance, goal in alliance_goal.items()}
    tally: Dict[int, Dict[str, int]] = {rid: {"red": 0, "blue": 0} for rid in robot_ids}
    for shot in shots:
        if shot.robot_id is None or shot.goal is None:
            continue
        alliance = goal_alliance.get(shot.goal)
        if alliance:
            tally[shot.robot_id][alliance] += 1

    scored = {
        rid: (t["red"] - t["blue"]) / max(1, t["red"] + t["blue"])
        for rid, t in tally.items()
    }
    # Robots with no shooting evidence lean on their starting x position; the
    # red goal sits on the right of the frame.
    for rid in robot_ids:
        if tally[rid]["red"] + tally[rid]["blue"] == 0:
            x = robot_start_x.get(rid, 0.0)
            scored[rid] = 0.001 if x > field_centre_x else -0.001

    order = sorted(robot_ids, key=lambda r: -scored[r])
    out: Dict[int, str] = {}
    half = len(order) // 2
    for i, rid in enumerate(order):
        out[rid] = "red" if i < half else "blue"
    return out
