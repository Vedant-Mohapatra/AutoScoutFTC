"""Multi-object tracking for the four robots and for artifacts in flight."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import N_ROBOTS, RobotParams, ShotParams
from .detect import Blob, RobotCandidate


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #
def greedy_assign(
    cost: np.ndarray, gate: float
) -> Dict[int, int]:
    """Cheapest-first assignment of rows to columns under a cost ceiling.

    With four tracks and a handful of detections this is indistinguishable from
    an optimal assignment, and it keeps the dependency list to numpy + OpenCV.
    """
    pairs: Dict[int, int] = {}
    if cost.size == 0:
        return pairs
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    taken_rows, taken_cols = set(), set()
    for r, c in order:
        r, c = int(r), int(c)
        if r in taken_rows or c in taken_cols:
            continue
        if not np.isfinite(cost[r, c]) or cost[r, c] > gate:
            break
        pairs[r] = c
        taken_rows.add(r)
        taken_cols.add(c)
    return pairs


# --------------------------------------------------------------------------- #
# Robots
# --------------------------------------------------------------------------- #
@dataclass
class RobotTrack:
    rid: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    coast: int = 0
    #: Running colour profile of this robot, updated slowly while it is tracked.
    signature: Optional[np.ndarray] = None
    history: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    def predict(self) -> Tuple[float, float]:
        return self.x + self.vx, self.y + self.vy

    def blend_signature(self, sig: Optional[np.ndarray], rate: float = 0.02) -> None:
        if sig is None or not sig.any():
            return
        if self.signature is None:
            self.signature = sig.astype(np.float32).copy()
        else:
            self.signature = (1 - rate) * self.signature + rate * sig


def signature_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """1 - histogram intersection, averaged over the hue and brightness halves."""
    if a is None or b is None:
        return 0.0
    hue = np.minimum(a[:8], b[:8]).sum()
    val = np.minimum(a[8:], b[8:]).sum()
    return float(1.0 - 0.5 * (hue + val))


class RobotTracker:
    """Keeps four stable identities for the whole match.

    Identities are seeded from the starting positions in autonomous and then
    maintained by prediction + nearest-detection association.  Robots are never
    created or destroyed mid-match, which matches reality and stops a
    momentarily-occluded robot from stealing a new id.
    """

    def __init__(self, params: RobotParams, n: int = N_ROBOTS):
        self.params = params
        self.n = n
        self.tracks: List[RobotTrack] = []

    def seed(
        self,
        candidates: Sequence[RobotCandidate],
        frame_index: int,
        strict: bool = True,
    ) -> bool:
        """Fix the four identities from one frame.

        Seeding on the very first frame of the match is tempting but fragile: if
        a robot happens to be faint against the background at that instant, the
        fourth slot goes to whatever noise scored highest and that identity
        spends the match somewhere it never was.  Waiting for a frame where four
        solid, clearly separate robots are visible costs a fraction of a second
        and avoids that.
        """
        ranked = sorted(candidates, key=lambda c: -c.score)
        picks = ranked[: self.n]
        if len(picks) < self.n:
            return False
        if strict:
            # More robot-sized masses than there are robots means something else
            # is on the field -- a broadcast wipe, or the reset crew still
            # clearing it.  Seeding then pins identities to people.
            strong = [c for c in ranked if c.score >= self.params.seed_min_fill]
            if len(strong) != self.n:
                return False
            for i in range(len(picks)):
                for j in range(i + 1, len(picks)):
                    if float(np.hypot(picks[i].x - picks[j].x,
                                      picks[i].y - picks[j].y)) < self.params.min_separation_px:
                        return False
        # Deterministic ids: top-to-bottom, then left-to-right.
        picks = sorted(picks, key=lambda c: (round(c.y / 120), c.x))
        self.tracks = [
            RobotTrack(
                rid=i, x=p.x, y=p.y,
                signature=None if p.signature is None else p.signature.copy(),
                history={frame_index: (p.x, p.y)},
            )
            for i, p in enumerate(picks)
        ]
        return True

    def update(
        self, candidates: Sequence[RobotCandidate], frame_index: int
    ) -> List[RobotTrack]:
        if not self.tracks:
            return []
        preds = np.array([t.predict() for t in self.tracks], dtype=np.float32)
        if candidates:
            dets = np.array([[c.x, c.y] for c in candidates], dtype=np.float32)
            dist = np.linalg.norm(preds[:, None, :] - dets[None, :, :], axis=2)
            cost = dist.copy()
            # Position decides which detections are even eligible; colour then
            # decides between them.  Without that second term two tracks drift
            # onto the same robot and never come apart, because the peak finder
            # can put two responses on one chassis.
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(candidates):
                    cost[i, j] += self.params.appearance_weight * signature_distance(
                        track.signature, det.signature
                    )
            cost[dist > self.params.assoc_gate_px] = np.inf
            pairs = greedy_assign(cost, np.inf)
        else:
            pairs = {}

        for i, track in enumerate(self.tracks):
            if i in pairs:
                det = candidates[pairs[i]]
                track.vx = 0.6 * (det.x - track.x) + 0.4 * track.vx
                track.vy = 0.6 * (det.y - track.y) + 0.4 * track.vy
                track.x, track.y = det.x, det.y
                track.coast = 0
                track.blend_signature(det.signature)
            else:
                # No detection: drift on the last velocity, decaying quickly so a
                # lost robot settles instead of flying off the field.
                track.x, track.y = track.predict()
                track.vx *= 0.5
                track.vy *= 0.5
                track.coast += 1

        self._recover(candidates, set(pairs.values()))
        self._separate()
        for track in self.tracks:
            track.history[frame_index] = (track.x, track.y)
        return self.tracks

    def _recover(
        self, candidates: Sequence[RobotCandidate], claimed: Set[int]
    ) -> None:
        """Let a track that has been lost for a while jump to a spare robot.

        The association gate is deliberately tight, which means a track that
        gets separated from its robot -- during a collision, or while the robot
        is pressed against the wall -- can never walk back to it and simply
        sits where it died for the rest of the match.  Since the field always
        holds exactly four robots, a strong detection that no track has claimed
        is almost certainly the missing one, and the colour signature says which
        missing one it is.
        """
        lost = [t for t in self.tracks if t.coast >= self.params.recover_after]
        spare = [
            j for j, c in enumerate(candidates)
            if j not in claimed and c.score >= self.params.recover_min_fill
        ]
        if not lost or not spare:
            return

        cost = np.array(
            [[signature_distance(t.signature, candidates[j].signature) for j in spare]
             for t in lost],
            dtype=np.float32,
        )
        for i, j in greedy_assign(cost, self.params.recover_max_signature_distance).items():
            det = candidates[spare[j]]
            track = lost[i]
            track.x, track.y = det.x, det.y
            track.vx = track.vy = 0.0
            track.coast = 0
            track.blend_signature(det.signature, rate=0.25)

    def _separate(self) -> None:
        """Push apart tracks that have ended up on the same robot.

        Two robots can touch, but their centres cannot coincide.  When a track
        loses its robot it tends to drift onto a neighbour and latch there, so
        any pair closer than a chassis width is nudged apart along the line
        between them -- enough for the next frame's assignment to pull them onto
        separate detections again.
        """
        floor = self.params.min_separation_px
        for _ in range(2):
            for i in range(len(self.tracks)):
                for j in range(i + 1, len(self.tracks)):
                    a, b = self.tracks[i], self.tracks[j]
                    dx, dy = b.x - a.x, b.y - a.y
                    d = float(np.hypot(dx, dy))
                    if d >= floor:
                        continue
                    if d < 1e-3:
                        dx, dy, d = 1.0, 0.0, 1.0
                    push = (floor - d) / 2.0
                    ux, uy = dx / d, dy / d
                    # Only a coasting track gives way -- a track sitting on a
                    # real detection is where the evidence says it is, and
                    # shoving it off would invent an error rather than fix one.
                    wa = 1.0 if a.coast else 0.0
                    wb = 1.0 if b.coast else 0.0
                    if wa + wb == 0:
                        continue
                    total = wa + wb
                    a.x -= ux * push * 2 * wa / total
                    a.y -= uy * push * 2 * wa / total
                    b.x += ux * push * 2 * wb / total
                    b.y += uy * push * 2 * wb / total

    def positions(self) -> List[Tuple[int, float, float]]:
        return [(t.rid, t.x, t.y) for t in self.tracks]


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
@dataclass
class ArtifactTrack:
    tid: int
    color: str
    points: List[Tuple[int, float, float]]     # frame_index, x, y
    vx: float = 0.0
    vy: float = 0.0
    miss: int = 0
    alive: bool = True

    @property
    def start_frame(self) -> int:
        return self.points[0][0]

    @property
    def end_frame(self) -> int:
        return self.points[-1][0]

    @property
    def displacement(self) -> float:
        _, x0, y0 = self.points[0]
        _, x1, y1 = self.points[-1]
        return float(np.hypot(x1 - x0, y1 - y0))

    @property
    def rise(self) -> float:
        """How far up-screen the artifact travelled (goals are at the top)."""
        return self.points[0][2] - self.points[-1][2]

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def speed(self) -> float:
        return self.displacement / self.duration if self.duration else 0.0

    def launch_velocity(self, n: int = 4) -> Tuple[float, float]:
        pts = self.points[: max(2, min(n, len(self.points)))]
        (f0, x0, y0), (f1, x1, y1) = pts[0], pts[-1]
        span = max(1, f1 - f0)
        return (x1 - x0) / span, (y1 - y0) / span


class ArtifactTracker:
    """Nearest-neighbour tracker with constant-velocity prediction.

    Artifacts are small, numerous and identical, so the only thing that
    separates a launched one from the dozens sitting on the floor is motion --
    which is why every candidate is tracked and the filtering happens later, on
    whole trajectories.
    """

    def __init__(self, params: ShotParams):
        self.params = params
        self.tracks: List[ArtifactTrack] = []
        self._live: List[ArtifactTrack] = []
        self._next_id = 0

    def update(self, blobs: Sequence[Blob], frame_index: int) -> None:
        preds = []
        for t in self._live:
            _, x, y = t.points[-1]
            preds.append((x + t.vx, y + t.vy))

        used: set[int] = set()
        if preds and blobs:
            p = np.array(preds, dtype=np.float32)
            d = np.array([[b.x, b.y] for b in blobs], dtype=np.float32)
            cost = np.linalg.norm(p[:, None, :] - d[None, :, :], axis=2)
            for i, t in enumerate(self._live):
                # Colour is a hard constraint: an artifact cannot change hue.
                for j, b in enumerate(blobs):
                    if b.color != t.color:
                        cost[i, j] = np.inf
                # A fast artifact needs a wider gate than a stationary one, both
                # because prediction error grows with speed and because it may
                # have been missed for a frame or two behind a robot.
                speed = float(np.hypot(t.vx, t.vy))
                gate = self.params.assoc_gate_px + self.params.gate_speed_gain * speed
                gate *= 1 + t.miss
                cost[i][cost[i] > gate] = np.inf
            pairs = greedy_assign(cost, np.inf)
        else:
            pairs = {}

        for i, track in enumerate(self._live):
            if i in pairs:
                j = pairs[i]
                used.add(j)
                b = blobs[j]
                _, lx, ly = track.points[-1]
                track.vx = 0.6 * (b.x - lx) + 0.4 * track.vx
                track.vy = 0.6 * (b.y - ly) + 0.4 * track.vy
                track.points.append((frame_index, b.x, b.y))
                track.miss = 0
            else:
                track.miss += 1
                if track.miss > self.params.max_miss:
                    track.alive = False

        for j, b in enumerate(blobs):
            if j in used:
                continue
            t = ArtifactTrack(self._next_id, b.color, [(frame_index, b.x, b.y)])
            self._next_id += 1
            self.tracks.append(t)
            self._live.append(t)

        self._live = [t for t in self._live if t.alive]

    def finish(self) -> List[ArtifactTrack]:
        for t in self._live:
            t.alive = False
        self._live = []
        return self.tracks
