"""Per-frame detection of robots and artifacts in the overhead field view."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import ARTIFACT_COLORS, BALL_AREA, BALL_DIM, N_ROBOTS, RobotParams
from .geometry import FieldGeometry


@dataclass(frozen=True)
class Blob:
    x: float
    y: float
    area: int
    color: str


#: Length of the colour signature attached to each robot candidate.
SIGNATURE_BINS = 12


@dataclass(frozen=True)
class RobotCandidate:
    x: float
    y: float
    score: float
    #: Rotation-invariant colour histogram of the robot's own pixels.  Robots
    #: spin constantly, so anything spatial would be useless; hue and brightness
    #: are what actually separate the black chassis from the green one.
    signature: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- #
# Background plate
# --------------------------------------------------------------------------- #
def build_background(
    capture: "cv2.VideoCapture",
    frame_indices: Sequence[int],
) -> np.ndarray:
    """Median plate of the empty-ish field.

    Robots and artifacts move constantly, so the per-pixel median across frames
    spread over the whole match reproduces the bare floor and the static
    surroundings.
    """
    stack: List[np.ndarray] = []
    for idx in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = capture.read()
        if ok:
            stack.append(frame)
    if not stack:
        raise RuntimeError("could not read any frame while building the background")

    frames = np.stack(stack)
    # np.median promotes to float64, which for eighty 1080p frames is several
    # gigabytes in one allocation.  Doing it in horizontal bands keeps the peak
    # working set to a few hundred megabytes.
    out = np.empty(frames.shape[1:], np.uint8)
    band = max(1, 2 ** 24 // (frames.shape[0] * frames.shape[2] * frames.shape[3]))
    for y in range(0, frames.shape[1], band):
        out[y: y + band] = np.median(frames[:, y: y + band], axis=0).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
# Masks, in work-window coordinates
# --------------------------------------------------------------------------- #
def crop(image: np.ndarray, geom: FieldGeometry) -> np.ndarray:
    """Restrict a full frame to the region the detectors care about."""
    x0, y0, x1, y1 = geom.work_roi
    return image[y0:y1, x0:x1]


def _work_shape(geom: FieldGeometry) -> Tuple[int, int]:
    x0, y0, x1, y1 = geom.work_roi
    return y1 - y0, x1 - x0


def field_mask(geom: FieldGeometry) -> np.ndarray:
    """Where a robot centre may be, in work-window coordinates."""
    x0, y0, _, _ = geom.work_roi
    mask = np.zeros(_work_shape(geom), np.uint8)
    cv2.fillPoly(mask, [geom.robot_poly - np.array([[x0, y0]], np.int32)], 255)
    return mask


def artifact_mask_region(geom: FieldGeometry) -> np.ndarray:
    """Where artifacts are allowed to be detected, in work-window coordinates.

    Covers the floor plus the band above the perimeter wall (a shot is still in
    view there as it drops into a goal) and excludes the two human-player
    chutes, which are permanently stacked with stationary artifacts.
    """
    ox, oy, _, _ = geom.work_roi
    mask = np.zeros(_work_shape(geom), np.uint8)
    x0, y0, x1, y1 = geom.ball_roi
    mask[max(0, y0 - oy): y1 - oy, max(0, x0 - ox): x1 - ox] = 255
    for cx0, cy0, cx1, cy1 in geom.chutes:
        mask[max(0, cy0 - oy): cy1 - oy, max(0, cx0 - ox): cx1 - ox] = 0
    return mask


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def detect_artifacts(
    work: np.ndarray, allowed: np.ndarray, geom: FieldGeometry
) -> List[Blob]:
    """Artifact blobs in ``work`` (a cropped frame), returned in frame coordinates."""
    ox, oy, _, _ = geom.work_roi
    area_lo = BALL_AREA[0] * geom.scale ** 2
    area_hi = BALL_AREA[1] * geom.scale ** 2
    dim_lo = BALL_DIM[0] * geom.scale
    dim_hi = BALL_DIM[1] * geom.scale

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)

    blobs: List[Blob] = []
    for spec in ARTIFACT_COLORS:
        m = (
            (h >= spec.h[0]) & (h <= spec.h[1]) & (s >= spec.s_min) & (v >= spec.v_min)
        ).astype(np.uint8)
        m &= (allowed > 0)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, count):
            _, _, w, ht, area = stats[i]
            if not (area_lo <= area <= area_hi):
                continue
            if not (dim_lo <= w <= dim_hi and dim_lo <= ht <= dim_hi):
                continue
            blobs.append(
                Blob(
                    float(centroids[i][0]) + ox,
                    float(centroids[i][1]) + oy,
                    int(area),
                    spec.name,
                )
            )
    return blobs


# --------------------------------------------------------------------------- #
# Robots
# --------------------------------------------------------------------------- #
def foreground(
    work: np.ndarray,
    background: np.ndarray,
    inside: np.ndarray,
    params: RobotParams,
) -> np.ndarray:
    """Moving-object mask for a cropped frame against a cropped background."""
    diff = cv2.absdiff(work, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    fg = ((gray > params.fg_threshold) & (inside > 0)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return fg


def colour_signature(
    work: np.ndarray, fg: np.ndarray, cx: int, cy: int, robot_size: int
) -> np.ndarray:
    """Colour histogram of one robot, in work-window coordinates.

    Eight hue bins over the colourful pixels plus four brightness bins, each
    normalised on its own so a robot that is mostly dark still gets a usable
    hue profile.
    """
    half = robot_size // 2
    y0, y1 = max(0, cy - half), min(work.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(work.shape[1], cx + half)
    patch = work[y0:y1, x0:x1]
    mask = fg[y0:y1, x0:x1] > 0
    sig = np.zeros(SIGNATURE_BINS, np.float32)
    if patch.size == 0 or not mask.any():
        return sig

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    colourful = mask & (s > 60) & (v > 40)
    if colourful.any():
        hue = np.clip((h[colourful].astype(np.int32) * 8) // 180, 0, 7)
        sig[:8] = np.bincount(hue, minlength=8)
        sig[:8] /= sig[:8].sum()
    bright = np.clip((v[mask].astype(np.int32) * 4) // 256, 0, 3)
    sig[8:] = np.bincount(bright, minlength=4)
    sig[8:] /= sig[8:].sum()
    return sig


def detect_robots(
    fg: np.ndarray,
    work: np.ndarray,
    inside: np.ndarray,
    geom: FieldGeometry,
    params: RobotParams,
    limit: int = N_ROBOTS + 2,
) -> List[RobotCandidate]:
    """Find robot-sized clumps of foreground.

    Connected components fuse robots that touch each other or the field wall,
    so instead of trusting component extents we slide a fixed robot-sized box
    over the foreground and take the strongest responses.  The footprint is
    known (a legal FTC robot is 18 inches on a 12 foot field), so this stays
    stable when two robots collide.

    Peaks whose centre falls outside the playing floor are dropped: the box
    filter reaches half a robot past the field mask, which otherwise conjures
    candidates out of the crowd standing behind the wall.

    Each accepted peak *consumes* the foreground underneath it before the next
    one is sought.  Suppressing a fixed radius instead would force a choice
    between two peaks landing on one robot and two touching robots collapsing
    into one; taking the pixels away makes that choice unnecessary, because a
    robot that has already been claimed has nothing left to peak on.
    """
    ox, oy, _, _ = geom.work_roi
    size = geom.robot_size
    half = size // 2
    height, width = fg.shape
    remaining = fg.astype(np.float32)
    density = cv2.boxFilter(remaining, -1, (size, size), normalize=True)
    density[inside == 0] = 0.0

    out: List[RobotCandidate] = []
    for _ in range(limit):
        _, max_val, _, max_loc = cv2.minMaxLoc(density)
        if max_val < params.min_fill:
            break
        cx, cy = int(max_loc[0]), int(max_loc[1])
        out.append(
            RobotCandidate(
                float(cx) + ox,
                float(cy) + oy,
                float(max_val),
                colour_signature(work, fg, cx, cy, size),
            )
        )

        # Take this robot's pixels away, then refresh the density only where
        # removing them can matter: a box filter of side `size` reaches half a
        # robot in every direction.
        remaining[max(0, cy - half): cy + half, max(0, cx - half): cx + half] = 0.0
        ry0, ry1 = max(0, cy - 2 * size), min(height, cy + 2 * size)
        rx0, rx1 = max(0, cx - 2 * size), min(width, cx + 2 * size)
        patch = cv2.boxFilter(remaining[ry0:ry1, rx0:rx1], -1, (size, size),
                              normalize=True)
        # The patch is only wrong where its edge is artificial.  Where it lines
        # up with the real frame edge the same border padding applies as in the
        # full-frame filter, so those rows and columns can be copied too.
        iy0 = ry0 + half if ry0 > 0 else 0
        iy1 = ry1 - half if ry1 < height else height
        ix0 = rx0 + half if rx0 > 0 else 0
        ix1 = rx1 - half if rx1 < width else width
        density[iy0:iy1, ix0:ix1] = patch[iy0 - ry0: iy1 - ry0, ix0 - rx0: ix1 - rx0]
        density[inside == 0] = 0.0
    return out
