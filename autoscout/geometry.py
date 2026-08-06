"""Working out where the field is, so one config fits any camera framing.

Every field-relative region the pipeline needs -- the playing floor, the goal
mouths, the human-player chutes, how big a robot is in pixels -- is a fixed
proportion of the field.  Measuring the floor once per video and deriving the
rest from it means a broadcast that frames the field differently, or shoots a
different field at the same event, does not need a new config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import Box, FieldProportions, DEFAULT_PROPORTIONS

#: Floor width, in pixels, that the shipped thresholds were tuned against.
REFERENCE_FLOOR_WIDTH = 637.0


@dataclass(frozen=True)
class FieldGeometry:
    """Everything downstream needs to know about where things are."""

    floor: Box                     # the bare playing surface
    field_poly: np.ndarray         # floor with the goal corners cut off
    robot_poly: np.ndarray         # field_poly, widened for perspective lean
    ball_roi: Box                  # floor plus the band above the wall
    work_roi: Box                  # bounding window for all per-pixel work
    chutes: Tuple[Box, ...]        # human-player artifact columns
    goals: Dict[str, Box]          # where a made shot lands
    robot_size: int                # chassis footprint in px
    scale: float                   # relative to REFERENCE_FLOOR_WIDTH

    @property
    def width(self) -> int:
        return self.floor[2] - self.floor[0]

    @property
    def height(self) -> int:
        return self.floor[3] - self.floor[1]

    def describe(self) -> str:
        x0, y0, x1, y1 = self.floor
        return (
            f"floor x{x0}-{x1} y{y0}-{y1} ({self.width}x{self.height} px), "
            f"robot {self.robot_size} px, scale {self.scale:.2f}"
        )


def _longest_span(profile: np.ndarray, frac: float) -> Tuple[int, int]:
    """Longest stretch where coverage stays above ``frac`` of the peak."""
    threshold = profile.max() * frac
    best = (0, 0)
    start: Optional[int] = None
    for i, value in enumerate(profile):
        if value >= threshold:
            if start is None:
                start = i
        elif start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if start is not None and len(profile) - start > best[1] - best[0]:
        best = (start, len(profile))
    return best


def floor_mask(background: np.ndarray, props: FieldProportions) -> np.ndarray:
    """Largest region that looks like bare playing surface.

    The field is a big slab of unsaturated grey; the goals are saturated red and
    blue, the surroundings are dark carpet and the scoreboard is a bar across the
    bottom.  That is enough to isolate it without knowing where it is.
    """
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)
    m = ((s < props.floor_max_saturation)
         & (v > props.floor_min_value)
         & (v < props.floor_max_value)).astype(np.uint8)

    h_img = background.shape[0]
    m[int(h_img * props.ignore_below):, :] = 0      # scoreboard bar
    m[: int(h_img * props.ignore_above), :] = 0

    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    best, best_area = 0, 0
    for i in range(1, count):
        _, _, w, h, area = stats[i]
        if area > best_area and w > 300 and h > 300:
            best, best_area = i, area
    if not best:
        raise RuntimeError(
            "could not find the playing floor; pass an explicit geometry or "
            "check that the video is an overhead view of the field"
        )
    return (labels == best).astype(np.uint8)


def calibrate(
    background: np.ndarray, props: FieldProportions = DEFAULT_PROPORTIONS
) -> FieldGeometry:
    """Measure the field in one background plate and derive every region from it."""
    mask = floor_mask(background, props)
    x0, x1 = _longest_span(mask.sum(axis=0).astype(np.float64), props.edge_fraction)
    y0, y1 = _longest_span(mask.sum(axis=1).astype(np.float64), props.edge_fraction)
    if x1 - x0 < 200 or y1 - y0 < 200:
        raise RuntimeError("the detected playing floor is implausibly small")

    h_img, w_img = background.shape[:2]
    w = float(x1 - x0)
    h = float(y1 - y0)

    def clampx(v: float) -> int:
        return int(round(min(max(v, 0), w_img)))

    def clampy(v: float) -> int:
        return int(round(min(max(v, 0), h_img)))

    cut = props.corner_cut * w
    field_poly = np.array(
        [
            (clampx(x0), clampy(y0 + cut)),
            (clampx(x0 + cut), clampy(y0)),
            (clampx(x1 - cut), clampy(y0)),
            (clampx(x1), clampy(y0 + cut)),
            (clampx(x1), clampy(y1)),
            (clampx(x0), clampy(y1)),
        ],
        dtype=np.int32,
    )

    margin = props.robot_margin * w
    centre = field_poly.mean(axis=0)
    robot_poly = np.array(
        [
            (
                clampx(px + (margin if px > centre[0] else -margin)),
                clampy(py + (margin if py > centre[1] else -margin)),
            )
            for px, py in field_poly
        ],
        dtype=np.int32,
    )

    ball_roi = (
        clampx(x0 - props.ball_margin_x * w),
        clampy(y0 - props.ball_margin_top * h),
        clampx(x1 + props.ball_margin_x * w),
        clampy(y1 + props.ball_margin_bottom * h),
    )
    goals = {
        "blue": (
            clampx(x0 - props.goal_outer * w),
            clampy(y0 - props.goal_top * h),
            clampx(x0 + props.goal_inner * w),
            clampy(y0 + props.goal_bottom * h),
        ),
        "red": (
            clampx(x1 - props.goal_inner * w),
            clampy(y0 - props.goal_top * h),
            clampx(x1 + props.goal_outer * w),
            clampy(y0 + props.goal_bottom * h),
        ),
    }
    chute_y0 = clampy(y0 + props.chute_top * h)
    chute_y1 = clampy(y1 + props.ball_margin_bottom * h)
    chutes = (
        (clampx(x0 - props.chute_outer * w), chute_y0,
         clampx(x0 + props.chute_inner * w), chute_y1),
        (clampx(x1 - props.chute_inner * w), chute_y0,
         clampx(x1 + props.chute_outer * w), chute_y1),
    )

    pad = 4
    work_roi = (
        clampx(min(ball_roi[0], robot_poly[:, 0].min()) - pad),
        clampy(min(ball_roi[1], robot_poly[:, 1].min()) - pad),
        clampx(max(ball_roi[2], robot_poly[:, 0].max()) + pad),
        clampy(max(ball_roi[3], robot_poly[:, 1].max()) + pad),
    )

    return FieldGeometry(
        floor=(clampx(x0), clampy(y0), clampx(x1), clampy(y1)),
        field_poly=field_poly,
        robot_poly=robot_poly,
        ball_roi=ball_roi,
        work_roi=work_roi,
        chutes=chutes,
        goals=goals,
        robot_size=int(round(props.robot_fraction * w)),
        scale=w / REFERENCE_FLOOR_WIDTH,
    )
