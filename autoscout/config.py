"""Tunable geometry, colours and thresholds for the FTC DECODE overhead broadcast view.

Every pixel coordinate here is in the 1920x1080 broadcast frame.  If you feed a
video with a different framing, run ``python scout.py calibrate <video>`` to dump
an annotated frame and adjust the values below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

Box = Tuple[int, int, int, int]          # x0, y0, x1, y1

FRAME_W, FRAME_H = 1920, 1080


# --------------------------------------------------------------------------- #
# Scoreboard overlay
# --------------------------------------------------------------------------- #
TIMER_BOX: Box = (872, 955, 1050, 1038)      # "2:30" / "0:53" digits
PERIOD_ICON_BOX: Box = (920, 908, 1000, 952)  # robot head / hand / gamepad

#: Rows the scoreboard bar occupies.  Value plates are located inside it rather
#: than pinned to fixed coordinates, because the layout shifts during a match
#: and differs between qualification and playoff broadcasts.
SCOREBOARD_BAR: Tuple[int, int] = (930, 1070)

#: The match caption, between the divider bar and the game logo.
MATCH_TITLE_BOX: Box = (1010, 820, 1665, 900)

# Alliance point totals (white text on saturated red / blue).
SCORE_BOXES: Dict[str, Box] = {
    "red": (656, 990, 836, 1070),
    "blue": (1100, 990, 1280, 1070),
}


# --------------------------------------------------------------------------- #
# Match structure (FTC DECODE)
# --------------------------------------------------------------------------- #
AUTO_SECONDS = 30
TELEOP_SECONDS = 120

#: The autonomous clock counts the *whole* match down (2:30 -> 2:00); teleop
#: restarts at 2:00.  The transition gets its own short countdown.
AUTO_CLOCK_START = 150
TELEOP_CLOCK_START = 120


# --------------------------------------------------------------------------- #
# Field geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldProportions:
    """How the field's parts relate to the playing floor.

    These are properties of the *game*, not of any particular camera, so they
    stay fixed while the pixel coordinates they produce follow whatever framing
    a broadcast happens to use.  Every fraction below is of the measured floor
    width (x) or height (y), and was checked against a hand-marked field.
    """

    # Isolating the floor in a background plate.
    floor_max_saturation: int = 55
    floor_min_value: int = 95
    floor_max_value: int = 235
    #: Rows to ignore, as a fraction of frame height: the scoreboard bar below
    #: and any letterboxing above.
    ignore_below: float = 0.815
    ignore_above: float = 0.018
    #: A row or column counts as field while it holds this share of the peak.
    edge_fraction: float = 0.55

    #: The goal structures cut this much off each top corner.
    corner_cut: float = 0.14

    #: Artifacts are tracked beyond the floor: sideways a little, and well above
    #: the top wall, where a shot is still visible as it drops into a goal.
    ball_margin_x: float = 0.12
    ball_margin_top: float = 0.16
    ball_margin_bottom: float = 0.04

    #: Goal mouths, measured out from the top-left and top-right floor corners.
    goal_outer: float = 0.11
    goal_inner: float = 0.16
    goal_top: float = 0.15
    goal_bottom: float = 0.12

    #: Human-player artifact columns hugging the side walls.
    chute_outer: float = 0.09
    chute_inner: float = 0.01
    chute_top: float = 0.14

    #: An 18" robot on a 12' field, plus the wall inset the floor measurement
    #: leaves out.
    robot_fraction: float = 0.144

    #: How far past the measured floor a robot centre may still be.  The camera
    #: is above the field but not infinitely far from it, so a tall robot
    #: pressed against a side wall leans outwards past the line where the floor
    #: visually ends.  Without this margin those robots stop being detected and
    #: their track drifts off.
    robot_margin: float = 0.06


DEFAULT_PROPORTIONS = FieldProportions()

N_ROBOTS = 4


# --------------------------------------------------------------------------- #
# Artifact colours (HSV, OpenCV ranges: H 0-179, S/V 0-255)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColorRange:
    name: str
    h: Tuple[int, int]
    s_min: int
    v_min: int


#: Deliberately generous.  A launched artifact is motion-blurred against the
#: floor, which washes out its saturation, and missing it for even two frames
#: breaks a trajectory in half.  The bare floor sits at S~17, so there is plenty
#: of headroom before grey starts registering as purple, and the speed filter in
#: ``shots.py`` throws out whatever else sneaks in.
ARTIFACT_COLORS: Tuple[ColorRange, ...] = (
    ColorRange("purple", (133, 160), 45, 35),
    ColorRange("green", (60, 95), 70, 55),
)

BALL_AREA = (60, 2000)          # px^2
BALL_DIM = (8, 70)              # bbox side length, px


# --------------------------------------------------------------------------- #
# Tracking / shot-detection thresholds
# --------------------------------------------------------------------------- #
@dataclass
class ShotParams:
    """Everything the shot detector is allowed to argue about.

    The defaults were tuned on a championship-level DECODE match by comparing
    the detected makes at each goal against the alliance totals the scoreboard
    reported (48 vs 64 and 99 vs 97).  They are a starting point, not a law --
    if a video is framed differently or shot from further away, expect to move
    ``min_speed_px_per_frame`` and ``min_displacement_px`` with the scale.
    """

    #: Frame-to-frame association gate for artifact blobs, plus how much it
    #: widens per px/frame of the track's current speed.
    assoc_gate_px: float = 26.0
    gate_speed_gain: float = 1.6
    #: Consecutive missed frames before an artifact track is closed.
    max_miss: int = 5

    #: A flight is a run of samples above this speed; it must also go somewhere
    #: and go up-screen, since both goals sit at the top of the frame.  Robots
    #: that park next to their goal produce short, fast flights, so the distance
    #: bar has to stay low or half their shooting disappears.
    min_points: int = 4
    min_speed_px_per_frame: float = 7.0
    min_displacement_px: float = 60.0
    min_rise_px: float = 40.0

    #: Frames of constant-velocity extrapolation allowed past the last sighting
    #: when deciding whether an artifact went in.  Artifacts are usually lost a
    #: frame or two before they disappear into the goal, but too much rope here
    #: credits shots that actually fell short.
    goal_lookahead_frames: int = 6

    #: How far back along the initial velocity we look for the shooter.
    backtrack_frames: int = 6
    #: Max distance from the back-projected launch point to a robot centre.
    attribution_radius_px: float = 165.0


@dataclass
class RobotParams:
    fg_threshold: int = 34
    #: Minimum share of a robot-sized box that must be foreground for a peak to
    #: count as a robot.
    min_fill: float = 0.22
    #: How hard colour similarity pulls the assignment, in px-equivalents.
    appearance_weight: float = 90.0
    #: Two robots may touch, but their centres cannot coincide.
    min_separation_px: float = 72.0
    #: Max plausible robot travel between frames.
    assoc_gate_px: float = 62.0
    #: A robot must look at least this solid to be worth seeding an identity on.
    seed_min_fill: float = 0.34
    #: How long to hold out for a clean seeding frame before taking what we have.
    seed_patience_frames: int = 240
    #: Frames a track may coast before it is allowed to jump to a spare robot.
    recover_after: int = 24
    #: A spare detection must look this solidly like a robot to be jumped to.
    recover_min_fill: float = 0.34
    #: ...and its colours must be at least this close to the lost track's.
    recover_max_signature_distance: float = 0.55
    #: Frames used to build the median background plate.
    bg_samples: int = 80


@dataclass
class Config:
    shot: ShotParams = field(default_factory=ShotParams)
    robot: RobotParams = field(default_factory=RobotParams)
    proportions: FieldProportions = field(default_factory=FieldProportions)
    #: Measured from the video unless pinned here.
    geometry: Optional[object] = None
    #: Which goal each alliance scores in.  Verified against the scoreboard at
    #: run time; override here if a broadcast mirrors the field.
    goal_of_alliance: Dict[str, str] = field(
        default_factory=lambda: {"red": "red", "blue": "blue"}
    )
    #: Optional manual team numbers, e.g. {"red": [18264, 19639], ...}.
    teams: Dict[str, List[int]] = field(default_factory=dict)
    #: Optional binding of tracked robot ids to team numbers.  Nothing in the
    #: video links a chassis to a plate on the scoreboard, so this can only come
    #: from a human who watched the match.
    robot_teams: Dict[int, int] = field(default_factory=dict)


DEFAULT = Config()
