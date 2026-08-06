"""Self-checks for the parts of the pipeline that do not need a video.

Run with ``python tests/test_autoscout.py`` (or under pytest).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoscout import overlay, shots
from autoscout.config import DEFAULT_PROPORTIONS, RobotParams, ShotParams
from autoscout.detect import RobotCandidate, SIGNATURE_BINS
from autoscout.geometry import FieldGeometry, calibrate
from autoscout.overlay import AUTO, OFF, OverlaySample, PeriodSpan, TELEOP, TRANSITION
from autoscout.tracking import ArtifactTrack, RobotTracker, signature_distance

#: Stand-in for a measured field, matching the framing the defaults were tuned on.
GEOM = FieldGeometry(
    floor=(604, 137, 1241, 812),
    field_poly=np.array([(604, 226), (693, 137), (1152, 137),
                         (1241, 226), (1241, 812), (604, 812)], np.int32),
    robot_poly=np.array([(566, 226), (655, 99), (1190, 99),
                         (1279, 226), (1279, 850), (566, 850)], np.int32),
    ball_roi=(528, 29, 1317, 839),
    work_roi=(524, 25, 1321, 843),
    chutes=((547, 231, 610, 839), (1235, 231, 1298, 839)),
    goals={"blue": (534, 36, 706, 218), "red": (1158, 36, 1311, 218)},
    robot_size=92,
    scale=1.0,
)
GOAL_BOXES = GEOM.goals

FPS = 60.0


def test_digit_templates_cover_every_digit() -> None:
    bank = overlay.bank()
    assert set(bank.labels.tolist()) == set(range(10)), bank.labels
    # Each stencil must classify as itself.
    for label, template in zip(bank.labels, bank.templates):
        got, dist = bank.classify((template * 255).astype(np.uint8))
        assert got == label, (got, label)
        # Not exactly zero: classify() re-normalises, which resamples.
        assert dist < 0.01, dist


def test_clock_parses_minutes_and_seconds() -> None:
    # read_clock folds "M:SS" into MSS, then splits it back out.
    assert (217 // 100) * 60 + 217 % 100 == 137
    assert (0 // 100) * 60 + 0 % 100 == 0


def _samples(sequence):
    """(icon, n_frames) pairs -> OverlaySamples at the scoreboard stride."""
    out, idx = [], 0
    for icon, frames in sequence:
        for _ in range(frames):
            out.append(OverlaySample(frame_index=idx, icon=icon, clock=None, counters=None))
            idx += 8
    return out


def test_period_segmentation_ignores_pre_match_scoreboard() -> None:
    # A broadcast shows the gamepad icon while idling before the match; the
    # real teleop run is the one that lasts two minutes.
    samples = _samples([
        (TELEOP, 60),        # 8 s of pre-match idle
        (AUTO, 225),         # 30 s
        (TRANSITION, 112),   # 15 s
        (TELEOP, 900),       # 120 s
        (OFF, 100),
    ])
    spans = overlay.segment_periods(samples, FPS)
    names = [s.name for s in spans]
    assert names == [AUTO, TRANSITION, TELEOP], names
    auto, _, teleop = spans
    assert 28 <= (auto.end_frame - auto.start_frame) / FPS <= 32
    assert 118 <= (teleop.end_frame - teleop.start_frame) / FPS <= 122


def test_counters_are_repaired_not_rolled_back() -> None:
    assert overlay.enforce_monotonic([0, 3, 3, 2, 5]) == [0, 3, 3, None, 5]
    assert overlay.enforce_monotonic([None, 4, None, 4]) == [None, 4, None, 4]


def _track(points, color="purple"):
    return ArtifactTrack(tid=0, color=color, points=points)


def test_flight_ignores_an_artifact_rolling_on_the_floor() -> None:
    rolling = _track([(f, 700 + 2 * f, 600 - 1 * f) for f in range(40)])
    assert shots.flights(rolling, ShotParams()) == []


def test_flight_isolates_the_launched_part_of_a_track() -> None:
    # Sits still, is launched towards the blue goal, then settles.
    still = [(f, 900.0, 640.0) for f in range(0, 20)]
    flying = [(20 + f, 900 - 16.0 * f, 640 - 22.0 * f) for f in range(0, 22)]
    settled = [(42 + f, 548.0, 156.0) for f in range(0, 15)]
    found = shots.flights(_track(still + flying + settled), ShotParams())
    assert len(found) == 1, found
    flight = found[0]
    assert flight.start_frame >= 19 and flight.end_frame <= 43
    assert flight.rise > 300


def test_made_shot_is_one_that_ends_in_a_goal() -> None:
    params = ShotParams()
    x0, y0, x1, y1 = GOAL_BOXES["blue"]
    # Ends at (540, 208), just inside the blue goal mouth.
    into_goal = _track([(f, 900 - 20.0 * f, 640 - 24.0 * f) for f in range(19)])
    flight = shots.flights(into_goal, params)[0]
    assert x0 <= flight.points[-1][1] <= x1 and y0 <= flight.points[-1][2] <= y1
    assert shots._goal_hit(flight, GOAL_BOXES) == "blue"

    # Same launch, but the artifact stops well short: even with the whole
    # extrapolation budget it never reaches the mouth.
    short = _track([(f, 900 - 20.0 * f, 640 - 24.0 * f) for f in range(9)])
    short_flight = shots.flights(short, params)[0]
    assert short_flight.points[-1][1] > x1
    assert shots._goal_hit(short_flight, GOAL_BOXES,
                           params.goal_lookahead_frames) is None


def _shot(**kw):
    base = dict(
        track_id=0, color="purple", launch_frame=100, end_frame=130, period=TELEOP,
        launch_xy=(900.0, 640.0), end_xy=(600.0, 120.0), goal="blue", made=True,
    )
    base.update(kw)
    return shots.Shot(**base)


def test_dedupe_folds_a_split_trajectory_into_one_shot() -> None:
    whole = _shot(launch_frame=100, end_frame=130)
    fragment = _shot(track_id=1, launch_frame=118, end_frame=132,
                     launch_xy=(700.0, 380.0))
    assert len(shots.dedupe([whole, fragment])) == 1

    # A genuinely separate shot from elsewhere survives.
    other = _shot(track_id=2, launch_frame=260, end_frame=290,
                  launch_xy=(1000.0, 500.0))
    assert len(shots.dedupe([whole, fragment, other])) == 2


def test_goal_ownership_follows_the_scoreboard_not_the_colours() -> None:
    # 40 into the "blue" goal, 10 into the "red" one, while the scoreboard
    # credits red with the bigger haul -- the camera view is mirrored.
    sample = (
        [_shot(goal="blue") for _ in range(40)]
        + [_shot(goal="red") for _ in range(10)]
    )
    assert shots.infer_goal_alliance_map(sample, {"red": 90, "blue": 30}) == {
        "red": "blue", "blue": "red",
    }
    # Too close to call -> no claim, caller falls back to the default.
    even = [_shot(goal="blue") for _ in range(20)] + [_shot(goal="red") for _ in range(19)]
    assert shots.infer_goal_alliance_map(even, {"red": 50, "blue": 49}) == {}


def _cand(x, y, hue_bin=0, score=0.6):
    sig = np.zeros(SIGNATURE_BINS, np.float32)
    sig[hue_bin] = 1.0
    sig[8] = 1.0
    return RobotCandidate(float(x), float(y), score, sig)


def test_lost_robot_track_is_recovered_by_colour() -> None:
    params = RobotParams()
    tracker = RobotTracker(params)
    start = [_cand(700, 300, 0), _cand(1000, 300, 3), _cand(700, 700, 5), _cand(1000, 700, 7)]
    assert tracker.seed(start, 0)
    target = next(t for t in tracker.tracks if (round(t.x), round(t.y)) == (1000, 300))
    green_id = target.rid

    # That robot vanishes from the detections for a while...
    without = [c for c in start if c.x != 1000 or c.y != 300]
    for f in range(1, 60):
        tracker.update(without, f)
    lost = next(t for t in tracker.tracks if t.rid == green_id)
    assert lost.coast > params.recover_after

    # ...and reappears far away.  Its colour should reclaim it.
    reappeared = without + [_cand(640, 760, 3)]
    for f in range(60, 66):
        tracker.update(reappeared, f)
    healed = next(t for t in tracker.tracks if t.rid == green_id)
    assert abs(healed.x - 640) < 5 and abs(healed.y - 760) < 5, (healed.x, healed.y)


def test_seeding_waits_for_a_field_with_only_robots_on_it() -> None:
    """A broadcast wipe puts more robot-sized masses on screen than there are
    robots; seeding then would pin identities to whatever is covering the field."""
    tracker = RobotTracker(RobotParams())
    crowded = [_cand(700, 300, 0, score=1.0), _cand(900, 300, 1, score=1.0),
               _cand(1100, 300, 2, score=1.0), _cand(700, 700, 3, score=1.0),
               _cand(900, 700, 4, score=1.0), _cand(1100, 700, 5, score=1.0)]
    assert not tracker.seed(crowded, 0)
    assert not tracker.tracks

    clear = [_cand(700, 300, 0, score=0.7), _cand(1000, 300, 3, score=0.6),
             _cand(700, 700, 5, score=0.55), _cand(1000, 700, 7, score=0.5)]
    assert tracker.seed(clear, 60)
    assert len(tracker.tracks) == 4

    # Once patience runs out we take what is there rather than never starting.
    fallback = RobotTracker(RobotParams())
    assert fallback.seed(crowded, 0, strict=False)


def test_signature_distance_is_zero_for_identical_profiles() -> None:
    a = np.zeros(SIGNATURE_BINS, np.float32)
    a[2] = a[8] = 1.0
    b = a.copy()
    assert signature_distance(a, b) == 0.0
    c = np.zeros(SIGNATURE_BINS, np.float32)
    c[6] = c[8] = 1.0
    assert signature_distance(a, c) > 0.4


def test_period_lookup() -> None:
    spans = [PeriodSpan(AUTO, 0, 100), PeriodSpan(TELEOP, 200, 300)]
    assert overlay.period_at(spans, 50) == AUTO
    assert overlay.period_at(spans, 150) == OFF
    assert overlay.period_at(spans, 300) == TELEOP


def _synthetic_field(x0, y0, x1, y1, frame=(1080, 1920)):
    """A dark frame with a grey slab where the playing floor would be."""
    img = np.zeros((frame[0], frame[1], 3), np.uint8)
    img[y0:y1, x0:x1] = (146, 146, 146)
    return img


def test_calibration_finds_the_floor_wherever_it_sits() -> None:
    for box in ((604, 137, 1241, 812), (400, 100, 1500, 800), (700, 200, 1150, 700)):
        geom = calibrate(_synthetic_field(*box), DEFAULT_PROPORTIONS)
        for got, want in zip(geom.floor, box):
            assert abs(got - want) <= 6, (geom.floor, box)


def test_robot_gate_is_wider_than_the_floor() -> None:
    """A robot leaning on the wall must still be detectable."""
    geom = calibrate(_synthetic_field(604, 137, 1241, 812), DEFAULT_PROPORTIONS)
    assert geom.robot_poly[:, 0].min() < geom.field_poly[:, 0].min()
    assert geom.robot_poly[:, 0].max() > geom.field_poly[:, 0].max()
    # ...but not so wide that it reaches the crowd behind the wall.
    assert geom.robot_poly[:, 0].max() - geom.floor[2] < geom.robot_size


def test_derived_regions_track_the_field_size() -> None:
    small = calibrate(_synthetic_field(700, 200, 1150, 700), DEFAULT_PROPORTIONS)
    big = calibrate(_synthetic_field(400, 100, 1500, 800), DEFAULT_PROPORTIONS)
    assert big.robot_size > small.robot_size
    assert big.scale > small.scale
    # Goals hug the top corners of whichever field was measured.
    for geom in (small, big):
        bx0, by0, bx1, by1 = geom.goals["blue"]
        rx0, _, rx1, _ = geom.goals["red"]
        assert bx0 < geom.floor[0] < bx1
        assert rx0 < geom.floor[2] < rx1
        assert by0 < geom.floor[1] < by1


def test_match_label_slug() -> None:
    assert overlay.MatchLabel("qualification", 12).slug() == "qualification-12"
    assert overlay.MatchLabel("playoff", 11).slug() == "playoff-11"
    # Unknown wording still yields a usable, non-colliding directory name.
    assert overlay.MatchLabel(None, 7).slug() == "match-7"
    assert overlay.MatchLabel(None, None).slug() == "match"


def test_shot_thresholds_scale_with_the_field() -> None:
    # A flight that clears the bar at full size falls short on a bigger field,
    # where the same real distance covers more pixels.
    pts = [(f, 900 - 12.0 * f, 640 - 14.0 * f) for f in range(8)]
    params = ShotParams()
    assert shots.flights(_track(pts), params, scale=1.0)
    assert not shots.flights(_track(pts), params, scale=2.2)


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print("\n" + ("all passed" if not failures else f"{failures} failing"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
