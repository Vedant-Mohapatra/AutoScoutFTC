#!/usr/bin/env python3
"""Command line front end for AutoScoutFTC.

    python scout.py analyse Videos/match.mp4 --out out/
    python scout.py analyse Videos/match.mp4 --out out/ --debug-video
    python scout.py calibrate Videos/match.mp4 --out out/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

from autoscout import Config, analyse
from autoscout.cache import Cache
from autoscout import report as report_mod
from autoscout import viz


def _progress_printer():
    state = {"stage": "", "last": 0.0}

    def emit(stage: str, frac: float) -> None:
        now = time.time()
        if stage != state["stage"]:
            if state["stage"]:
                sys.stderr.write("\n")
            state["stage"] = stage
            state["last"] = 0.0
        if now - state["last"] < 0.25 and frac < 1.0:
            return
        state["last"] = now
        width = 30
        done = int(width * min(1.0, max(0.0, frac)))
        sys.stderr.write(f"\r  {stage:<12} [{'#' * done}{'.' * (width - done)}] {frac * 100:5.1f}%")
        sys.stderr.flush()

    return emit


def _parse_teams(values: Optional[List[str]]) -> dict:
    teams: dict = {}
    for item in values or []:
        alliance, _, numbers = item.partition("=")
        alliance = alliance.strip().lower()
        if alliance not in ("red", "blue") or not numbers:
            raise SystemExit(f"--team expects red=1234,5678 (got {item!r})")
        teams[alliance] = [int(n) for n in numbers.split(",") if n.strip()]
    return teams


def _parse_robot_teams(values: Optional[List[str]]) -> dict:
    out: dict = {}
    for item in values or []:
        rid, _, number = item.partition("=")
        if not number.strip().isdigit() or not rid.strip().isdigit():
            raise SystemExit(f"--robot-team expects 0=18264 (got {item!r})")
        out[int(rid)] = int(number)
    return out


def cmd_analyse(args: argparse.Namespace) -> int:
    cfg = Config()
    cfg.teams = _parse_teams(args.team)
    cfg.robot_teams = _parse_robot_teams(args.robot_team)

    os.makedirs(args.out, exist_ok=True)
    cache = Cache(None if args.no_cache else os.path.join(args.out, ".cache"))

    started = time.time()
    result = analyse(
        args.video,
        cfg,
        progress=_progress_printer() if not args.quiet else (lambda *_: None),
        cache=cache,
    )
    if not args.quiet:
        sys.stderr.write("\n\n")

    text = report_mod.summary(result)
    print(text)
    print(f"\nAnalysed in {time.time() - started:.0f}s")

    # One directory per match, named from the caption on the scoreboard, so a
    # season's worth of videos lands in a browsable tree instead of one heap.
    match_dir = os.path.join(args.out, result.label.slug())
    os.makedirs(match_dir, exist_ok=True)
    paths = {
        "robots.csv": os.path.join(match_dir, "robots.csv"),
        "shots.csv": os.path.join(match_dir, "shots.csv"),
        "match.json": os.path.join(match_dir, "match.json"),
        "summary.txt": os.path.join(match_dir, "summary.txt"),
    }
    report_mod.write_csv(result, paths["robots.csv"])
    report_mod.write_shots_csv(result, paths["shots.csv"])
    report_mod.write_json(result, paths["match.json"])
    with open(paths["summary.txt"], "w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    print("\nWrote:")
    for path in paths.values():
        print(f"  {path}")

    if args.debug_video:
        out_video = os.path.join(match_dir, "debug.mp4")
        print(f"\nRendering debug video -> {out_video}")
        viz.render(result, args.video, out_video, scale=args.debug_scale)
        print("  done")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    path = os.path.join(args.out, f"{stem}_calibration.png")
    measured = viz.calibration_sheet(args.video, path, args.at)
    print(f"Measured {measured}")
    print(f"Wrote {path}")
    print("The field is measured from the video, so this is a check rather than a "
          "setup step; if a region looks wrong, adjust FieldProportions in "
          "autoscout/config.py.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scout", description="Scout ball-shooting robots from FTC match video."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyse", aliases=["analyze"], help="run the full analysis")
    a.add_argument("video")
    a.add_argument("--out", default="out", help="output directory (default: out)")
    a.add_argument("--team", action="append",
                   help="override team numbers, e.g. --team red=18264,19639")
    a.add_argument("--robot-team", action="append",
                   help="bind a tracked robot id to a team, e.g. --robot-team 0=15083 "
                        "(watch the --debug-video to see which id is which robot)")
    a.add_argument("--debug-video", action="store_true",
                   help="also render an annotated copy of the match")
    a.add_argument("--debug-scale", type=float, default=0.6)
    a.add_argument("--no-cache", action="store_true",
                   help="ignore the decode cache under <out>/.cache")
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(func=cmd_analyse)

    c = sub.add_parser("calibrate", help="dump a frame with all regions drawn on it")
    c.add_argument("video")
    c.add_argument("--out", default="out")
    c.add_argument("--at", type=float, default=60.0, help="timestamp in seconds")
    c.set_defaults(func=cmd_calibrate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
