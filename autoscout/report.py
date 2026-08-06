"""Human- and spreadsheet-readable output."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

from .overlay import AUTO, TELEOP
from .pipeline import MatchResult
from .shots import makes_per_goal

ROBOT_FIELDS = [
    "robot_id",
    "alliance",
    "team",
    "auto_attempts",
    "auto_made",
    "teleop_attempts",
    "teleop_made",
    "total_attempts",
    "total_made",
    "accuracy",
    "tracked",
]


def _fmt_pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v * 100:.0f}%"


def _clock(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def robot_rows(result: MatchResult) -> List[Dict[str, object]]:
    rows = []
    for s in result.robots:
        rows.append(
            {
                "robot_id": s.robot_id,
                "alliance": s.alliance,
                "team": s.team if s.team is not None else "",
                "auto_attempts": s.attempts[AUTO],
                "auto_made": s.made[AUTO],
                "teleop_attempts": s.attempts[TELEOP],
                "teleop_made": s.made[TELEOP],
                "total_attempts": s.total_attempts,
                "total_made": s.total_made,
                "accuracy": "" if s.accuracy is None else round(s.accuracy, 3),
                "tracked": round(s.tracked, 3),
            }
        )
    return rows


def write_csv(result: MatchResult, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROBOT_FIELDS)
        writer.writeheader()
        writer.writerows(robot_rows(result))


def write_shots_csv(result: MatchResult, path: str) -> None:
    fields = [
        "shot_index",
        "robot_id",
        "period",
        "match_clock",
        "video_time_s",
        "artifact_color",
        "goal",
        "made",
        "launch_x",
        "launch_y",
        "end_x",
        "end_y",
        "attribution_px",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, shot in enumerate(sorted(result.shots, key=lambda s: s.launch_frame)):
            writer.writerow(
                {
                    "shot_index": i,
                    "robot_id": "" if shot.robot_id is None else shot.robot_id,
                    "period": shot.period,
                    "match_clock": _clock(result.clock_at(shot.launch_frame)),
                    "video_time_s": round(shot.launch_frame / result.fps, 3),
                    "artifact_color": shot.color,
                    "goal": shot.goal or "",
                    "made": int(shot.made),
                    "launch_x": round(shot.launch_xy[0], 1),
                    "launch_y": round(shot.launch_xy[1], 1),
                    "end_x": round(shot.end_xy[0], 1),
                    "end_y": round(shot.end_xy[1], 1),
                    "attribution_px": round(shot.attribution_px, 1),
                }
            )


def write_json(result: MatchResult, path: str) -> None:
    payload = {
        "match": {
            "kind": result.label.kind,
            "number": result.label.number,
            "slug": result.label.slug(),
        },
        "video": os.path.basename(result.video),
        "fps": result.fps,
        "field": {
            "floor": list(result.geometry.floor),
            "robot_size_px": result.geometry.robot_size,
            "scale": round(result.geometry.scale, 4),
        },
        "periods": [
            {
                "name": s.name,
                "start_s": round(s.start_frame / result.fps, 2),
                "end_s": round(s.end_frame / result.fps, 2),
                "duration_s": round((s.end_frame - s.start_frame) / result.fps, 2),
            }
            for s in result.spans
        ],
        "teams": result.teams,
        "final_score": result.final_score,
        "alliance_goal": result.alliance_goal,
        "scoreboard_artifacts": result.scoreboard,
        "robots": robot_rows(result),
        "warnings": result.warnings,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def summary(result: MatchResult) -> str:
    lines: List[str] = []
    lines.append(f"Match      : {result.label.describe()}")
    lines.append(f"Video      : {os.path.basename(result.video)}")
    lines.append(f"Frame rate : {result.fps:g} fps")
    lines.append(f"Field      : {result.geometry.describe()}")
    lines.append("")

    lines.append("Periods (from the on-screen timer)")
    for span in result.spans:
        a = span.start_frame / result.fps
        b = span.end_frame / result.fps
        lines.append(
            f"  {span.name:<11} video {a:7.2f}s -> {b:7.2f}s   ({b - a:5.1f}s)"
        )
    lines.append("")

    if result.teams:
        red = ", ".join(str(t) for t in result.teams.get("red", [])) or "?"
        blue = ", ".join(str(t) for t in result.teams.get("blue", [])) or "?"
        lines.append(f"Teams      : red {red}   |   blue {blue}")
    if result.final_score:
        r = result.final_score.get("red")
        b = result.final_score.get("blue")
        lines.append(f"Final score: red {r}   |   blue {b}")
    lines.append(
        "Goals      : "
        + ", ".join(f"{a} alliance -> {g} goal" for a, g in result.alliance_goal.items())
    )
    lines.append("")

    lines.append("Per-robot shooting (vision)")
    header = (
        f"  {'robot':<7}{'alliance':<10}{'team':<8}"
        f"{'auto a/m':<11}{'teleop a/m':<13}{'total a/m':<12}{'acc':<6}{'tracked':<8}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for s in result.robots:
        lines.append(
            f"  {s.robot_id:<7}{s.alliance:<10}{str(s.team or '-'):<8}"
            f"{f'{s.attempts[AUTO]}/{s.made[AUTO]}':<11}"
            f"{f'{s.attempts[TELEOP]}/{s.made[TELEOP]}':<13}"
            f"{f'{s.total_attempts}/{s.total_made}':<12}"
            f"{_fmt_pct(s.accuracy):<6}"
            f"{_fmt_pct(s.tracked):<8}"
        )
    lines.append(
        "  (a = attempts, m = made; 'tracked' is how much of the match this "
        "identity stayed locked on a robot)"
    )
    lines.append("")

    lines.append("Alliance artifacts scored, read from the scoreboard (exact)")
    for alliance in ("red", "blue"):
        board = result.scoreboard.get(alliance, {})
        lines.append(
            f"  {alliance:<5} auto {board.get(AUTO, '-'):>4}"
            f"   teleop {board.get(TELEOP, '-'):>4}"
            f"   total {board.get('total', '-'):>4}"
        )

    seen = makes_per_goal(result.shots)
    lines.append("")
    lines.append("Vision vs scoreboard, made shots (how much to trust the table above)")
    for alliance, goal in result.alliance_goal.items():
        truth = result.scoreboard.get(alliance, {}).get("total")
        got = seen.get(goal, 0)
        share = f"{got / truth * 100:3.0f}%" if truth else "  -"
        lines.append(
            f"  {alliance:<5} vision saw {got:3d} land in the {goal} goal, "
            f"scoreboard counted {truth if truth is not None else '-':>3}   ({share} found)"
        )

    if result.warnings:
        lines.append("")
        lines.append("Warnings")
        for w in result.warnings:
            lines.append(f"  ! {w}")
    return "\n".join(lines)
