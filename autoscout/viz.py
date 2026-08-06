"""Debug rendering: an annotated copy of the match so results can be eyeballed."""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List, Optional

import cv2

from .overlay import AUTO, TELEOP
from .pipeline import MatchResult

ALLIANCE_BGR = {"red": (60, 60, 235), "blue": (235, 150, 60)}
TRAIL_BGR = {"purple": (200, 60, 190), "green": (60, 230, 90)}


def _clock(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def _transcode_h264(path: str, crf: int = 26) -> bool:
    """Re-encode in place to H.264, if ffmpeg is available.

    OpenCV can only write the ancient MPEG-4 Part 2 codec, which most browsers
    and phones refuse to play and which runs about ten times the size for the
    same picture.  This is a debug artifact people actually watch, so it is
    worth the extra pass -- but it stays optional, since ffmpeg is not a
    dependency of anything else here.  The quality target is set for a debug
    overlay of boxes and text, not for archival footage.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    tmp = path + ".h264.mp4"
    try:
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", path,
             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp],
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, path)
    return True


def render(
    result: MatchResult,
    video_path: str,
    out_path: str,
    scale: float = 0.6,
    trail_frames: int = 45,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    first = min(s.start_frame for s in result.spans)
    last = max(s.end_frame for s in result.spans)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), result.fps, (w, h)
    )

    geom = result.geometry
    alliance_of = {s.robot_id: s.alliance for s in result.robots}
    by_launch: Dict[int, List] = {}
    for shot in result.shots:
        by_launch.setdefault(shot.launch_frame, []).append(shot)

    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    running: Dict[int, Dict[str, int]] = {
        rid: {AUTO: 0, TELEOP: 0} for rid in alliance_of
    }
    made_running: Dict[int, Dict[str, int]] = {
        rid: {AUTO: 0, TELEOP: 0} for rid in alliance_of
    }

    for idx in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break

        cv2.polylines(frame, [geom.field_poly], True, (0, 200, 200), 1)
        for name, (x0, y0, x1, y1) in geom.goals.items():
            cv2.rectangle(frame, (x0, y0), (x1, y1), ALLIANCE_BGR[name], 2)
            cv2.putText(
                frame, f"{name} goal", (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ALLIANCE_BGR[name], 2,
            )
        for x0, y0, x1, y1 in geom.chutes:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (110, 110, 110), 1)

        for rid, hist in result.robot_tracks.items():
            pos = hist.get(idx)
            if pos is None:
                continue
            x, y = int(pos[0]), int(pos[1])
            colour = ALLIANCE_BGR[alliance_of.get(rid, "red")]
            r = geom.robot_size // 2
            cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), colour, 2)
            cv2.putText(
                frame, f"R{rid}", (x - r, y - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2,
            )

        for shot in result.shots:
            if not (shot.launch_frame <= idx <= shot.end_frame + trail_frames):
                continue
            p0 = tuple(int(v) for v in shot.launch_xy)
            p1 = tuple(int(v) for v in shot.end_xy)
            colour = TRAIL_BGR[shot.color]
            cv2.arrowedLine(frame, p0, p1, colour, 2, tipLength=0.06)
            cv2.circle(frame, p1, 12, (0, 255, 0) if shot.made else (0, 0, 255), 2)

        for shot in by_launch.get(idx, []):
            if shot.robot_id in running and shot.period in (AUTO, TELEOP):
                running[shot.robot_id][shot.period] += 1
                if shot.made:
                    made_running[shot.robot_id][shot.period] += 1

        period = next(
            (s.name for s in result.spans if s.contains(idx)), "off"
        )
        panel_y = 40
        cv2.putText(
            frame,
            f"{period.upper()}  {_clock(result.clock_at(idx))}",
            (40, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3,
        )
        for rid in sorted(running):
            panel_y += 34
            a = running[rid][AUTO] + running[rid][TELEOP]
            m = made_running[rid][AUTO] + made_running[rid][TELEOP]
            cv2.putText(
                frame, f"R{rid} {alliance_of[rid]:<4} shots {a:3d}  made {m:3d}",
                (40, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                ALLIANCE_BGR[alliance_of[rid]], 2,
            )

        writer.write(cv2.resize(frame, (w, h)))

    writer.release()
    cap.release()
    _transcode_h264(out_path)


def calibration_sheet(video_path: str, out_path: str, at_second: float = 60.0) -> str:
    """Dump one frame with every measured and configured region drawn on it."""
    import numpy as np

    from . import detect, overlay
    from .config import (
        MATCH_TITLE_BOX,
        PERIOD_ICON_BOX,
        SCORE_BOXES,
        TIMER_BOX,
        DEFAULT_PROPORTIONS,
    )
    from .geometry import calibrate

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    background = detect.build_background(
        cap, np.linspace(total * 0.05, total * 0.85, 40).astype(int)
    )
    geom = calibrate(background, DEFAULT_PROPORTIONS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(at_second * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read {video_path} at {at_second}s")

    def box(b, colour, text):
        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colour, 2)
        cv2.putText(frame, text, (int(b[0]), max(14, int(b[1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    box(geom.floor, (255, 255, 255), "floor (measured)")
    cv2.polylines(frame, [geom.field_poly], True, (0, 255, 255), 2)
    box(geom.ball_roi, (255, 200, 0), "ball roi")
    for i, c in enumerate(geom.chutes):
        box(c, (160, 160, 160), f"chute{i}")
    for name, g in geom.goals.items():
        box(g, ALLIANCE_BGR[name], f"goal:{name}")
    box(TIMER_BOX, (255, 255, 255), "clock")
    box(PERIOD_ICON_BOX, (255, 255, 255), "period icon")
    box(MATCH_TITLE_BOX, (0, 255, 0), "match title")
    for name, b in SCORE_BOXES.items():
        box(b, (0, 165, 255), f"score:{name}")
    for plate in overlay.find_counter_plates(frame):
        if plate:
            box(plate, (200, 0, 200), "counters")

    cv2.imwrite(out_path, frame)
    return geom.describe()
