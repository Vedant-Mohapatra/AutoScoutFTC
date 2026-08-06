#!/usr/bin/env python3
"""Regenerate ``autoscout/assets/match_types.npz`` from labelled match videos.

The match-type word in the broadcast caption is matched as a whole shape, so
teaching the tool a new type ("practice", "semifinal", ...) means pointing this
script at one video of that type:

    python tools/build_match_types.py --add qualification=Videos/quals12.mp4 \
                                      --add playoff=Videos/playoff11.mp4

Existing entries are kept unless a new one uses the same label.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoscout import overlay


def prefix_bitmap(frame: np.ndarray):
    """The caption's leading words, as one normalised bitmap."""
    glyphs = overlay._title_glyphs(frame)
    if not glyphs:
        return None
    digit_bank = overlay.bank()
    for i, (_, _, _, _, mask) in enumerate(glyphs):
        normalised = overlay._normalise(mask)
        if normalised is None:
            continue
        _, dist = digit_bank.classify(normalised)
        if dist <= overlay._TITLE_DIGIT_DISTANCE:
            return overlay._word_bitmap(glyphs[:i]) if i else None
    return overlay._word_bitmap(glyphs)


def sample(video: str, count: int = 9):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    shots = []
    for idx in np.linspace(total * 0.15, total * 0.75, count).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        bm = prefix_bitmap(frame)
        if bm is not None:
            shots.append(bm.astype(np.float32))
    cap.release()
    if not shots:
        raise RuntimeError(f"no caption text found in {video}")
    # Median across frames shrugs off compression noise on any single one.
    return np.median(np.stack(shots), axis=0) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="append", required=True,
                    help="label=path/to/video.mp4")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "autoscout", "assets", "match_types.npz"))
    args = ap.parse_args()

    templates: Dict[str, np.ndarray] = {}
    if os.path.exists(args.out):
        old = np.load(args.out)
        for label, tpl in zip(old["labels"], old["tpl"]):
            templates[str(label)] = tpl.astype(np.float32)

    for item in args.add:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--add expects label=video (got {item!r})")
        templates[label.strip().lower()] = sample(path)
        print(f"learned {label.strip().lower():<16} from {os.path.basename(path)}")

    labels: List[str] = sorted(templates)
    np.savez(args.out,
             labels=np.array(labels),
             tpl=np.stack([templates[k] for k in labels]).astype(np.float32))
    print(f"wrote {args.out} with {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
