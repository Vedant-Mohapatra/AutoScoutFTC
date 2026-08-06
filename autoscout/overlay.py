"""Reading the FTC broadcast scoreboard: match clock, period, artifact counters.

The overlay is a vector graphic re-rendered every frame, so template matching on
size-normalised glyphs is both exact and fast.  ``assets/digits.npz`` holds a
28x28 stencil per digit, harvested from the clock (the largest, cleanest text on
the bar) and reused for every numeric field.

Positions are found rather than assumed wherever the layout moves: the panel of
autonomous results appears part-way through a match and shoves the counters
sideways, and a playoff scoreboard carries three teams per alliance where a
qualification one carries two.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import (
    MATCH_TITLE_BOX,
    PERIOD_ICON_BOX,
    SCORE_BOXES,
    SCOREBOARD_BAR,
    TIMER_BOX,
    Box,
)

GLYPH = 28
WORD_W, WORD_H = 128, 24
_ASSETS = os.path.join(os.path.dirname(__file__), "assets")
_DIGITS = os.path.join(_ASSETS, "digits.npz")
_WORDS = os.path.join(_ASSETS, "match_types.npz")

AUTO, TRANSITION, TELEOP, OFF = "auto", "transition", "teleop", "off"


# --------------------------------------------------------------------------- #
# Digit templates
# --------------------------------------------------------------------------- #
class DigitBank:
    def __init__(self, path: str = _DIGITS):
        d = np.load(path)
        self.labels: np.ndarray = d["labels"]
        self.templates: np.ndarray = d["tpl"].astype(np.float32)

    def classify(self, glyph: np.ndarray) -> Tuple[int, float]:
        g = glyph.astype(np.float32) / 255.0
        dist = np.mean(np.abs(self.templates - g[None]), axis=(1, 2))
        i = int(np.argmin(dist))
        return int(self.labels[i]), float(dist[i])


_BANK: Optional[DigitBank] = None


def bank() -> DigitBank:
    global _BANK
    if _BANK is None:
        _BANK = DigitBank()
    return _BANK


def _normalise(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    crop = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1].astype(np.uint8) * 255
    h, w = crop.shape
    scale = (GLYPH - 4) / max(h, w)
    crop = cv2.resize(
        crop,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    out = np.zeros((GLYPH, GLYPH), np.uint8)
    y0 = (GLYPH - crop.shape[0]) // 2
    x0 = (GLYPH - crop.shape[1]) // 2
    out[y0: y0 + crop.shape[0], x0: x0 + crop.shape[1]] = crop
    return out


def read_number(
    frame: np.ndarray,
    box: Box,
    *,
    dark_text: bool = True,
    min_height_frac: float = 0.45,
    max_distance: float = 0.115,
) -> Optional[int]:
    """OCR a small integer out of ``box``.

    Returns ``None`` when the region does not contain a confidently readable
    number, which is how callers tell a value plate from a plate of tick marks.
    """
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = ((gray < 110) if dark_text else (gray > 190)).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    h_roi, w_roi = binary.shape
    glyphs: List[Tuple[int, np.ndarray]] = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if h < min_height_frac * h_roi or area < 40:
            continue
        if w > 0.85 * w_roi or not (0.18 <= w / h <= 1.2):
            continue
        if y <= 0 or y + h >= h_roi:      # clipped by the box -> panel border
            continue
        g = _normalise(labels[y: y + h, x: x + w] == i)
        if g is not None:
            glyphs.append((x, g))
    if not glyphs:
        return None

    glyphs.sort(key=lambda t: t[0])
    digits: List[int] = []
    for _, g in glyphs:
        d, dist = bank().classify(g)
        if dist > max_distance:
            return None
        digits.append(d)
    return int("".join(str(d) for d in digits))


# --------------------------------------------------------------------------- #
# Clock and period
# --------------------------------------------------------------------------- #
def read_clock(frame: np.ndarray) -> Optional[int]:
    """Match clock in seconds.  The overlay renders ``M:SS``; the colon is
    dropped by the glyph filter, so ``2:17`` arrives as ``217`` = 2*60 + 17."""
    raw = read_number(frame, TIMER_BOX)
    if raw is None:
        return None
    return (raw // 100) * 60 + raw % 100


def read_period_icon(frame: np.ndarray) -> str:
    """Classify the pictogram above the clock.

    A robot head means autonomous, an open palm means the hands-off transition,
    a gamepad means driver control, and a blank box means the match is over.
    They differ enough in aspect ratio that no template is needed.
    """
    x0, y0, x1, y1 = PERIOD_ICON_BOX
    roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    ink = (roi < 110).astype(np.uint8)
    if ink.mean() < 0.05:
        return OFF
    ys, xs = np.nonzero(ink)
    aspect = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    if aspect < 1.05:
        return TRANSITION
    if aspect < 1.45:
        return AUTO
    return TELEOP


# --------------------------------------------------------------------------- #
# Value plates
# --------------------------------------------------------------------------- #
@dataclass
class Counters:
    red_scored: int
    blue_scored: int
    red_arrow: int
    blue_arrow: int

    def scored(self, alliance: str) -> int:
        return self.red_scored if alliance == "red" else self.blue_scored


def _white_boxes(frame: np.ndarray) -> List[Box]:
    """Solid white value plates in the scoreboard bar.

    The bar is drawn as coloured alliance panels with white boxes punched into
    them, so thresholding for white and keeping the well-filled rectangles finds
    every value plate without knowing where any of them are.
    """
    top, bottom = SCOREBOARD_BAR
    strip = cv2.cvtColor(frame[top:bottom], cv2.COLOR_BGR2GRAY)
    mask = (strip > 185).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes: List[Box] = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if w < 45 or h < 25 or area < 0.55 * w * h:
            continue
        boxes.append((int(x), int(y) + top, int(x + w), int(y + h) + top))
    return sorted(boxes)


#: The artifact counters are a stacked pair of value boxes -- "passed" above,
#: "scored" below -- which the white-box pass sees as one tall, narrow plate.
_COUNTER_W = (65, 100)
_COUNTER_H = (80, 125)
#: Team plates are the wide ones towards the middle of the bar.  The width alone
#: does not separate them from the autonomous-results panel, so the real test is
#: that a plate holds a whole roster: an alliance never has fewer than two teams.
_TEAM_W = (120, 190)
_MIN_ROSTER = 2


def find_counter_plates(frame: np.ndarray) -> Tuple[Optional[Box], Optional[Box]]:
    """The red and blue artifact-counter plates, by shape and position.

    Both the qualification and playoff scoreboards slide these sideways once the
    autonomous results panel appears, so they are found rather than assumed.
    Their proportions are unlike anything else on the bar -- tall and narrow --
    and they are always the outermost plates of that shape.
    """
    candidates = [
        b for b in _white_boxes(frame)
        if _COUNTER_W[0] <= b[2] - b[0] <= _COUNTER_W[1]
        and _COUNTER_H[0] <= b[3] - b[1] <= _COUNTER_H[1]
    ]
    if len(candidates) < 2:
        return None, None
    return candidates[0], candidates[-1]


def _read_plate(frame: np.ndarray, plate: Box) -> Tuple[Optional[int], Optional[int]]:
    """Split a counter plate into its two stacked value boxes and read both."""
    x0, y0, x1, y1 = plate
    mid = (y0 + y1) // 2
    top = read_number(frame, (x0 - 2, y0 - 3, x1 + 2, mid + 2), min_height_frac=0.4)
    bottom = read_number(frame, (x0 - 2, mid - 2, x1 + 2, y1 + 3), min_height_frac=0.4)
    return top, bottom


def read_counters(frame: np.ndarray) -> Optional[Counters]:
    """Artifact tallies for both alliances."""
    red_plate, blue_plate = find_counter_plates(frame)
    if red_plate is None or blue_plate is None:
        return None
    red_arrow, red_scored = _read_plate(frame, red_plate)
    blue_arrow, blue_scored = _read_plate(frame, blue_plate)
    if None in (red_arrow, red_scored, blue_arrow, blue_scored):
        return None
    return Counters(
        red_scored=red_scored,
        blue_scored=blue_scored,
        red_arrow=red_arrow,
        blue_arrow=blue_arrow,
    )


def read_scores(frame: np.ndarray) -> Dict[str, Optional[int]]:
    return {
        alliance: read_number(frame, box, dark_text=False)
        for alliance, box in SCORE_BOXES.items()
    }


def _plate_rows(frame: np.ndarray, x0: int, x1: int) -> List[int]:
    """Team numbers stacked inside one alliance plate, top to bottom.

    A qualification alliance has two teams and a playoff alliance three, and the
    plate arrives sometimes as one merged white box and sometimes as several, so
    the rows are recovered from the digits themselves rather than from the boxes.
    """
    top, bottom = SCOREBOARD_BAR
    roi = cv2.cvtColor(frame[top:bottom, x0:x1], cv2.COLOR_BGR2GRAY)
    dark = (roi < 110).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    centres: List[Tuple[int, int]] = []
    for i in range(1, count):
        _, y, w, h, area = stats[i]
        if area < 40 or not (10 <= h <= 45) or w > 0.5 * (x1 - x0):
            continue
        centres.append((int(y) + int(h) // 2 + top, int(h)))
    if not centres:
        return []

    centres.sort()
    height = int(np.median([h for _, h in centres]))
    rows: List[List[int]] = [[centres[0][0]]]
    for cy, _ in centres[1:]:
        if cy - rows[-1][-1] > height:
            rows.append([cy])
        else:
            rows[-1].append(cy)

    out: List[int] = []
    for row in rows:
        cy = int(np.mean(row))
        value = read_number(
            frame, (x0, cy - height, x1, cy + height), min_height_frac=0.3
        )
        if value is not None and 1 <= value <= 99999:
            out.append(value)
    return out


def read_teams(frames: Iterable[np.ndarray]) -> Dict[str, List[int]]:
    """Team numbers, decided by majority vote over many frames.

    Individual reads occasionally fuse two touching digits; the plates never
    change during a match, so voting on the whole roster removes that noise.
    """
    votes: Dict[str, Dict[Tuple[int, ...], int]] = {"red": {}, "blue": {}}
    for frame in frames:
        plates = [
            b for b in _white_boxes(frame) if _TEAM_W[0] <= b[2] - b[0] <= _TEAM_W[1]
        ]
        if not plates:
            continue
        # A plate can arrive as one merged white box or as several stacked ones,
        # so group by column before reading the rows out of it.
        columns: Dict[int, Tuple[int, int]] = {}
        for x0, _, x1, _ in plates:
            key = x0 // 40
            lo, hi = columns.get(key, (x0, x1))
            columns[key] = (min(lo, x0), max(hi, x1))

        rosters = []
        for x0, x1 in sorted(columns.values()):
            found = tuple(_plate_rows(frame, x0, x1))
            if len(found) >= _MIN_ROSTER:
                rosters.append(found)
        if len(rosters) < 2:
            continue
        for alliance, found in (("red", rosters[0]), ("blue", rosters[-1])):
            votes[alliance][found] = votes[alliance].get(found, 0) + 1

    out: Dict[str, List[int]] = {"red": [], "blue": []}
    for alliance, tally in votes.items():
        if not tally:
            continue
        winner = max(tally, key=tally.get)
        if tally[winner] >= 3 and tally[winner] >= 0.4 * sum(tally.values()):
            out[alliance] = list(winner)
    return out


# --------------------------------------------------------------------------- #
# Match label
# --------------------------------------------------------------------------- #
class WordBank:
    """Whole-word stencils for the match-type text in the broadcast title.

    Matching a word as a single shape sidesteps letter segmentation entirely,
    which is why this is a small bank of words rather than an alphabet.
    """

    def __init__(self, path: str = _WORDS):
        if os.path.exists(path):
            data = np.load(path)
            self.labels: List[str] = [str(v) for v in data["labels"]]
            self.templates: np.ndarray = data["tpl"].astype(np.float32)
        else:
            self.labels = []
            self.templates = np.zeros((0, WORD_H, WORD_W), np.float32)

    def classify(
        self, bitmap: np.ndarray, max_distance: float = 0.09
    ) -> Optional[str]:
        if not self.labels:
            return None
        g = bitmap.astype(np.float32) / 255.0
        dist = np.mean(np.abs(self.templates - g[None]), axis=(1, 2))
        i = int(np.argmin(dist))
        return self.labels[i] if dist[i] <= max_distance else None


_WORD_BANK: Optional[WordBank] = None


def word_bank() -> WordBank:
    global _WORD_BANK
    if _WORD_BANK is None:
        _WORD_BANK = WordBank()
    return _WORD_BANK


def _title_glyphs(frame: np.ndarray):
    x0, y0, x1, y1 = MATCH_TITLE_BOX
    roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    bw = (roi > 150).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 25 or not (8 <= h <= 40) or w > 60:
            continue
        out.append((int(x), int(y), int(w), int(h), labels[y:y + h, x:x + w] == i))
    return sorted(out, key=lambda g: g[0])


def _word_bitmap(group) -> np.ndarray:
    x0 = min(g[0] for g in group)
    x1 = max(g[0] + g[2] for g in group)
    y0 = min(g[1] for g in group)
    y1 = max(g[1] + g[3] for g in group)
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for x, y, w, h, m in group:
        canvas[y - y0: y - y0 + h, x - x0: x - x0 + w] |= m.astype(np.uint8)
    return cv2.resize(canvas * 255, (WORD_W, WORD_H), interpolation=cv2.INTER_AREA)


@dataclass
class MatchLabel:
    kind: Optional[str]            # "qualification", "playoff", ... or None
    number: Optional[int]

    def slug(self) -> str:
        kind = self.kind or "match"
        return f"{kind}-{self.number}" if self.number is not None else kind

    def describe(self) -> str:
        if self.number is None:
            return self.kind or "unknown match"
        return f"{(self.kind or 'match').capitalize()} {self.number}"


#: Digits in the title score far below this; letters far above it.
_TITLE_DIGIT_DISTANCE = 0.08


def read_match_label(frame: np.ndarray) -> MatchLabel:
    """Parse the "Qualification 12 of 143" / "Playoff Match 11" caption.

    The match number is the first run of glyphs that are confidently digits, and
    whatever words precede it are matched as one shape against the known match
    types.  Splitting on digit-ness rather than on letter spacing matters: the
    gap inside "12" is nearly as wide as the gap between words, so a spacing
    rule tears the number in half.
    """
    glyphs = _title_glyphs(frame)
    if not glyphs:
        return MatchLabel(None, None)

    digit_bank = bank()
    is_digit: List[bool] = []
    values: List[int] = []
    for _, _, _, _, mask in glyphs:
        normalised = _normalise(mask)
        if normalised is None:
            is_digit.append(False)
            values.append(0)
            continue
        digit, dist = digit_bank.classify(normalised)
        is_digit.append(dist <= _TITLE_DIGIT_DISTANCE)
        values.append(digit)

    start = next((i for i, d in enumerate(is_digit) if d), None)
    if start is None:
        return MatchLabel(word_bank().classify(_word_bitmap(glyphs)), None)

    end = start
    while end < len(is_digit) and is_digit[end]:
        end += 1
    number = int("".join(str(values[i]) for i in range(start, end)))
    kind = word_bank().classify(_word_bitmap(glyphs[:start])) if start else None
    return MatchLabel(kind, number)


def vote_match_label(frames: Iterable[np.ndarray]) -> MatchLabel:
    """The caption never changes during a match, so take the commonest reading."""
    tally: Dict[Tuple[Optional[str], Optional[int]], int] = {}
    for frame in frames:
        label = read_match_label(frame)
        key = (label.kind, label.number)
        tally[key] = tally.get(key, 0) + 1
    if not tally:
        return MatchLabel(None, None)
    kind, number = max(tally, key=tally.get)
    return MatchLabel(kind, number)


# --------------------------------------------------------------------------- #
# Period segmentation
# --------------------------------------------------------------------------- #
@dataclass
class PeriodSpan:
    name: str
    start_frame: int
    end_frame: int          # inclusive

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame


@dataclass
class OverlaySample:
    frame_index: int
    icon: str
    clock: Optional[int]
    counters: Optional[Counters]


#: Nominal length of each period, used to reject spurious runs of the same icon
#: outside the match (pre-match idle, replays, awards footage).
_EXPECTED_SECONDS = {AUTO: 30.0, TRANSITION: 12.0, TELEOP: 120.0}


def segment_periods(samples: Sequence[OverlaySample], fps: float) -> List[PeriodSpan]:
    """Turn per-frame icon readings into contiguous auto / transition / teleop spans.

    The icon is a categorical signal that never blends between states, so a
    short median filter plus run-length encoding is enough.  Broadcasts show the
    scoreboard before and after the match too, so for each period we keep the
    single run whose length best matches how long that period actually lasts --
    a pre-match idle screen showing a gamepad cannot then masquerade as teleop.
    """
    if not samples:
        return []

    labels = [s.icon for s in samples]
    smoothed: List[str] = []
    k = 5
    for i in range(len(labels)):
        window = labels[max(0, i - k // 2): i + k // 2 + 1]
        smoothed.append(max(set(window), key=window.count))

    runs: List[Tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(smoothed) + 1):
        if i == len(smoothed) or smoothed[i] != smoothed[start]:
            runs.append((smoothed[start], samples[start].frame_index,
                         samples[i - 1].frame_index))
            start = i

    min_len = int(3.0 * fps)
    best: Dict[str, Tuple[int, int]] = {}
    for name, a, b in runs:
        if name == OFF or b - a < min_len:
            continue
        target = _EXPECTED_SECONDS.get(name, 0.0) * fps
        if name not in best:
            best[name] = (a, b)
            continue
        pa, pb = best[name]
        if abs((b - a) - target) < abs((pb - pa) - target):
            best[name] = (a, b)

    spans = [PeriodSpan(name, a, b) for name, (a, b) in best.items()]
    spans.sort(key=lambda s: s.start_frame)
    return spans


def period_at(spans: Sequence[PeriodSpan], frame_index: int) -> str:
    for span in spans:
        if span.contains(frame_index):
            return span.name
    return OFF


def enforce_monotonic(values: Sequence[Optional[int]]) -> List[Optional[int]]:
    """Artifact counters only ever go up.  Repair single-frame OCR dropouts."""
    out: List[Optional[int]] = list(values)
    best = None
    for i, v in enumerate(out):
        if v is None:
            continue
        if best is not None and v < best:
            out[i] = None                  # transient misread; drop it
        else:
            best = v
    return out
