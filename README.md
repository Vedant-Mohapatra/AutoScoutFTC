# AutoScoutFTC

Automatic scouting of artifact-shooting robots from an FTC **DECODE** match
video. Point it at a broadcast recording of a match and it will tell you, for
each of the four robots on the field, how many artifacts it launched and how
many of those went into a goal — split into the autonomous and driver-controlled
periods using the on-screen match timer.

```
python scout.py analyse Videos/my_match.mp4 --out out/
```

## What it produces

Running it on one of the bundled matches (FIRST Championship 2026, Lovelace
Division, Qualification 12):

```
Periods (from the on-screen timer)
  auto        video    7.60s ->   36.53s   ( 28.9s)
  transition  video   36.67s ->   51.60s   ( 14.9s)
  teleop      video   51.73s ->  172.40s   (120.7s)

Teams      : red 18264, 19639   |   blue 15083, 25963
Final score: red 208   |   blue 343

Per-robot shooting (vision)
  robot  alliance  team    auto a/m   teleop a/m   total a/m   acc   tracked
  ---------------------------------------------------------------------------
  0      blue      -       17/14      46/41        63/55       87%   100%
  2      blue      -       11/6       52/33        63/39       62%   100%
  1      red       -       13/11      42/35        55/46       84%   99%
  3      red       -       3/2        0/0          3/2         67%   99%

Alliance artifacts scored, read from the scoreboard (exact)
  red   auto   18   teleop   46   total   64
  blue  auto   22   teleop   75   total   97

Vision vs scoreboard, made shots (how much to trust the table above)
  red   vision saw  48 land in the red goal, scoreboard counted  64   ( 75% found)
  blue  vision saw  98 land in the blue goal, scoreboard counted  97   (101% found)
```

The reading: blue did most of the scoring, one of its robots at 87% accuracy; on
red, one robot took essentially every shot at 84% while its partner took three
all match and then stopped — a feeder or defensive role, and exactly the sort of
thing scouting is for.

The playoff match in `Videos/` is a different field with a different scoreboard
(three teams per alliance, "Alliance 1/2" instead of "Red"/"Blue") and needs no
extra configuration:

```
python scout.py analyse Videos/...Playoff-Match-11....mp4 --out out/
# -> out/playoff-11/
Match      : Playoff 11
Field      : floor x623-1256 y129-810 (633x681 px), robot 91 px, scale 0.99
Teams      : red 30030, 21087, 11228   |   blue 15083, 24006, 19639
  1      blue      -       8/7        40/37        48/44       92%   100%
  2      blue      -       13/9       30/24        43/33       77%   100%
  0      red       -       8/7        35/31        43/38       88%   100%
  3      red       -       20/13      67/41        87/54       62%   100%
```

Each match gets its own directory, named from the caption on the scoreboard, so
a season's worth of video lands in a browsable tree rather than one heap:

```
out/
  qualification-12/
    robots.csv       one row per robot: attempts and makes, per period
    shots.csv        one row per detected shot: match clock, robot, goal, made/missed, coordinates
    match.json       everything above plus periods, teams, field measurements, warnings
    summary.txt      the text report shown above
    debug.mp4        (with --debug-video)
  playoff-11/
    ...
```

The caption is read the same way as the rest of the overlay. The match number is
the first run of glyphs that are confidently digits, and the words in front of
it are matched as a single shape against known match types, so `Qualification
12 of 143` becomes `qualification-12` and `Playoff Match 11` becomes
`playoff-11`. An unrecognised wording still yields a usable directory
(`match-7`); teach it a new one with:

```
python tools/build_match_types.py --add practice=Videos/some_practice_match.mp4
```

Add `--debug-video` to also render an annotated copy of the match: robot boxes
coloured by alliance and labelled with their track id, the measured field
outline, goal mouths, and every detected shot drawn as an arrow from launch to
landing with a green or red ring for made or missed — plus the period, the match
clock and a running per-robot tally in the corner. It is the fastest way to
sanity-check a result, and the quickest way to see which track id is which
robot before using `--robot-team`.

OpenCV can only write MPEG-4 Part 2, which most browsers refuse to play and
which runs several times the size for the same picture, so the render is
re-encoded to H.264 afterwards if `ffmpeg` is on PATH. Without ffmpeg you still
get a valid file, just a bigger and fussier one.

## How it works

The pipeline makes two passes over the video.

**Pass 1 — the scoreboard.** The FTC broadcast overlay is a vector graphic
re-rendered every frame, so its numbers can be read exactly rather than
guessed at. `autoscout/overlay.py` segments each numeric field into glyphs,
size-normalises them and matches them against the digit stencils in
`autoscout/assets/digits.npz`. From this pass we get:

* the **match clock** (`2:30` → `217` → 137 seconds),
* the **period**, from the pictogram above the clock — a robot head for
  autonomous, an open palm for the hands-off transition, a gamepad for driver
  control, and nothing once the match is over. The three shapes differ enough in
  aspect ratio that no template is needed,
* each alliance's **artifact counter**, which is the ground truth for how many
  artifacts actually went in,
* the **team numbers** on the alliance plates, decided by majority vote across
  the match.

None of those are read from fixed coordinates. The overlay grows an extra panel
once autonomous has been scored, which shifts the counters sideways, and a
playoff scoreboard is laid out differently from a qualification one. So the
value plates are *located*: threshold the bar for white, and the artifact
counters are the two outermost plates with the distinctive tall-and-narrow shape
(they are a stacked pair of boxes that reads as one), while a team roster is a
wide plate holding at least two numbers — which is what separates it from the
single-value autonomous panel next to it.

**Pass 2 — the field.** A per-pixel median across the match gives a background
plate of the bare floor. Subtracting it and sliding a robot-sized box over the
result locates the four robots; because a legal FTC robot is a known fraction of
the field, this keeps working when two robots collide and would otherwise fuse
into one connected component. Each accepted peak *consumes* the foreground
underneath it before the next is sought — suppressing a fixed radius instead
forces a choice between two peaks landing on one chassis and two touching robots
collapsing into one, and taking the pixels away makes that choice unnecessary.

Identities are seeded once the field holds exactly four solid, well-separated
masses — a broadcast wipe at the start of the playoff match puts six on screen,
and seeding then pins an identity to whatever is covering the field. They are
then maintained by prediction plus nearest-detection association, with a colour
histogram of each robot's own pixels breaking ties (robots spin constantly, so
the profile has to be rotation-invariant). Robots are never created or destroyed
mid-match. A track that does lose its robot — during a collision, or against the
wall — can jump back to an unclaimed detection whose colours match; without
that, a tight association gate means a track separated once simply sits where it
died for the rest of the match. Between them, those two fixes took the worst
identity across the two sample matches from 16% of frames locked on to 99-100%.

Artifacts are found by hue. Purple is unambiguous; green is not — several robots
carry green LEDs in exactly the artifact's hue — so **every** colour blob is
tracked and the filtering happens afterwards, on whole trajectories. Each track
is then cut into the stretches where it was moving fast enough to be airborne. A
flight that travels far enough, and travels *up-screen* towards the goals, is a
shot. It counts as **made** if its tail lands inside a goal region.

Each shot is traced back to a shooter by walking the trajectory backwards along
its launch velocity — the artifact has usually already cleared the robot by the
time it is first visible — and taking the closest approach to a robot centre.

Finally, which alliance owns which goal is *measured* rather than assumed, by
comparing totals: the goal that received more artifacts must belong to the
alliance the scoreboard credits with more. (Matching individual makes to
individual counter ticks does not work — both alliances score more or less
continuously, so within any plausible scoring-latency window there is nearly
always a tick from *either* alliance and the nearest one is a coin flip. On the
qualification match that method votes 38-34, i.e. noise.) When the two alliances finish
too close together to call, the tool says so and falls back to the
colour-matched default instead of guessing. A robot is then assigned to the
alliance whose goal it shoots at, and one that never shoots falls back to the
side of the field it started on.

## Accuracy, honestly

Three different things come out of this tool and they do not deserve equal
trust.

**Exact.** The period boundaries, the match clock, the team numbers, the final
score and the per-alliance artifact totals are read off the broadcast overlay.
They are not estimates — they are the scoring system's own numbers.

**Close.** How many artifacts landed in each goal. On the qualification match
vision found 98 of blue's 97 and 48 of red's 64; on the playoff, 98 of red's 138
and 78 of blue's 125. Two things make a goal harder: robots that shoot from
close range (short trajectories, easily missed) and a high scoring rate, which
puts several artifacts in the air at once for the tracker to confuse.

**Indicative.** The per-robot split. Attribution is only as good as the make
detection feeding it, so read that table as *which robot is doing the shooting
and how their accuracies compare*, not as an exact count. The `tracked` column
says how much of the match each identity stayed locked onto a real robot; if it
drops well below 100%, discount that row. `scout.py` also warns whenever the
vision and scoreboard totals disagree by more than 20%.

Recall is not uniform across matches -- roughly 60-100% of makes depending on
how fast the alliance is scoring -- so compare robots within a match rather than
across matches.

Nothing here uses a trained model — it is classical CV on hue, motion and a
known robot footprint. If you need exact per-robot counts, the upgrade path is a
small detector (e.g. YOLO on a few hundred labelled frames) dropped into
`autoscout/detect.py`; the tracking, shot and reporting layers above it would
not have to change.

## Adapting to a different video

The field is **measured, not configured**. Nothing in `config.py` names a pixel
coordinate on the playing surface: the floor is found in the background plate (a
big slab of unsaturated grey, unlike the saturated goals or the dark carpet
around it), and every other region — goal mouths, artifact chutes, robot
footprint, even the shot distance and speed thresholds — is a fixed proportion of
it. Point the tool at a different field, or at a broadcast that frames it
differently, and it adapts.

The scoreboard works the same way: counter plates and team rosters are located by
shape within the bar rather than pinned to coordinates, so the results panel that
appears mid-match and shoves everything sideways, and the playoff layout carrying
three teams per alliance instead of two, both read correctly with no second
config.

To see what it measured:

```
python scout.py calibrate Videos/my_match.mp4 --out out/
```

That prints the measured field and writes a frame with every region drawn on it.
If something looks off, the proportions live in `FieldProportions` in
`autoscout/config.py`, and a geometry can be pinned outright via
`Config.geometry`.

Team numbers can always be supplied by hand:

```
python scout.py analyse video.mp4 --team red=18264,19639 --team blue=15083,25963
```

Nothing on screen links a chassis to a plate on the scoreboard, so the `team`
column stays blank unless an alliance has only one tracked robot. Watch the
debug video to see which id is which robot, then bind them:

```
python scout.py analyse video.mp4 --robot-team 0=15083 --robot-team 3=18264
```

## Tests

```
python tests/test_autoscout.py
```

Covers the parts that do not need a video: digit templates, period
segmentation, counter repair, flight extraction, made/missed classification,
duplicate collapsing, goal-ownership inference and track recovery.

## Caching

Both passes are cached under `<out>/.cache`, keyed on the video and on the
settings that affect each pass. Re-running after changing a tracking or shot
threshold reuses the decode and finishes in ~25 s instead of ~5 min; changing a
detection threshold correctly invalidates it. Entries are per-video and
superseded ones are pruned, so the directory settles at roughly 18 MB per match.
`--no-cache` disables it.

## Install

```
pip install -r requirements.txt
```

Python 3.9+, NumPy and OpenCV. Nothing else.

## Layout

```
autoscout/
  config.py     geometry, colours and thresholds -- the only file you should need to edit
  overlay.py    scoreboard OCR, period segmentation
  detect.py     background model, robot and artifact detection
  tracking.py   robot identities and artifact trajectories
  shots.py      flights, made/missed, shooter attribution, alliance inference
  pipeline.py   orchestration
  report.py     CSV / JSON / text output
  viz.py        debug video and calibration sheet
  cache.py      on-disk cache for the decode passes
scout.py        command line entry point
```
