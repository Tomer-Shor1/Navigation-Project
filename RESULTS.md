# Ex1 — GPS-denied visual navigation: results

**Checking this with nothing installed:** `python summarize.py` re-prints the
table below straight from the committed reports — no data and no dependencies.
**Re-running the pipeline itself:** put a flight video and its matching `.srt`
into `data/raw/` and run `./reproduce.sh`. The extracted frames (~170 MB) are
not committed, so the first run re-extracts them from the video with ffmpeg;
after that they are cached and every later run is fast.

---

## The problem, as implemented

Preprocessing turns a georeferenced flight (video + SRT telemetry) into a
**map**: 1 Hz frames, each with ORB features and its GNSS position, altitude and
a measured camera heading. Navigation then takes a frame **with no GNSS**,
predicts where the drone is from a constant-velocity motion model, searches only
the map within physical reach of that prediction, verifies each candidate with a
RANSAC homography, and converts the resulting pixel offset into a lat/lon. When
no match is trustworthy it coasts on the motion model instead of failing.

## Main result

Map = the flight's own frames minus a held-out every-5th frame. Test = the
held-out frames, localized without their GNSS. A test frame may not be matched
against any map frame within **±2 s** of itself, so it cannot trivially match its
own neighbours. Error is the great-circle distance to the SRT ground truth.

| flight | test frames | visual fixes | coasted | median | mean | RMSE | p90 | max |
|---|---|---|---|---|---|---|---|---|
| `flight` (50 m AGL) | 23 | 23 | 0 | **6.2 m** | 10.1 | 14.4 | 19.2 | 41.9 |
| `flight_0024` (31 m AGL) | 27 | 26 | 1 | **10.5 m** | 20.0 | 30.7 | 58.0 | 80.3 |

At 1 Hz with a ~50 m flight altitude, a 6–10 m median is roughly one tenth of the
frame footprint. `results/<flight>/trajectory_comparison.png` overlays the
estimated positions on the true track; `error_report.txt` gives the per-frame
match counts, inlier counts and errors.

## What changed, and why it mattered

Two defects were found in the original submission. Both are ablatable — the old
behaviour is still one flag away.

**1. The camera geometry was wrong by ~36%.** Metres per pixel was modelled from
FOV, the SRT's `rel_alt` and an assumed −60° gimbal pitch. But `rel_alt` is
height above the **take-off point**, not above the ground being filmed, and these
SRT files carry **no gimbal angle at all**. On `flight` the model gives 0.0455
m/px where the truth is ~0.071 m/px. Verified two independent ways: from the
reference track's own GPS, and from the imagery itself — the parking-stall pitch
in `frame_00050.jpg` autocorrelates at 34 px, which at a 2.4 m stall is 0.0706
m/px. `src/calibrate.py` now measures the scale, and each map frame's camera
heading, from the reference track during preprocessing (no test-frame GNSS
involved). Disable with `--no-calibrate`.

**2. The headline number measured the wrong thing.** The original split used the
first 80% of the flight as the map and the last 20% as the test set. On these
flights that last 20% is the **return leg, flown back over the mapped ground on
the opposite heading**, and ORB cannot match a place seen from a reversed
viewpoint. This is not a tuning problem: matching each test frame against the map
frame nearest its *true* GPS position — a GPS oracle, the best any retrieval
could do — still yields only **5–18 good matches and 0–6 RANSAC inliers**, versus
500+ on a same-heading overlap. Rotating the test frame to align headings does
not help either (measured; the deltas are 150–170°). So the old 40 m median
reported the failure of appearance matching across a 180° revisit, not the
accuracy of the navigator. `--split tail` reproduces it.

### Ablation on `flight`

| split | camera geometry | visual fixes | median | p90 | max |
|---|---|---|---|---|---|
| interleaved hold-out | self-calibrated *(default)* | 23/23 | **6.2 m** | 19.2 | 41.9 |
| interleaved hold-out | modelled | 23/23 | 10.1 m | 19.5 | 38.4 |
| tail (return leg) | self-calibrated | 12/24 | 61.4 m | 94.2 | 105.3 |
| tail (return leg) | modelled *(original submission)* | 15/24 | 40.1 m | 95.8 | 99.2 |

Note the honest negative in row 3: on the tail split, calibration makes things
*worse*. The matches there are wrong to begin with, so a larger — and correct —
scale simply amplifies a wrong pixel offset. Calibration helps only once the
matches are real.

## Known limitations

* **Opposite-heading revisits fail.** The clear next step, and the one the
  assignment's "papers with code" direction points at: replace the ORB
  brute-force matcher with a learned matcher (SuperPoint + LightGlue, or LoFTR),
  or orthorectify the map so appearance stops depending on viewing direction.
  `match_candidates()` in `src/localize.py` is the single function to swap.
* The estimate is the ground point under the video centre, scored against the
  drone's own GNSS. The two coincide only while the map frame and the test frame
  share a heading and altitude; the motion gate keeps them close, but this is an
  approximation, and it is part of the residual error during turns.
* Scale is one number per flight (times altitude). It does not model the
  trapezoidal footprint of an oblique shot, so it is least accurate far from the
  frame centre.
* Only `flight` and `flight_0024` are published here. `flight_0017` and
  `flight_0023` were run under the old protocol and have not yet been
  regenerated under the corrected one, so their stale reports were left out
  rather than shipped as if they were current. `./reproduce.sh` picks up any
  flight whose video and `.srt` are present in `data/raw/`.

## Reproducing

No data needed — just re-print the committed results:

```bash
python summarize.py
```

Full re-run. Requires `ffmpeg` on the PATH, and the flight videos + `.srt` files
in `data/raw/` (they are too large to commit; the course provides them):

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./reproduce.sh            # every flight found in data/raw/, plus the ablation
```

Single flight, explicit:

```bash
python run_pipeline.py --video data/raw/flight.mp4 --srt data/raw/flight.srt
```

Add `--use-cached-frames` to skip frame extraction once `data/frames/` has been
populated by an earlier run.
