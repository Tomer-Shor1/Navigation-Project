# Ex1 — GPS-denied visual navigation: results

**Checking this with nothing installed:** `python summarize.py` re-prints the
table below straight from the committed reports — no data and no dependencies.
**Re-running the pipeline itself:** put a flight video and its matching `.srt`
into `data/raw/` and run `./reproduce.sh`. The extracted frames (~170 MB) are
not committed, so the first run re-extracts them from the video with ffmpeg;
after that they are cached and every later run is fast.

> **Regenerated on OpenCV 4.13 / NumPy 2.5 (2026-08-24).** Every number below was
> re-measured on this stack after `np.cross`'s removed 2-D form was fixed in
> `homography_is_plausible` (NumPy 2.0 dropped it, so the pipeline could not run
> at all). The fix is arithmetically identical to what it replaced; the numbers
> nonetheless moved a little against the previous submission, because ORB and
> RANSAC do not return bit-identical results across OpenCV versions. Every
> conclusion below survived the re-run except one, which is called out where it
> appears (raster map on the return leg).

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

Two numbers matter and they say different things. **Visual fixes** is the error
of the positions the system actually trusted. **All reported** also folds in the
frames where it found nothing trustworthy and coasted on the motion model — what
a real navigator would have output.

| flight | test frames | visual fixes | visual-fix median / p90 | all-reported median / p90 |
|---|---|---|---|---|
| `flight` (50 m AGL) | 23 | 23 | **6.5 m** / 18.3 | **6.5 m** / 18.3 |
| `flight_0024` (31 m AGL, incl. take-off + landing) | 27 | 16 | **6.8 m** / 16.6 | 12.4 m / 163.9 |

`flight_0024` shows the gap plainly: where it commits, it is accurate to ~7 m,
but a third of that flight is take-off and landing below 20 m altitude, where
the camera sees too little ground to match anything, and coasting through those
gaps drifts badly. `results/<flight>/trajectory_comparison.png` overlays the
estimates on the true track; `error_report.txt` has the per-frame detail. To step
through the run that produced them frame by frame — what the matcher saw, which
map view it chose and why — run `./run.sh`, pick a flight and press Start (see
README, "Watching it run").

## What changed, and why it mattered

Four defects were found. The first two are ablatable — the old behaviour is one
flag away. The last two were plain bugs and are simply fixed.

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
| interleaved hold-out | self-calibrated *(default)* | 23/23 | **6.5 m** | 18.3 | 33.8 |
| interleaved hold-out | modelled | 23/23 | 10.9 m | 19.6 | 22.5 |
| tail (return leg) | self-calibrated | 8/24 | 47.2 m | 63.4 | 66.8 |
| tail (return leg) | modelled *(original submission)* | 10/24 | 21.3 m | 67.2 | 198.8 |

Note the honest negative in row 3: on the tail split, calibration makes things
*worse*. The matches there are wrong to begin with, so a larger — and correct —
scale simply amplifies a wrong pixel offset. Calibration helps only once the
matches are real.

**3. RANSAC homographies were never validated.** `cv2.findHomography` returns a
transform with a high inlier count from points that are clustered or nearly
collinear — a fit that folds or collapses the frame and throws the mapped centre
hundreds of metres away, while reporting 40–60 inliers. Nothing downstream could
distinguish that from a real match. It was found on the raster map, where one
patch became a magnet that won almost every frame at 200–270 m error, but it was
degrading the flight-frame map too: adding two standard checks (the frame's
projected outline must stay convex, and its area must stay within 4x) cut
`flight_0024`'s visual-fix p90 from 58 m to 17 m. See `homography_is_plausible()`.

**4. Contrast normalisation was applied to one side only.** The map got CLAHE,
the drone frame did not, so ORB was comparing differently-normalised intensities
and weak false matches could outrank true ones. Now both go through
`normalize_for_matching()`.

## Final project, part 1: a georeferenced raster as the map

Ex1's map is the flight's own earlier frames. The final project replaces it with
a **georeferenced raster** — Google Earth or satellite imagery — so the drone can
be localized over ground it has never itself flown. That is implemented here:

* `src/gis_reference.py` — Web-Mercator geometry, the basemap loader, and the
  patch indexer that turns a raster into `ReferenceEntry` objects. Nothing
  downstream changes, because the map still arrives through the same
  `ReferenceSource` interface Ex1 used.
* `tools/fetch_basemap.py` — downloads XYZ satellite tiles for a flight's
  bounding box and writes the raster plus a four-number bounds sidecar. A Google
  Earth Pro export works through the same loader.
* `src/orthomosaic.py` — builds a north-up georeferenced raster **out of the
  flight frames themselves**, in the identical `Basemap` format. This is what
  makes the whole GIS path testable and measurable without a network, and it is
  a real method in its own right.

A raster map is structurally better than a stack of oblique frames: both error
sources that had to be *measured* out of the flight-frame map disappear. Scale is
a closed form (`156543.03392 * cos(lat) / 2^zoom` m/px) and heading is exactly 0,
because the raster is north-up. No FOV guess, no altitude datum, no gimbal angle.

### Measured: raster map vs. flight-frame map

`--map-source ortho` against `--map-source flight`, everything else identical.
All-reported error, since the two differ in how often they commit.

| experiment | map source | visual fixes | median | p90 | max |
|---|---|---|---|---|---|
| `flight`, interleaved | flight frames | 23/23 | 6.5 m | 18.3 | 33.8 |
| `flight`, interleaved | **north-up raster** | 21/23 | 7.2 m | **15.2** | 51.2 |
| `flight`, **return leg** | flight frames | 8/24 | **40.4 m** | 88.3 | 112.8 |
| `flight`, **return leg** | **north-up raster** | 5/24 | 62.9 m | **73.3** | **81.8** |
| `flight_0024`, interleaved | flight frames | 16/27 | 12.4 m | **163.9** | 248.2 |
| `flight_0024`, interleaved | **north-up raster** | 10/27 | 11.4 m | 253.4 | 294.6 |

Read this as three findings, not one:

1. **On the easy case it is a wash** — 6.5 vs 7.2 m. The raster does not buy
   accuracy where the flight-frame map already works, but it does not cost any
   either, while needing no calibration.
2. **On the return leg it does *not* help, and this reverses an earlier claim.**
   The previous submission measured the raster improving the return-leg median
   from 61.5 m to 46.0 m and concluded that making the map north-up removes its
   dependence on the heading it was flown at. On the re-run that reverses: median
   40.4 → 62.9 m. What survives is narrower and worth keeping — the raster still
   has the *tighter tail* (p90 88.3 → 73.3, max 112.8 → 81.8) and it commits more
   often (5/24 vs 3/24 before). So north-up orientation bounds how wrong a
   return-leg fix can be, but on this evidence it does not make the typical one
   better. With 5 and 8 accepted fixes respectively, neither median is worth much;
   the honest reading is that this comparison is under-powered, not that the
   raster lost.
3. **It does not solve the return leg.** Only 5 of 24 frames get a trusted fix.
   Rotation was never the whole problem: the raster is still built from *outbound*
   imagery, so a building's sunlit face and the side of every tree still look
   different coming back the other way. That is a property of the imagery, not of
   the map's orientation — which is exactly the argument for a genuinely nadir
   satellite basemap, and for a learned matcher.

`results/<flight>_ortho/orthomosaic.jpg` is the raster the drone navigated
against (downscaled for the repository).

> **Since measured: real satellite imagery.** The tile downloader could not be run
> when the paragraph above was written. It runs now — two defects in this
> repository were what stopped it, not the network — and the flight has been
> navigated against real Google satellite imagery at 0.126 m/px. The prediction
> here was half right and half wrong: cross-sensor appearance change is indeed the
> problem, but the nadir viewpoint **hurts** rather than helps, because the drone
> frames are oblique. Part 2 below has the measurement and the evidence.

## Final project, part 2: real-time navigation on previous video **and** GIS

> *"Consider Ex1, design a real-time visual navigation algorithm based on
> predefined (annotated) previous videos, and GIS datasets (such as Google
> Earth)."*

Three claims to make good on: **real-time**, **previous videos**, and **GIS**.
Ex1 delivered the second. This section is the other two, and one of them is a
negative result that is worth more than a positive one would have been.

### The GIS map is real now, and that took two bug fixes

The previous submission could not test against satellite imagery at all. Both
reasons turned out to be defects in this repository, not in the network:

1. **TLS.** `tools/fetch_basemap.py` failed every request with
   `CERTIFICATE_VERIFY_FAILED`. A python.org build on macOS ships without a CA
   bundle wired up, so Python could not verify a chain that `curl` on the same
   machine verified fine. Fixed by verifying against `certifi`'s root store.
2. **Silent blank basemaps.** With TLS fixed, Esri returned HTTP 200 for all 357
   tiles and the tool reported *"357/357 tiles ok"* — for a basemap containing no
   imagery whatsoever. They were **"Map data not yet available" placeholders**:
   Esri World Imagery has no coverage below zoom 18 here. The fetcher now
   measures each tile's contrast, counts placeholders separately, and refuses to
   write a basemap that is mostly blank.

With both fixed, `--source google` gives real imagery at zoom 20:
**0.1265 m/px, 4094 × 4970 px, 354 of 357 tiles with imagery**, indexed as 252
overlapping patches.

### Result: ORB cannot match a drone frame to satellite imagery

| flight | map | visual fixes | median | p90 | max |
|---|---|---|---|---|---|
| `flight` | previous flight frames | 23/23 | **6.5 m** | 18.3 | 33.8 |
| `flight` | orthomosaic built from those frames | 21/23 | 6.1 m | 12.7 | 51.2 |
| `flight` | **Google satellite, zoom 20** | **1/23** | **218.1 m** | — | — |
| `flight_0024` | previous flight frames | 16/27 | 6.8 m | 16.6 | 18.4 |
| `flight_0024` | **Google satellite, zoom 20** | **3/27** | **313.3 m** | 337.9 | 344.0 |

One accepted fix in twenty-three on the first flight, three in twenty-seven on
the second, and every one of them hundreds of metres wrong. The failure
replicates.

### Why — measured with a GPS oracle, not guessed

The same diagnostic this project used for the return-leg question: hand each
test frame **the basemap patch at its true GPS position** and see what the
matcher can do in the best case there is. Three detectors, each with and without
the frame de-rotated to north-up using its measured heading (10 frames):

| detector | frame rotated north-up | avg good matches | avg inliers | plausible homographies |
|---|---|---|---|---|
| ORB | no | 14.8 | 4.7 | 1/10 |
| ORB | yes | 18.5 | 4.8 | 0/10 |
| SIFT | no | 18.1 | 4.5 | 0/10 |
| SIFT | yes | 22.9 | 5.4 | 0/10 |
| AKAZE | no | 28.1 | 5.3 | 0/10 |
| AKAZE | yes | **38.0** | 5.4 | 0/10 |

For scale, a same-heading flight-frame overlap yields **500–1500** good matches,
and an orthomosaic patch yields 78–131. Here the best classical detector
available manages 38 good matches and *not one usable homography*.

So this is not a threshold that needs loosening, not the motion gate, and not the
scale conversion. Removing the rotation helps (AKAZE 28 → 38) and removing the
detector's weaknesses helps (ORB 14.8 → AKAZE 38.0), and neither comes close.

Putting a drone frame beside the satellite patch at its own GPS shows why in one
look: **the drone is oblique and the satellite is nadir**, so the drone sees
building *facades* where the basemap has *roofs*, and every tree and parked car
casts its structure in a different direction. That is not an appearance nuisance
a hand-crafted descriptor can normalise away; it is a different view of a
three-dimensional scene.

### The hybrid map: both sources, competing per frame

`--map-source hybrid` puts the previous flight's frames and the satellite raster
behind one `CompositeReferenceSource`, so a single radius query returns both. By
default the two are **tiers**, not competitors: the previous flight's video is
tried first and the GIS raster is consulted only for frames the video could not
explain (`--hybrid-policy compete` restores per-frame competition). On
`flight_0024` that fallback fires on **2 of 18 fixes** — one of them 7.3 m from
truth on a frame the flight map had failed outright, the other 344 m wrong.
Nothing in the localizer needed changing to arbitrate between them — it already
verifies every candidate by RANSAC inliers and keeps the best, so the two maps
compete **per frame** rather than being chosen between per flight. The one real
change was that a hybrid map needs the drone frame prepared *twice*: at native
resolution for oblique flight frames, and resampled plus contrast-equalised for
north-up raster tiles (`FrameView` in `src/localize.py`).

    flight          Which map produced each fix:
                      previous flight video     23 fixes (100%)  median   6.53  max  33.76
                      GIS / satellite raster     0 fixes (  0%)  offered, never won

    flight_0024     Which map produced each fix:
                      previous flight video     16 fixes ( 89%)  median   6.83  max  18.44
                      GIS / satellite raster     2 fixes ( 11%)  median 175.65  max 344.00

The player makes this legible: open a frame on a hybrid map and the candidate
table lists flight frames and satellite tiles side by side, with the tiles
scoring 12-14 good matches and being thrown out as degenerate homographies while
a flight frame wins with 71 good matches and 34 inliers.

**And the second flight shows a hybrid can make things *worse*, which is the more
useful result.** On `flight_0024` the satellite map won two frames outright, and
both were catastrophic: 176 m and 344 m. Map-fix median went from 6.83 m to
7.30 m and the worst fix from 18 m to 344 m. A map that contributed nothing
useful still contributed two confident lies.

The mechanism is a design lesson rather than a bug. Both maps are judged against
**one shared acceptance bar** (6 inliers, 12% ratio), and the innovation gate that
would normally veto a 344 m jump had been widened by a run of dead-reckoned
frames -- after a 30 s coast the reachable radius is 22 x 30 + 30 = 690 m, so
344 m is "physically plausible" by then. A hybrid map therefore needs
**per-source confidence**, not a shared threshold: a map that has never produced a
good fix should have to clear a far higher bar than one that produces them
constantly. That is the concrete next change, and `--max-evidence-scale`
(previously measured as a wash on single-source maps) is the hook it plugs
into.

**What the hybrid is still worth.** Architecturally it is the right shape and it
costs nothing to keep: the satellite raster is the map that exists everywhere,
including ground this drone has never flown, and the moment the matcher can
bridge the viewpoint gap it starts contributing without another line of code.
What it needs is a **learned matcher** — SuperPoint + LightGlue or LoFTR, trained
precisely to survive this kind of appearance change — or drone frames that are
orthorectified to nadir before matching, which needs a gimbal pitch these SRT
files do not contain.

### Real-time: measured

Real-time here means one thing: a frame must be localized in less than the
interval between frames — 1000 ms at 1 Hz. `tools/benchmark.py` reports it.

| map | views | searched (median) | frame cost (median / p90) | vs 1 Hz budget |
|---|---|---|---|---|
| previous flight frames | 94 | 52 | **272 / 329 ms** | **3.7x faster than real time** |
| hybrid (video + satellite) | 346 | 110 | **558 / 621 ms** | **1.8x faster than real time** |

Worst single frame: 336 ms flight-only, 643 ms hybrid — both inside the budget.

Where the time goes, and why the gate is the whole design:

| phase | flight map | hybrid |
|---|---|---|
| decode JPEG | 9 ms (3%) | 8 ms (1%) |
| ORB on the frame | 42 ms (15%) | 62 ms (11%) |
| motion gate | 0.1 ms (0%) | 0.2 ms (0%) |
| **descriptor matching** | **209 ms (77%)** | **450 ms (81%)** |
| RANSAC verify | 11 ms (4%) | 38 ms (7%) |

Matching dominates, and it is linear in the number of map views searched —
measured at **5.0–7.2 ms per view**, flat across gate sizes. That is the argument
for motion gating stated as a number: the per-frame bill is set by how many views
the gate admits (52 of 94, 110 of 346), *not* by how large the map is. A
city-scale basemap costs the same per frame as this one, as long as the drone
still only searches its own neighbourhood.

It is also where the headroom goes. The hybrid more than doubles the cost for
zero fixes, and brute-force Hamming over every gated view is the reason. A
descriptor index (vocabulary tree, FAISS) would break the linearity, and that is
the first thing to do before this runs on a real map at a real frame rate.

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
* The system is now well calibrated in the "when it commits, trust it" sense and
  under-confident in the other: `homography_is_plausible` correctly refuses bad
  fits, but every refusal becomes a dead-reckoned frame, and dead reckoning
  drifts by ~100 m over a long gap. Better coasting (an IMU, or frame-to-frame
  visual odometry between map fixes) would convert that precision into a better
  aggregate.
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
