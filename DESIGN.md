# Design notes — GPS-denied visual navigation

The long-form write-up: what the problem is, what the algorithm does, what it
assumes, and where it breaks. `README.md` is the short version and how to run it;
`RESULTS.md` has the measurements.

---

## Problem statement

GNSS (GPS) signals can be jammed, spoofed, or simply unavailable, yet a drone
still needs to know roughly where it is. This MVP demonstrates the core idea
behind GPS-denied visual navigation: if you have a *georeferenced reference*
of a place (in this case, an earlier segment of the same flight, with known
GPS per frame), you can localize a *new, GPS-less* view of that place by
matching it visually against the reference and projecting the match back into
real-world coordinates. Concretely, this project takes one DJI drone flight
video with known telemetry, treats the first 80% of it as a "known" reference
map and the last 20% as if its GNSS were unavailable, and estimates the ground
coordinate under each last-20% frame's center using classical image feature
matching **fused with the non-GNSS telemetry** (barometric height, camera
angle, and a motion model seeded from the reference track) — then compares the
estimate against the real GPS from the flight log.

This directly targets the assignment's core problem: *"given new real-time
flight data (video + telemetry, without GNSS), compute in real-time the
coordinate of the center point of the video, assuming the preprocessing data
contains the current position."* The reference set is the preprocessing; the
motion-gated, inlier-verified matcher is the real-time navigation loop; and the
output is the video-center ground coordinate.

### What sensor data is used (and why it's fair under GNSS denial)

The DJI SRT gives per-frame `latitude/longitude`, `rel_alt`/`abs_alt`
(barometric), `focal_len`, and exposure — at 30 fps. Under GNSS denial the
*test* frames' lat/lon are withheld (used only for scoring), but the rest is
legitimately available in a real GPS-denied flight and is now actually used:

- **Barometric height** (`rel_alt`) → ground-sample-distance (meters-per-pixel).
  A barometer keeps working when GNSS is jammed, so this is fair to rely on.
- **Camera angle** → slant-range correction of the GSD. This flight's SRT has no
  gimbal field, so the assignment's known 60° angle is used as the fallback.
- **Reference-track GPS** (preprocessing only) → seeds a constant-velocity
  **motion model** and resolves each reference frame's **heading**. The test
  frames then dead-reckon + visually correct with *no* test-frame GNSS.

### Does it need prior knowledge of the course? (and what if it doesn't have it)

Yes — and this is fundamental, not a shortcut. Any method that recovers an
*absolute* position from images needs *some* georeferenced reference of the area
being overflown. Vision without a map can only measure *relative* motion (visual
odometry), which drifts without bound. So there are three regimes, and this repo
is built to span all three:

1. **Known course (implemented, measured here).** The reference is an earlier
   georeferenced segment of the same flight. This is the MVP.
2. **No prior video of the course → use a GIS map (Google Earth).** The honest
   answer to "what if it doesn't know the course" is: swap the reference source
   from prior-flight frames to a georeferenced map that already covers
   everywhere. The navigator is given a **starting position**, propagates it with
   the motion model, and **corrects against map tiles** pulled from a *radius of
   interest* around the current estimate — exactly the loop already implemented,
   with the reference source changed. See "Extending to GIS / Google Earth". (The
   hard part there is matching an oblique drone frame to a nadir map tile, which
   needs a learned matcher — noted as future work.)
3. **No map at all, momentarily → coast on motion.** When *no* trustworthy match
   is available for a frame, the navigator does **not** reject it: it reports a
   **dead-reckoned** position from the motion model (flagged, lower-confidence)
   and keeps going. This is graceful degradation, like an INS bridging a GPS
   dropout.

**A measured caveat on regime 3 (important):** relying on motion *alone* only
works for short gaps here, because this dataset has **no IMU** and the SRT gives
no inertial data. Two things were tested:

- **Visual odometry** (estimate motion from consecutive frames): on this 1 Hz,
  oblique-60° video it did *not* track the drone — magnitude correlation with
  true GPS displacement was ≈ 0 (it produced 55 m single-step jumps vs a true
  12 m max). Too noisy to integrate, so it is *not* used.
- **Constant-velocity coasting**: uncapped, it diverged to ~500 m over a long
  gap (the drone hovers and turns, so last-known velocity stops being valid).
  It is therefore **capped** (`dead_reckon_report` in `src/motion.py`) to a short
  extrapolation then hold-near-last-fix, which keeps dead-reckon error bounded
  (~100 m here) instead of runaway. The takeaway that answers the question
  directly: *without a motion sensor you cannot navigate on movement alone — the
  map (prior video or Google Earth) is what makes it work; motion only bridges
  the gaps.*

## Related work

- **Cross-view drone geo-localization benchmark.** Zheng, Wei & Yang, *University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization*, ACM Multimedia 2020 ([paper](https://arxiv.org/abs/2002.12186), [code](https://github.com/layumi/University1652-Baseline)). Establishes the standard drone/satellite/ground multi-view retrieval benchmark that most learned cross-view geo-localization work (including the University-1652-style approaches referenced below) is trained and evaluated against. This project doesn't use a learned retrieval model, but the benchmark defines the harder version of the problem this MVP simplifies away by keeping reference and test images same-flight, same-viewpoint-family.
- **Classical (non-deep) feature-based UAV image registration.** Luo, Wei, Jin, Wang, Lin, Wei & Zhou, *Fast Automatic Registration of UAV Images via Bidirectional Matching*, Sensors 2023 ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10610563/)). Uses the same core primitives as this project — ORB keypoints, Hamming-distance descriptor matching, and a RANSAC-family outlier filter (PROSAC here vs. `cv2.findHomography`'s RANSAC in `src/localize.py`) — but adds bidirectional (forward/backward) matching consistency checks before accepting a match. That's a concrete, low-effort improvement this MVP doesn't implement: currently a test frame is matched to references in one direction only, with no check that the reference frame's best match points back to the same test frame.
- **GPS-denied UAV localization via reference-imagery matching, with code.** Goforth & Lucey, *GPS-Denied UAV Localization using Pre-existing Satellite Imagery*, ICRA 2019 ([paper](https://publications.ri.cmu.edu/gps-denied-uav-localization-using-pre-existing-satellite-imagery), [code](https://github.com/hmgoforth/gps-denied-uav-localization)). Solves the same top-level problem as this project (replace GPS by matching a live UAV frame against a pre-existing georeferenced reference), but where this MVP uses ORB descriptors, they fine-tune a CNN (VGG16) on satellite imagery specifically because hand-crafted features degrade under the seasonal/appearance and viewpoint differences between UAV and satellite imagery — the same failure mode flagged in this project's limitations section.
- **Viewpoint gap between oblique UAV imagery and nadir reference imagery.** Kinnari, Verdoja & Kyrki, *GNSS-denied geolocalization of UAVs by visual matching of onboard camera images with orthophotos*, 2021 ([paper](https://arxiv.org/abs/2103.14381)). Directly addresses this project's largest unmodeled error source: rather than assuming a nadir (straight-down) camera, they orthorectify the UAV image under a local-planarity assumption before matching, which removes the oblique-viewpoint distortion instead of only correcting its average scale the way this project's slant-range GSD correction does (see Limitations — a single per-frame GSD doesn't model the trapezoidal ground footprint of an oblique shot).

## Algorithm

```
 flight.srt ──► [1] parse_srt ──► telemetry table (t, lat, lon, alt, gimbal)
                                        │
 flight.mp4 ──► [2] extract_frames ─────┤ (attach nearest telemetry per frame)
                                        ▼
                              tagged frames (chronological)
                                        │
                     [3] split 80% / 20% ──────────────┐
                          │                             │
                          ▼                             ▼
                 reference frames                  test frames
                          │                             │
              [4] build_reference                       │
              (ORB features + known GPS                 │
               + resolved heading)                      │
                          │                             │
                          │      ┌─── real-time loop (per test frame) ────┐
                          │      │  motion model PREDICTS position         │
                          └──────┤  [5] query ReferenceSource for a        │
                                 │      RADIUS OF INTEREST (dynamic tiles)  │
                                 │  [5] match neighborhood (BF+ratio),      │
                                 │      take top-K candidates               │
                                 │  [6] homography (RANSAC) per cand.;      │
                                 │      pick best by INLIER count           │
                                 │  accept iff inliers/ratio pass           │
                                 │  [7] center px → m (GSD) → lat/lon,       │
                                 │      rotated by ref heading (UTM)        │
                                 │  INNOVATION GATE: reject impossible jump │
                                 │  ├─ accepted → MAP FIX, update model     │
                                 │  └─ else      → DEAD-RECKON (coast),     │
                                 │                 report flagged position  │
                                 └──────────────────────────────────────────┘
                                        │
                              [8] evaluate vs true GPS (median/mean/rmse/p90/max)
                                        │
                              [9] visualize trajectory (PNG)
```

1. **Parse telemetry** (`src/parse_srt.py`): regex-based parser reads the
   DJI `.srt` file's subtitle blocks and extracts timestamp, GPS lat/lon,
   altitude, and gimbal pitch/yaw (when present). Blocks that don't contain
   parsable GPS fields are logged and skipped, not treated as fatal errors,
   since real-world DJI SRT files vary in field naming across firmware
   versions.
2. **Extract frames** (`src/extract_frames.py`): `ffmpeg` extracts frames at
   a fixed rate (default 1 Hz, configurable). Each frame is tagged with the
   nearest telemetry record by timestamp.
3. **Split reference/test** (`src/build_reference.py`): the first 80% of
   frames (chronologically) become the "known" reference set; the last 20%
   become the test set, whose GPS is only used for evaluation afterward.
4. **Build reference index** (`src/build_reference.py`): ORB keypoints and
   descriptors are computed for every reference frame and kept in memory
   together with that frame's known GPS, altitude, gimbal angle, and a
   **resolved heading** (measured gimbal yaw if present, else the GPS
   travel-bearing estimate). `ReferenceEntry` is deliberately *source-agnostic*
   (see "Extending to GIS / Google Earth" below): today each entry is a
   same-flight frame, but the same struct describes a georeferenced map tile.
5. **Match test frame → reference, with a motion gate** (`src/localize.py`,
   `src/motion.py`, `src/reference_source.py`): a constant-velocity **motion
   model** — seeded from the reference track and propagated using only previous
   *estimates* (no test-frame GNSS) — predicts where the drone is, and the
   **`ReferenceSource` is queried for only the views within a radius of interest**
   of that prediction. This both eliminates *perceptual aliasing* (matching a
   look-alike from a distant part of the flight) and is the hook for **lazily
   loading map tiles instead of holding the whole map** in memory. ORB
   descriptors are brute-force Hamming-matched (Lowe ratio test) against that
   neighborhood, keeping the top-K candidates by good-match count.
6. **Homography + inlier confidence** (`src/localize.py`): each top-K candidate
   is verified with a `cv2.findHomography` RANSAC fit, and the candidate with
   the most **geometric inliers** is chosen — *not* the one with the most raw
   matches, which is what previously picked confident-but-wrong frames. A frame
   is accepted only if it clears `--min-inliers` (default 6) and
   `--min-inlier-ratio` (default 0.12); otherwise it's an honest localization
   failure, not a fabricated position. If the gate is *empty* (no reachable
   reference), a global re-acquisition search runs under a stricter bar.
7. **Center pixel → ground coordinate** (`src/localize.py`): the test frame's
   center is mapped through the homography into the reference frame's pixel
   space; the pixel offset from the reference center is converted to meters via
   ground sample distance (slant range from **barometric altitude** + the DJI
   Mini 3 Pro's ~82.1° diagonal FOV split by the 16:9 aspect ratio, accounting
   for the ~60° gimbal tilt), rotated into world East/North by the reference
   frame's resolved heading, and applied to the reference GPS via UTM. The
   result is the ground coordinate under the video center — the assignment's
   target output. Finally an **innovation gate** rejects any estimate implying
   an impossible jump from the last confirmed fix (a standard tracking-filter
   check); accepted estimates update the motion model for the next frame.
7b. **Dead-reckon fallback** (`src/localize.py`, `src/motion.py`): if no match
   is trustworthy (or the innovation gate rejected it), the frame is **not**
   dropped — the navigator reports a **dead-reckoned** position by coasting the
   motion model (capped extrapolation, then hold-near-last-fix so it can't run
   away), flagged with a `dead_reckon` mode and a coast counter. Graceful
   degradation, like an INS bridging a GNSS dropout.
8. **Evaluate** (`src/evaluate.py`): for every successfully localized test
   frame, the haversine distance (meters) to the SRT's true GPS is computed.
   Because the error distribution is bimodal, **median / mean / RMSE / p90 /
   max** are all reported, plus a breakdown of in-gate vs. re-acquired matches.
9. **Visualize** (`src/visualize.py`): the true GPS trajectory (full flight,
   from the SRT) and the estimated positions (test frames only) are
   projected into local x/y meters via UTM and plotted together, saved to
   `results/trajectory_comparison.png`.

## Real-time navigation loop

The localizer is structured as an online predict → gate → match → correct loop
(`localize_stream` in `src/localize.py`, which `localize_all` collects), which is
exactly the shape a real-time navigator needs:

1. **Predict** the next position from the motion model (constant velocity,
   dead-reckoned from the last fix — this alone works during short visual
   dropouts).
2. **Gate** the reference index to physically reachable candidates. This also
   makes matching *cheaper* as the reference map grows: instead of brute-forcing
   every reference frame, only a local neighborhood is searched.
3. **Match + verify** within the gate, accepting only on geometric inlier
   confidence.
4. **Correct or coast**: fold an accepted fix back into the motion model; reject
   impossible jumps (innovation gate); and when no fix is trustworthy, **report a
   dead-reckoned position** (capped coast) rather than dropping the frame.

Nothing in the loop uses the test frame's GNSS, so it is a faithful stand-in for
the real GPS-denied case. The current bottleneck for true real-time is the
brute-force ORB matcher; a spatial index over the reference set (e.g. a k-d tree
on positions, already half-enabled by the radius query) plus a
BoW/vocabulary-tree or FAISS descriptor index is the natural next step.

## Limitations / future work

- **No yaw compensation without SRT yaw**: if the telemetry doesn't include
  gimbal/camera yaw, a travel-bearing estimate (direction of travel between
  nearby GPS samples) is used as a proxy, falling back further to a
  north-aligned camera only if that's unavailable too (e.g. a hovering
  reference frame with negligible GPS displacement). The travel-bearing proxy
  assumes the camera faces the direction of travel, which breaks down during
  hovering, orbiting, or independent gimbal panning — this flight's SRT has
  no yaw field at all, so this assumption was active for every test frame,
  and it's the likely explanation for the largest remaining errors (frames
  near the loop's turn, where the drone wasn't flying straight).
- **Classical (ORB) features only**: by design, this MVP uses only classical
  OpenCV features, not deep-learned matchers (SuperGlue, LoFTR, etc.). This
  keeps the dependency list light and setup fast, but it would likely fail on
  cross-view matching against satellite/GIS reference imagery, since the
  viewpoint gap between an oblique drone frame and a nadir satellite image is
  much larger than between two drone frames from the same flight — a known
  harder extension, out of scope here.
- **Same-flight reference only (by default)**: the reference set is drawn from
  the same video as the test set, not from an independent map or an orthomosaic.
  It demonstrates the localization mechanism, not a production mapping pipeline —
  but the `ReferenceSource` seam (see "Extending to GIS / Google Earth") is
  exactly what an independent-map / Google-Earth backend plugs into.
- **Dead reckoning is short-gap only**: with no IMU and unreliable visual
  odometry on this data (see "Does it need prior knowledge of the course?"),
  motion-only bridging is capped and coarse (~100 m here). It keeps 24/24
  coverage but is not a substitute for map fixes; long dropouts need a denser
  map / better matcher, not better coasting.
- **No lens distortion correction**: frames are matched and projected as-is,
  without correcting for camera lens distortion.
- **Still only a single, uniform ground sample distance per frame**: the
  meters-per-pixel conversion now uses slant range (accounting for gimbal
  tilt away from nadir — see step 7 above) rather than raw altitude, which
  corrects the systematic bias of a pure nadir assumption. It still applies
  one GSD value to the whole frame, though: a truly oblique shot's ground
  footprint is trapezoidal (the top of the frame is farther from the camera,
  and therefore covers more ground per pixel, than the bottom), which this
  MVP does not model per-row.
- **ORB may fail on low-texture terrain**: water, uniform fields, or other
  textureless surfaces can yield too few keypoints for reliable matching,
  which is exactly why failed localizations are explicitly detected and
  excluded rather than silently producing bad estimates.

## Extending to GIS / Google Earth (final-project seam)

The professor's final-project idea — *"real-time visual navigation based on
predefined (annotated) previous videos and GIS datasets such as Google Earth"* —
is a change of **reference source only**. Everything downstream (gating,
matching, inlier verification, pixel→ground projection, motion model, dead
reckoning, evaluation) is already source-agnostic. Two seams make this concrete:

- **`ReferenceEntry`** (`src/build_reference.py`) — one georeferenced,
  feature-indexed view. It carries a center `latitude/longitude`, an
  `altitude`-equivalent scale, a `heading_deg`, ORB features, and a `source` tag.
  Today it's a same-flight frame; the same struct describes a map tile.
- **`ReferenceSource`** (`src/reference_source.py`) — access to the map via
  `query(lat, lon, radius)` + `all()`. The navigator only ever asks for the
  *radius of interest* around its current estimate, so a backend is free to load
  and evict tiles on demand: **constant memory regardless of map size**, and
  matching cost bounded by the local neighborhood (real-time as the map grows).

To plug in Google Earth / GIS you would:

1. **Implement `TiledMapReferenceSource(ReferenceSource)`**: given the query
   (lat, lon, radius), load the georeferenced tiles overlapping that disc from
   disk/network, build a `ReferenceEntry` per tile (lat/lon = tile center, GSD =
   the tile's known meters-per-pixel, `source="gis_tile"`), cache them, and drop
   tiles that fall outside the radius. This directly realizes the
   "dynamically change the photos based on a prefixed radius of interest" idea —
   nothing else in the pipeline changes, because it already talks to the map only
   through `query`/`all`.
2. **Handle the wider viewpoint gap.** Map tiles are typically nadir while drone
   frames here are oblique (60°); classical ORB will struggle across that gap.
   This is the point to swap the matcher for a learned one (SuperPoint+SuperGlue,
   LoFTR) or to orthorectify the drone frame first (local-planarity assumption)
   — the matcher lives behind `match_candidates`, so it can be replaced without
   touching the rest of the loop.
3. **Everything else is reused as-is**: give the drone a **start position**, and
   the same predict → query-radius → match → correct → (dead-reckon on gaps) loop
   runs against the map. Motion model, spatial + innovation gates, inlier
   confidence, center-pixel→ground projection, and the evaluation harness all
   operate on `ReferenceEntry`/`ReferenceSource`/`LocalizationResult` and don't
   care whether a reference came from a prior video or Google Earth.

This keeps the current same-flight MVP as the honest, runnable baseline while
laying the exact interface — radius-queried reference source + swappable matcher
+ start-position-and-coast loop — that the GIS extension slots into.
