"""Steps 5-7: match each test frame against the reference index, estimate a
homography to the best-matching reference frame, and convert the resulting
pixel offset into an estimated GPS position -- the ground point under the video
center (the coordinate the assignment asks for).

Two things make this more than brute-force image matching, and both lean on the
telemetry sensors rather than pixels alone:

* **Motion-gated matching** (see `src/motion.py`): a constant-velocity model,
  seeded from the reference track, restricts candidate reference frames to those
  physically reachable from the current estimate. This is what kills the
  perceptual-aliasing errors (a frame matching a look-alike from a distant part
  of the flight). It uses no test-frame GNSS.
* **Inlier-based confidence + selection**: among the gated candidates we don't
  trust raw good-match count (which happily picked wrong frames before); we
  verify each with a RANSAC homography and pick / accept based on the geometric
  inlier count, which actually discriminates a correct match from a confident
  false one.

Altitude (barometric, GNSS-independent) and camera angle feed the pixel->meters
conversion; the reference frame's resolved heading rotates that offset into
world East/North.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import utm

from .build_reference import ReferenceEntry, compute_orb_features
from .extract_frames import FrameRecord
from .geo import offset_latlon
from .gis_reference import normalize_for_matching, rescale_frame_to_map
from .motion import MotionState, seed_from_reference_track
from .reference_source import ReferenceSource, as_reference_source

# DJI Mini 3 Pro: ~82.1 degree diagonal FOV, 16:9 sensor aspect ratio.
# Named clearly so it can be corrected later if the actual spec differs.
DJI_MINI_3_PRO_DIAGONAL_FOV_DEG = 82.1
SENSOR_ASPECT_W = 16
SENSOR_ASPECT_H = 9

DEFAULT_RATIO_THRESH = 0.75
DEFAULT_MIN_GOOD_MATCHES = 15
# Acceptance is driven by RANSAC inliers, not raw good matches: a correct
# homography between overlapping aerial frames yields many geometrically
# consistent inliers, while a confident *false* match scatters. These thresholds
# are what separate "confident-correct" from "confident-wrong".
DEFAULT_MIN_INLIERS = 6
DEFAULT_MIN_INLIER_RATIO = 0.12
# How many top good-match candidates (within the motion gate) to verify with a
# homography before choosing. Small, so this stays real-time even before gating.
DEFAULT_TOP_K = 5

# DJI gimbal pitch convention: 0 = horizontal (forward), -90 = nadir (straight
# down). Used as a fallback only when a frame's telemetry has no gimbal pitch
# reading (true for this project's real flight SRT) -- matches the ~60 degree
# down-from-horizontal gimbal angle this flight was shot at. Named clearly so
# it can be corrected if a different flight's actual angle differs.
ASSUMED_GIMBAL_PITCH_DEG = -60.0

# Search radius at which the acceptance thresholds are the quoted defaults;
# wider searches scale them up proportionally (see localize_frame).
GATE_RADIUS_REFERENCE_M = 120.0

# Map views that are north-up georeferenced raster, rather than an oblique frame
# from a previous flight. They need the drone frame resampled to their ground
# scale and contrast-equalised before ORB will match anything (see FrameView).
RASTER_SOURCES = frozenset({"gis_tile"})


@dataclass
class LocalizationResult:
    frame_path: str
    timestamp_sec: float
    true_latitude: float
    true_longitude: float
    success: bool                 # True only for a trusted visual (map) fix
    # How the reported position was obtained:
    #   "map_fix"     - accepted visual match against the reference/map
    #   "dead_reckon" - no trustworthy match; coasted on the motion model
    #   "failed"      - no estimate at all (no match and no motion state)
    mode: str = "failed"
    estimated_latitude: Optional[float] = None
    estimated_longitude: Optional[float] = None
    matched_ref_frame: Optional[str] = None
    # Which map produced this fix -- "flight_frame" for previously-flown video,
    # "gis_tile" for a satellite/orthophoto raster. On a hybrid map the two
    # compete per frame, so this is what the source split is counted from.
    matched_source: Optional[str] = None
    num_good_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    num_refs_fused: int = 1       # how many reference views were fused into this fix
    gated: bool = False           # was the match found within the motion gate?
    coast_steps: int = 0          # consecutive dead-reckoned frames (0 for a fix)
    failure_reason: Optional[str] = None

    @property
    def has_estimate(self) -> bool:
        return self.estimated_latitude is not None


def horizontal_vertical_fov_deg(
    diagonal_fov_deg: float = DJI_MINI_3_PRO_DIAGONAL_FOV_DEG,
    aspect_w: int = SENSOR_ASPECT_W,
    aspect_h: int = SENSOR_ASPECT_H,
) -> tuple[float, float]:
    """Derive horizontal and vertical FOV from a diagonal FOV + sensor aspect ratio."""
    diag_ratio = math.hypot(aspect_w, aspect_h)
    half_diag_rad = math.radians(diagonal_fov_deg / 2)
    horiz_fov_rad = 2 * math.atan((aspect_w / diag_ratio) * math.tan(half_diag_rad))
    vert_fov_rad = 2 * math.atan((aspect_h / diag_ratio) * math.tan(half_diag_rad))
    return math.degrees(horiz_fov_rad), math.degrees(vert_fov_rad)


def slant_range_m(altitude_m: float, gimbal_pitch_deg: Optional[float]) -> float:
    """Distance from the camera to the ground point at the center of the
    frame, accounting for gimbal tilt away from nadir (falls back to
    ASSUMED_GIMBAL_PITCH_DEG if the telemetry has no pitch reading). Equal to
    `altitude_m` exactly at nadir, and grows as the gimbal tilts toward
    horizontal.
    """
    pitch_deg = gimbal_pitch_deg if gimbal_pitch_deg is not None else ASSUMED_GIMBAL_PITCH_DEG
    angle_from_nadir_deg = 90.0 + pitch_deg  # 0 at nadir, 90 at horizontal
    angle_from_nadir_deg = min(max(angle_from_nadir_deg, 0.0), 89.0)  # guard against tan/cos blowup
    return altitude_m / math.cos(math.radians(angle_from_nadir_deg))


def ground_sample_distance(
    altitude_m: float,
    image_width_px: int,
    image_height_px: int,
    gimbal_pitch_deg: Optional[float] = None,
) -> tuple[float, float]:
    """Meters-per-pixel (x, y) at the reference frame's center.

    Uses the slant range to the frame's center (accounting for gimbal tilt
    away from nadir) rather than raw altitude, correcting the systematic
    underestimate a straight-nadir assumption would produce on an oblique
    gimbal angle (e.g. -60 degrees). This is still a single uniform GSD
    applied across the whole frame -- it does not model the trapezoidal
    ground footprint of an oblique shot (top-of-frame vs. bottom-of-frame
    have different true GSD); see README limitations.
    """
    distance_m = slant_range_m(altitude_m, gimbal_pitch_deg)
    horiz_fov_deg, vert_fov_deg = horizontal_vertical_fov_deg()
    ground_width_m = 2 * distance_m * math.tan(math.radians(horiz_fov_deg / 2))
    ground_height_m = 2 * distance_m * math.tan(math.radians(vert_fov_deg / 2))
    gsd_x = ground_width_m / image_width_px
    gsd_y = ground_height_m / image_height_px
    return gsd_x, gsd_y


def match_candidates(
    test_descriptors: np.ndarray,
    candidate_entries: list[ReferenceEntry],
    ratio_thresh: float = DEFAULT_RATIO_THRESH,
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[ReferenceEntry, list]]:
    """Brute-force Hamming + Lowe-ratio match the test descriptors against each
    candidate reference entry, and return the `top_k` entries with the most good
    matches, each paired with its good-match list.

    `candidate_entries` is already the (motion-gated) neighborhood to search --
    the reference source does the spatial filtering. Returning several candidates
    (not just the single best) lets the caller pick by geometric inliers rather
    than raw match count. This is the one function a GIS/learned-matcher backend
    would replace.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    scored: list[tuple[int, ReferenceEntry, list]] = []

    for entry in candidate_entries:
        if entry.descriptors is None or len(entry.descriptors) < 2:
            continue
        knn = bf.knnMatch(test_descriptors, entry.descriptors, k=2)
        good = [m for m, n in knn if m.distance < ratio_thresh * n.distance]
        if good:
            scored.append((len(good), entry, good))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [(entry, good) for _n, entry, good in scored[:top_k]]


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """z-component of the cross product of two 2-D vectors.

    NumPy 2.0 removed `np.cross`'s 2-D form, so spell it out rather than depend
    on a version-specific behaviour.
    """
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


def homography_is_plausible(H: np.ndarray, width: int, height: int,
                            max_area_ratio: float = 4.0) -> bool:
    """Reject homographies that are geometrically impossible for this problem.

    RANSAC will happily return a *high-inlier* transform from points that are
    clustered or near-collinear -- which is exactly what happens on a map patch
    whose content sits in a thin strip along one edge. The fit is degenerate: it
    collapses or flips the frame and throws the mapped centre hundreds of metres
    away, while reporting 40-60 inliers. Nothing downstream could tell that from
    a good match, so it has to be caught here.

    Two cheap, standard checks on the frame's projected outline:
      * it must stay a convex quadrilateral (no folding or mirroring), and
      * its area must stay within a factor of `max_area_ratio` of the original
        (two overlapping aerial views differ in scale by a little, not by 10x).
    """
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        return False

    # Convex and consistently wound: every cross product must share a sign.
    signs = []
    for i in range(4):
        a, b, c = projected[i], projected[(i + 1) % 4], projected[(i + 2) % 4]
        signs.append(np.sign(_cross2(b - a, c - b)))
    if len(set(s for s in signs if s != 0)) != 1:
        return False

    area = 0.5 * abs(_cross2(projected[2] - projected[0], projected[3] - projected[1]))
    original = float(width) * float(height)
    if area <= 0:
        return False
    ratio = area / original
    return (1.0 / max_area_ratio) <= ratio <= max_area_ratio


def estimate_homography(test_kp, ref_kp, good_matches) -> tuple[Optional[np.ndarray], int, float, Optional[np.ndarray]]:
    """RANSAC homography mapping test-frame pixel coords -> reference-frame pixel
    coords. Returns (H, num_inliers, inlier_ratio, inlier_mask);
    (None, 0, 0.0, None) if it can't be estimated. The inlier count/ratio are the
    match's geometric confidence; the mask says *which* matches were inliers,
    which the trace/player uses to draw the surviving correspondences.
    """
    if len(good_matches) < 4:
        return None, 0, 0.0, None
    src_pts = np.float32([test_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None, 0, 0.0, None
    num_inliers = int(mask.sum())
    inlier_ratio = num_inliers / len(good_matches)
    return H, num_inliers, inlier_ratio, mask.ravel().astype(bool)


def pixel_offset_to_latlon(
    H: np.ndarray,
    ref_entry: ReferenceEntry,
    test_image_width: int,
    test_image_height: int,
) -> tuple[float, float]:
    """Map the test frame's center through H into the reference frame, convert
    the resulting pixel offset (from the reference frame's center) into a
    meters offset via ground sample distance, rotate it by the reference
    frame's heading, and apply it to the reference frame's known GPS point.

    The returned point is the ground coordinate under the *test* frame's center.
    """
    test_center = np.array([[test_image_width / 2, test_image_height / 2]], dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(test_center, H)
    mapped_x, mapped_y = mapped[0, 0]

    ref_center_x = ref_entry.image_width / 2
    ref_center_y = ref_entry.image_height / 2
    dx_px = mapped_x - ref_center_x
    dy_px = mapped_y - ref_center_y

    # Prefer the scale measured from the reference track during preprocessing
    # (src/calibrate.py) over the modelled one: on these flights the model is
    # ~36% low, because DJI's rel_alt is height above the take-off point rather
    # than above the imaged ground, and the SRT carries no gimbal angle at all.
    if getattr(ref_entry, "gsd_m_per_px", None):
        gsd_x = gsd_y = ref_entry.gsd_m_per_px
    else:
        gsd_x, gsd_y = ground_sample_distance(
            ref_entry.altitude, ref_entry.image_width, ref_entry.image_height, ref_entry.gimbal_pitch
        )
    dx_m = dx_px * gsd_x
    dy_m = dy_px * gsd_y

    # Image +x = right, +y = down. At heading=0 (north-aligned camera) we treat
    # +x as East and +y (down) as South, i.e. -North.
    east0_m = dx_m
    north0_m = -dy_m

    # Rotate the image-plane offset into world axes using the reference frame's
    # resolved heading (measured gimbal yaw > GPS travel-bearing estimate,
    # resolved once in build_reference). None means neither was available (e.g.
    # a hovering reference frame with no yaw field) -> assume north-aligned. The
    # motion gate keeps the matched reference close, so this offset is small and
    # a heading error here costs little.
    yaw_deg = ref_entry.heading_deg if ref_entry.heading_deg is not None else 0.0
    yaw_rad = math.radians(yaw_deg)
    east_m = east0_m * math.cos(yaw_rad) + north0_m * math.sin(yaw_rad)
    north_m = -east0_m * math.sin(yaw_rad) + north0_m * math.cos(yaw_rad)

    return offset_latlon(ref_entry.latitude, ref_entry.longitude, east_m, north_m)


def _verify_candidates(test_kp, candidates, img_w, img_h, rejected: Optional[list] = None):
    """Verify each (entry, good_matches) candidate with a RANSAC homography and
    return, best-inliers-first, a dict per candidate that survived the fit:
    {entry, good, n_in, ratio, lat, lon, H, inlier_mask}. The lat/lon is that
    candidate's independent position estimate (so the caller can fuse several).

    `rejected`, when a list is passed, collects (entry, n_good, reason) for the
    candidates thrown out here. Purely diagnostic -- it is what lets the live
    player explain *why* a confident-looking candidate was discarded.
    """
    out = []
    for entry, good in candidates:
        H, n_in, ratio, mask = estimate_homography(test_kp, entry.keypoints, good)
        if H is None:
            if rejected is not None:
                rejected.append((entry, len(good), "no homography"))
            continue
        if not homography_is_plausible(H, img_w, img_h):
            if rejected is not None:
                rejected.append((entry, len(good), "degenerate homography"))
            continue
        lat, lon = pixel_offset_to_latlon(H, entry, img_w, img_h)
        # The frame centre must land on (or near) the reference view it matched.
        # Landing far outside means the fit is describing something other than
        # this overlap, however many inliers it collected.
        mapped = cv2.perspectiveTransform(
            np.float32([[img_w / 2, img_h / 2]]).reshape(-1, 1, 2), H).reshape(2)
        margin_x, margin_y = entry.image_width * 0.75, entry.image_height * 0.75
        if not (-margin_x <= mapped[0] <= entry.image_width + margin_x
                and -margin_y <= mapped[1] <= entry.image_height + margin_y):
            if rejected is not None:
                rejected.append((entry, len(good), "centre projects off the reference view"))
            continue
        out.append({"entry": entry, "good": good, "n_in": n_in, "ratio": ratio,
                    "lat": lat, "lon": lon, "H": H, "inlier_mask": mask})
    out.sort(key=lambda d: d["n_in"], reverse=True)
    return out


def _fuse_positions(accepted):
    """Inlier-count-weighted average of several candidates' position estimates
    (in UTM, so the averaging is metric). Returns (lat, lon).

    OFF by default: tested across the four flights it was a wash-to-negative --
    it helped the hardest high-altitude flight (flight_0017: mean 109->97 m) but
    hurt others (the baseline flight: max 93->143 m), because averaging in a
    spatially-different-but-still-accepted reference (e.g. a loop revisit) pulls
    the estimate off when the single best match was already good. Kept as an
    opt-in (`--fuse`) and an honest recorded negative result rather than a
    default. The robust default is the single best-inlier match.
    """
    if len(accepted) == 1:
        return accepted[0]["lat"], accepted[0]["lon"]
    _e0, _n0, zn, zl = utm.from_latlon(accepted[0]["lat"], accepted[0]["lon"])
    we = wn = w = 0.0
    for d in accepted:
        e, n, _, _ = utm.from_latlon(d["lat"], d["lon"], force_zone_number=zn, force_zone_letter=zl)
        weight = d["n_in"]
        we += e * weight
        wn += n * weight
        w += weight
    return utm.to_latlon(we / w, wn / w, zn, zl)


def _view_kind(entry: ReferenceEntry) -> str:
    """Which preparation of the drone frame this map view needs.

    A north-up raster tile and an oblique flight frame are not comparable at the
    same scale: the tile is typically two to five times coarser, and it has been
    contrast-equalised. `source` is the honest discriminator -- an entry's
    `gsd_m_per_px` is set on both kinds once self-calibration has run, so it
    cannot be used to tell them apart.
    """
    return "raster" if entry.source in RASTER_SOURCES else "native"


@dataclass
class FrameView:
    """The drone frame prepared for one kind of map, with its ORB features.

    A hybrid map needs two of these from the same JPEG, which is why the
    preparation that used to be inline in `localize_frame` is a value now: every
    downstream step (homography, plausibility, pixel->metres) has to use the
    width and height of the view the match was actually found in.
    """

    image: np.ndarray
    keypoints: list
    descriptors: Optional[np.ndarray]
    rescale_factor: float = 1.0
    clahe: bool = False

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def is_usable(self) -> bool:
        return self.descriptors is not None and len(self.keypoints) > 0


def prepare_frame_view(
    gray: np.ndarray,
    n_features: int,
    map_gsd_m_per_px: Optional[float] = None,
    frame_gsd_m_per_px: Optional[float] = None,
) -> FrameView:
    """Resample and equalise the frame for one map, then extract its features.

    Against a raster map the drone frame is typically several times finer than
    the map. ORB is only weakly scale invariant, so it is resampled to the map's
    ground scale first; otherwise almost nothing matches. Contrast equalisation
    is applied to the frame exactly as `build_gis_index` applied it to the map --
    normalising only one side changes ORB's intensity comparisons on that side
    alone, which weakens true matches and lets confident false ones win.

    With no `map_gsd_m_per_px` this is the identity plus feature extraction,
    which is what an oblique flight-frame map wants.
    """
    image, factor, clahe = gray, 1.0, False
    if map_gsd_m_per_px:
        if frame_gsd_m_per_px:
            image, factor = rescale_frame_to_map(image, frame_gsd_m_per_px, map_gsd_m_per_px)
        image = normalize_for_matching(image)
        clahe = True
    keypoints, descriptors = compute_orb_features(image, n_features=n_features)
    return FrameView(image=image, keypoints=keypoints, descriptors=descriptors,
                     rescale_factor=factor, clahe=clahe)


def localize_frame(
    test_frame: FrameRecord,
    source: ReferenceSource,
    motion_state: Optional[MotionState] = None,
    n_features: int = 2000,
    ratio_thresh: float = DEFAULT_RATIO_THRESH,
    min_good_matches: int = DEFAULT_MIN_GOOD_MATCHES,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
    top_k: int = DEFAULT_TOP_K,
    fuse_references: bool = False,
    exclusion_sec: float = 0.0,
    frame_gsd_m_per_px: Optional[float] = None,
    map_gsd_m_per_px: Optional[float] = None,
    frame_scale_per_alt: Optional[float] = None,
    max_evidence_scale: float = 1.0,
    fallback_order: Optional[tuple] = None,
    debug: Optional[dict] = None,
) -> LocalizationResult:
    """Attempt a single visual (map) fix for one test frame.

    If `motion_state` is given, the reference source is queried only for views
    within the motion gate (radius of interest around the predicted position);
    if that neighborhood is empty, we fall back to a global re-acquisition search
    under a stricter acceptance bar. Every candidate that passes the inlier bar
    contributes an independent position estimate; with `fuse_references` on, those
    are inlier-weighted-averaged (lower variance). A failure here is not the final
    word -- `localize_all` falls back to dead reckoning.

    Candidates are grouped by the preparation of the frame they need (see
    `_view_kind`) and each group is matched against its own `FrameView`. With a
    single-source map that is one group and one view, exactly as before; with a
    hybrid satellite-plus-flight map it is two, and there are two policies:

    * `fallback_order=("native", "raster")` -- **tiers**. Try the previous
      flight's frames; only if nothing there is trustworthy, fall back to the
      GIS raster. Over ground the drone has flown before, same-sensor imagery is
      far more reliable, so the satellite map is a safety net for the gaps
      rather than a competitor.
    * `fallback_order=None` -- **compete**. Match both, keep whichever produced
      more inliers.

    `debug`, when a dict is passed, is filled in with this frame's internals
    (the image actually matched, its keypoints, the motion prediction and gate,
    every candidate considered and why it was kept or dropped). It changes
    nothing about the decision -- it is how `src/trace.py` shows the live player
    what this function did, without the player re-implementing any of it.
    """
    # Phase timing, for the real-time claim. Only accumulated when a caller asks
    # for diagnostics, so the batch pipeline pays nothing for it.
    clock = {"decode_ms": 0.0, "prepare_ms": 0.0, "gate_ms": 0.0,
             "match_ms": 0.0, "verify_ms": 0.0}
    started = time.perf_counter()

    tick = time.perf_counter()
    gray = cv2.imread(test_frame.frame_path, cv2.IMREAD_GRAYSCALE)
    clock["decode_ms"] = (time.perf_counter() - tick) * 1e3
    base = dict(
        frame_path=test_frame.frame_path,
        timestamp_sec=test_frame.timestamp_sec,
        true_latitude=test_frame.latitude,
        true_longitude=test_frame.longitude,
    )

    if gray is None:
        return LocalizationResult(**base, success=False, failure_reason="could not read image")

    # This frame's own ground scale, from its barometric altitude where we have
    # it. A single flight-wide number is wrong the moment altitude varies: on a
    # flight that includes take-off and landing, a frame shot at 6 m sees ground
    # five times finer than one at 31 m, and resampling them all by the same
    # factor makes the low ones unmatchable. Altitude is GNSS-independent, so
    # using it here costs nothing in fairness.
    this_gsd = frame_gsd_m_per_px
    if frame_scale_per_alt and test_frame.altitude:
        this_gsd = frame_scale_per_alt * test_frame.altitude

    views: dict[str, FrameView] = {}

    def view_for(kind: str) -> FrameView:
        """Prepare a view once, on first demand -- ORB on a 1920x1080 frame is
        not free, and a single-source map must not pay for the other kind."""
        if kind not in views:
            tick = time.perf_counter()
            views[kind] = prepare_frame_view(
                gray, n_features,
                map_gsd_m_per_px=map_gsd_m_per_px if kind == "raster" else None,
                frame_gsd_m_per_px=this_gsd if kind == "raster" else None)
            clock["prepare_ms"] += (time.perf_counter() - tick) * 1e3
        return views[kind]

    # Ask the reference source only for the neighborhood the drone could be in
    # (radius of interest) -- this is both the anti-aliasing gate and the hook
    # for lazily loading map tiles instead of holding the whole map.
    ts = test_frame.timestamp_sec
    gated = False
    if motion_state is not None:
        tick = time.perf_counter()
        pred_lat, pred_lon = motion_state.predict(ts)
        radius = motion_state.gating_radius_m(ts)
        candidate_entries = source.query(pred_lat, pred_lon, radius)
        clock["gate_ms"] += (time.perf_counter() - tick) * 1e3
        gated = True
        # Optionally demand stronger evidence from a wider search (a wide gate
        # sees more candidates, so more chances for a look-alike to clear a fixed
        # bar). OFF by default (--max-evidence-scale 1.0): it was added to stop a
        # 380 m gate on flight_0024 accepting a 7-inlier match 300 m away, but
        # once `homography_is_plausible` rejected that match on geometry instead,
        # scaling only cost recall -- measured worse or tied on every flight and
        # both map sources. Kept as an honest recorded negative, like --fuse.
        evidence_scale = min(max_evidence_scale, max(1.0, radius / GATE_RADIUS_REFERENCE_M))
        min_inliers = int(round(min_inliers * evidence_scale))
        min_good_matches = int(round(min_good_matches * evidence_scale))
        if debug is not None:
            debug["predicted_latlon"] = (pred_lat, pred_lon)
            debug["gate_radius_m"] = radius
            debug["evidence_scale"] = evidence_scale
    else:
        candidate_entries = source.all()
    if debug is not None:
        debug["n_in_gate"] = len(candidate_entries)

    # Temporal exclusion: refuse to localize against reference views recorded
    # within `exclusion_sec` of this frame. With an interleaved hold-out the
    # neighbouring map frames are only ~1 s away, which would make the match
    # trivial and the reported accuracy meaningless.
    if exclusion_sec > 0:
        candidate_entries = [e for e in candidate_entries
                             if abs(e.timestamp_sec - ts) >= exclusion_sec]

    if debug is not None:
        debug["n_searched"] = len(candidate_entries)
        debug["searched_latlon"] = [(e.latitude, e.longitude) for e in candidate_entries]

    def accepted_from(entries, min_in, min_r):
        """Match and verify `entries`, grouped by the frame view each one needs.

        `top_k` is applied per group, so each map gets its own shortlist rather
        than the weaker one being crowded out before it is ever verified.
        """
        by_kind: dict[str, list] = {}
        for entry in entries:
            by_kind.setdefault(_view_kind(entry), []).append(entry)

        verified_all, rejected_all = [], []

        def try_group(kind):
            view = view_for(kind)
            if not view.is_usable:
                return []
            tick = time.perf_counter()
            cands = match_candidates(view.descriptors, by_kind[kind], ratio_thresh, top_k)
            clock["match_ms"] += (time.perf_counter() - tick) * 1e3
            rejected = [] if debug is not None else None
            tick = time.perf_counter()
            verified = _verify_candidates(view.keypoints, cands, view.width, view.height, rejected)
            clock["verify_ms"] += (time.perf_counter() - tick) * 1e3
            for d in verified:
                d["view"] = kind
            verified_all.extend(verified)
            if rejected:
                rejected_all.extend(rejected)
            return [d for d in verified
                    if d["n_in"] >= min_in and d["ratio"] >= min_r
                    and len(d["good"]) >= min_good_matches]

        kept_all: list = []
        tier_used = None
        if fallback_order:
            # Tiers: stop at the first map that produces something trustworthy,
            # so the fallback is only ever consulted for frames the primary map
            # could not explain -- and is never merely outvoted by it.
            for kind in fallback_order:
                if kind not in by_kind:
                    continue
                kept_all = try_group(kind)
                if kept_all:
                    tier_used = kind
                    break
        else:
            for kind in list(by_kind):
                kept_all.extend(try_group(kind))

        verified_all.sort(key=lambda d: d["n_in"], reverse=True)
        kept_all.sort(key=lambda d: d["n_in"], reverse=True)
        if debug is not None:
            debug["verified"] = verified_all
            debug["rejected"] = rejected_all
            debug["accepted"] = kept_all
            debug["tier_used"] = tier_used
            debug["tiers_available"] = list(by_kind)
            debug["bar"] = {"min_inliers": min_in, "min_inlier_ratio": min_r,
                            "min_good_matches": min_good_matches}
        return kept_all

    accepted = accepted_from(candidate_entries, min_inliers, min_inlier_ratio)
    # Re-acquire only when the gated neighborhood is *empty* (nothing within
    # physical reach) -- NOT merely because the gated match was weak. Falling
    # back on every weak match would just re-find the distant look-alike the gate
    # exists to exclude. A genuine teleport must clear a stricter bar.
    if not candidate_entries and motion_state is not None:
        fallback = source.all()
        if exclusion_sec > 0:
            fallback = [e for e in fallback if abs(e.timestamp_sec - ts) >= exclusion_sec]
        accepted = accepted_from(fallback, 2 * min_inliers, min_inlier_ratio + 0.1)
        gated = False
        if debug is not None:
            debug["reacquisition"] = True
            debug["n_searched"] = len(fallback)
            debug["searched_latlon"] = [(e.latitude, e.longitude) for e in fallback]

    if debug is not None:
        clock["total_ms"] = (time.perf_counter() - started) * 1e3
        clock["n_searched"] = len(candidate_entries)
        debug["timing"] = clock
        debug["gated"] = gated
        # The player draws the view the decision was actually made in.
        shown = views.get(accepted[0]["view"]) if accepted else (
            views.get("raster") or views.get("native"))
        if shown is not None:
            debug["image_shape"] = shown.image.shape
            debug["clahe"] = shown.clahe
            debug["rescale_factor"] = shown.rescale_factor
            debug["frame_gsd_m_per_px"] = this_gsd
            debug["test_kp_xy"] = (np.float32([kp.pt for kp in shown.keypoints])
                                   if shown.keypoints else np.zeros((0, 2), np.float32))

    if not views or not any(v.is_usable for v in views.values()):
        return LocalizationResult(**base, success=False, failure_reason="no ORB features in test frame")

    if not accepted:
        return LocalizationResult(
            **base, success=False, failure_reason="insufficient geometric match confidence",
        )

    best = accepted[0]  # most inliers; used for reporting + as the fusion base
    if debug is not None:
        debug["best"] = best
    if fuse_references:
        est_lat, est_lon = _fuse_positions(accepted)
    else:
        est_lat, est_lon = best["lat"], best["lon"]

    return LocalizationResult(
        **base,
        success=True,
        mode="map_fix",
        estimated_latitude=est_lat,
        estimated_longitude=est_lon,
        matched_ref_frame=best["entry"].frame_path,
        matched_source=best["entry"].source,
        num_good_matches=len(best["good"]),
        num_inliers=best["n_in"],
        inlier_ratio=best["ratio"],
        num_refs_fused=len(accepted) if fuse_references else 1,
        gated=gated,
    )



def localize_stream(
    test_frames: list[FrameRecord],
    reference,
    n_features: int = 2000,
    ratio_thresh: float = DEFAULT_RATIO_THRESH,
    min_good_matches: int = DEFAULT_MIN_GOOD_MATCHES,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
    top_k: int = DEFAULT_TOP_K,
    use_motion: bool = True,
    fuse_references: bool = False,
    exclusion_sec: float = 0.0,
    frame_gsd_m_per_px: Optional[float] = None,
    map_gsd_m_per_px: Optional[float] = None,
    frame_scale_per_alt: Optional[float] = None,
    max_evidence_scale: float = 1.0,
    fallback_order: Optional[tuple] = None,
    seed_latlon: Optional[tuple] = None,
    collect_debug: bool = False,
    verbose: bool = True,
):
    """Run the real-time navigation loop over the test frames in order, yielding
    `(LocalizationResult, debug, motion_state)` after each frame is processed.

    `reference` is a ReferenceSource (or a plain list of entries, which is
    wrapped). The loop is: **predict** the position from the motion model,
    **gate** the map to the radius of interest, **match + verify** visually, and
    then **correct** -- or, when no visual fix is trustworthy, **dead-reckon**
    on the drone's motion and still report a (lower-confidence) position rather
    than rejecting the frame. This mirrors a real GPS-denied navigator that is
    given a starting position and coasts on its motion between visual fixes.

    With `use_motion=False` there is no start position and no dead reckoning:
    unmatched frames are genuine failures (the old pure-matching behavior).

    This is a generator so a caller can drive the navigator **one frame at a
    time** -- which is what `src/trace.py` and the live player do. `localize_all`
    is the same loop, collected. `debug` is None unless `collect_debug` is set;
    `motion_state` is the live object, so a caller that wants a per-frame history
    must copy it.
    """
    assert test_frames, "Sanity check failed: test frame set is empty, nothing to localize."
    source = as_reference_source(reference)
    assert source.all(), "Sanity check failed: reference source is empty, cannot localize."

    if not use_motion:
        motion_state = None
    elif seed_latlon is not None:
        # A raster map has no track to seed from, so the navigator is handed its
        # starting position explicitly -- the assignment's "assuming that the
        # preprocessing data contains the current position". Nothing after this
        # point uses test-frame GNSS.
        seed_lat, seed_lon, seed_t = seed_latlon
        motion_state = MotionState(latitude=seed_lat, longitude=seed_lon, timestamp_sec=seed_t)
    else:
        motion_state = seed_from_reference_track(
            source.all(), start_time_sec=test_frames[0].timestamp_sec)

    for tf in test_frames:
        debug: Optional[dict] = {} if collect_debug else None
        result = localize_frame(
            tf, source, motion_state=motion_state, n_features=n_features,
            ratio_thresh=ratio_thresh, min_good_matches=min_good_matches,
            min_inliers=min_inliers, min_inlier_ratio=min_inlier_ratio, top_k=top_k,
            fuse_references=fuse_references, exclusion_sec=exclusion_sec,
            frame_gsd_m_per_px=frame_gsd_m_per_px, map_gsd_m_per_px=map_gsd_m_per_px,
            frame_scale_per_alt=frame_scale_per_alt, max_evidence_scale=max_evidence_scale,
            fallback_order=fallback_order, debug=debug,
        )

        if motion_state is None:
            # No motion model -> no dead reckoning; report the fix or a failure.
            if not result.success and verbose:
                print(
                    f"[localize] FAILED '{os.path.basename(tf.frame_path)}': {result.failure_reason}.",
                    file=sys.stderr,
                )
            yield result, debug, motion_state
            continue

        # Disbelieve a geometrically-confident but physically impossible jump.
        if result.success and not motion_state.is_output_plausible(
            result.estimated_latitude, result.estimated_longitude, tf.timestamp_sec
        ):
            result.success = False
            result.failure_reason = "rejected by motion innovation gate"
            result.estimated_latitude = result.estimated_longitude = None

        if result.success:
            motion_state.update(result.estimated_latitude, result.estimated_longitude, tf.timestamp_sec)
        else:
            # Graceful degradation: rely on the drone's motion instead of
            # rejecting. The gate stays anchored to the last confirmed fix
            # (mark_miss only widens the search + counts the gap); the reported
            # position is a capped dead-reckon so it can't run away over a long
            # dropout. Drift still accumulates -- coast_steps flags how stale it is.
            motion_state.mark_miss()
            dr_lat, dr_lon = motion_state.dead_reckon_report(tf.timestamp_sec)
            result.mode = "dead_reckon"
            result.estimated_latitude, result.estimated_longitude = dr_lat, dr_lon
            result.coast_steps = motion_state.misses
            if verbose:
                print(
                    f"[localize] DEAD-RECKON '{os.path.basename(tf.frame_path)}' "
                    f"(coast #{result.coast_steps}): no trusted match ({result.failure_reason}).",
                    file=sys.stderr,
                )
        yield result, debug, motion_state


def localize_all(*args, **kwargs) -> list[LocalizationResult]:
    """Run the navigation loop over every test frame and collect the results.

    Thin wrapper over `localize_stream` so the batch pipeline and the live player
    are literally the same loop -- see that generator for the arguments.
    """
    kwargs.pop("collect_debug", None)
    return [result for result, _debug, _motion in localize_stream(*args, **kwargs)]


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python -m src.localize <video_path> <srt_path> <frames_dir> [rate_hz] [max_seconds]")
        sys.exit(1)

    from .parse_srt import parse_srt
    from .extract_frames import extract_frames, attach_telemetry
    from .build_reference import split_reference_test, build_reference_index

    video_path, srt_path, frames_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    rate_hz = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    max_seconds = float(sys.argv[5]) if len(sys.argv) > 5 else None

    telemetry = parse_srt(srt_path)
    raw_frames = extract_frames(video_path, frames_dir, rate_hz=rate_hz, max_seconds=max_seconds)
    tagged_frames = attach_telemetry(raw_frames, telemetry)
    ref_frames, test_frames = split_reference_test(tagged_frames, ref_fraction=0.8)
    ref_index = build_reference_index(ref_frames)

    results = localize_all(test_frames, ref_index)
    n_ok = sum(r.success for r in results)
    print(f"\nLocalized {n_ok}/{len(results)} test frames.")
    for r in results:
        status = "OK" if r.success else f"FAIL ({r.failure_reason})"
        print(f"  {os.path.basename(r.frame_path)} t={r.timestamp_sec:.2f}s -> {status}")
