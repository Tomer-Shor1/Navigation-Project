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
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import utm

from .build_reference import ReferenceEntry, compute_orb_features
from .extract_frames import FrameRecord
from .geo import offset_latlon
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


def estimate_homography(test_kp, ref_kp, good_matches) -> tuple[Optional[np.ndarray], int, float]:
    """RANSAC homography mapping test-frame pixel coords -> reference-frame pixel
    coords. Returns (H, num_inliers, inlier_ratio); (None, 0, 0.0) if it can't
    be estimated. The inlier count/ratio are the match's geometric confidence.
    """
    if len(good_matches) < 4:
        return None, 0, 0.0
    src_pts = np.float32([test_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None, 0, 0.0
    num_inliers = int(mask.sum())
    inlier_ratio = num_inliers / len(good_matches)
    return H, num_inliers, inlier_ratio


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


def _verify_candidates(test_kp, candidates, img_w, img_h):
    """Verify each (entry, good_matches) candidate with a RANSAC homography and
    return, best-inliers-first, a dict per candidate that survived the fit:
    {entry, good, n_in, ratio, lat, lon}. The lat/lon is that candidate's
    independent position estimate (so the caller can fuse several).
    """
    out = []
    for entry, good in candidates:
        H, n_in, ratio = estimate_homography(test_kp, entry.keypoints, good)
        if H is None:
            continue
        lat, lon = pixel_offset_to_latlon(H, entry, img_w, img_h)
        out.append({"entry": entry, "good": good, "n_in": n_in, "ratio": ratio, "lat": lat, "lon": lon})
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
) -> LocalizationResult:
    """Attempt a single visual (map) fix for one test frame.

    If `motion_state` is given, the reference source is queried only for views
    within the motion gate (radius of interest around the predicted position);
    if that neighborhood is empty, we fall back to a global re-acquisition search
    under a stricter acceptance bar. Every candidate that passes the inlier bar
    contributes an independent position estimate; with `fuse_references` on, those
    are inlier-weighted-averaged (lower variance). A failure here is not the final
    word -- `localize_all` falls back to dead reckoning.
    """
    image = cv2.imread(test_frame.frame_path, cv2.IMREAD_GRAYSCALE)
    base = dict(
        frame_path=test_frame.frame_path,
        timestamp_sec=test_frame.timestamp_sec,
        true_latitude=test_frame.latitude,
        true_longitude=test_frame.longitude,
    )

    if image is None:
        return LocalizationResult(**base, success=False, failure_reason="could not read image")

    test_kp, test_desc = compute_orb_features(image, n_features=n_features)
    if test_desc is None or len(test_kp) == 0:
        return LocalizationResult(**base, success=False, failure_reason="no ORB features in test frame")

    # Ask the reference source only for the neighborhood the drone could be in
    # (radius of interest) -- this is both the anti-aliasing gate and the hook
    # for lazily loading map tiles instead of holding the whole map.
    ts = test_frame.timestamp_sec
    gated = False
    if motion_state is not None:
        pred_lat, pred_lon = motion_state.predict(ts)
        radius = motion_state.gating_radius_m(ts)
        candidate_entries = source.query(pred_lat, pred_lon, radius)
        gated = True
    else:
        candidate_entries = source.all()

    # Temporal exclusion: refuse to localize against reference views recorded
    # within `exclusion_sec` of this frame. With an interleaved hold-out the
    # neighbouring map frames are only ~1 s away, which would make the match
    # trivial and the reported accuracy meaningless.
    if exclusion_sec > 0:
        candidate_entries = [e for e in candidate_entries
                             if abs(e.timestamp_sec - ts) >= exclusion_sec]

    def accepted_from(entries, min_in, min_r):
        cands = match_candidates(test_desc, entries, ratio_thresh, top_k)
        verified = _verify_candidates(test_kp, cands, image.shape[1], image.shape[0])
        return [d for d in verified if d["n_in"] >= min_in and d["ratio"] >= min_r and len(d["good"]) >= min_good_matches]

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

    if not accepted:
        return LocalizationResult(
            **base, success=False, failure_reason="insufficient geometric match confidence",
        )

    best = accepted[0]  # most inliers; used for reporting + as the fusion base
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
        num_good_matches=len(best["good"]),
        num_inliers=best["n_in"],
        inlier_ratio=best["ratio"],
        num_refs_fused=len(accepted) if fuse_references else 1,
        gated=gated,
    )


def localize_all(
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
) -> list[LocalizationResult]:
    """Run the real-time navigation loop over the test frames in order.

    `reference` is a ReferenceSource (or a plain list of entries, which is
    wrapped). The loop is: **predict** the position from the motion model,
    **gate** the map to the radius of interest, **match + verify** visually, and
    then **correct** -- or, when no visual fix is trustworthy, **dead-reckon**
    on the drone's motion and still report a (lower-confidence) position rather
    than rejecting the frame. This mirrors a real GPS-denied navigator that is
    given a starting position and coasts on its motion between visual fixes.

    With `use_motion=False` there is no start position and no dead reckoning:
    unmatched frames are genuine failures (the old pure-matching behavior).
    """
    assert test_frames, "Sanity check failed: test frame set is empty, nothing to localize."
    source = as_reference_source(reference)
    assert source.all(), "Sanity check failed: reference source is empty, cannot localize."

    motion_state = (seed_from_reference_track(source.all(), start_time_sec=test_frames[0].timestamp_sec)
                    if use_motion else None)

    results = []
    for tf in test_frames:
        result = localize_frame(
            tf, source, motion_state=motion_state, n_features=n_features,
            ratio_thresh=ratio_thresh, min_good_matches=min_good_matches,
            min_inliers=min_inliers, min_inlier_ratio=min_inlier_ratio, top_k=top_k,
            fuse_references=fuse_references, exclusion_sec=exclusion_sec,
        )

        if motion_state is None:
            # No motion model -> no dead reckoning; report the fix or a failure.
            if not result.success:
                print(
                    f"[localize] FAILED '{os.path.basename(tf.frame_path)}': {result.failure_reason}.",
                    file=sys.stderr,
                )
            results.append(result)
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
            print(
                f"[localize] DEAD-RECKON '{os.path.basename(tf.frame_path)}' "
                f"(coast #{result.coast_steps}): no trusted match ({result.failure_reason}).",
                file=sys.stderr,
            )
        results.append(result)
    return results


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
