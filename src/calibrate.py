"""Preprocessing self-calibration of the image->ground transform.

Why this module exists
----------------------
The original pipeline converted a pixel offset into metres with a *modelled*
ground sample distance: camera FOV, the reported ``rel_alt`` and an assumed
gimbal pitch. On the flights in ``data/raw`` that model is wrong by a large,
systematic factor. Two reasons:

* DJI's ``rel_alt`` is height above the **take-off point**, not above the ground
  actually being imaged. Fly off a ridge and the true AGL is far larger than the
  logged number.
* The SRT files here carry **no gimbal pitch or yaw field at all**, so the
  pipeline fell back to a hard-coded ``-60`` degrees and to "camera heading ==
  GPS travel bearing", which is wrong whenever the drone turns.

Measured on ``data/raw/flight.mp4``: the modelled GSD is 0.0455 m/px while the
true scale is ~0.071 m/px -- a 36% underestimate of every distance. That was
independently confirmed from the imagery itself (the parking-stall pitch in
``frame_00050.jpg`` autocorrelates at 34 px; at a 2.4 m stall width that is
0.0706 m/px).

What it does instead
--------------------
The preprocessing pass *has* GNSS -- that is the whole premise of the
assignment. So rather than trusting the camera model, we measure the transform
directly from the reference track:

For each pair of consecutive reference frames we estimate the homography
between them, map one frame centre into the other, and compare that pixel
displacement with the GPS displacement between the two frames. Each pair yields

* a **metric scale** (metres per pixel), normalised by the reported altitude so
  it still adapts if the drone changes height, and
* a **camera heading**: the bearing of the GPS displacement minus the direction
  of the image displacement.

Both are aggregated robustly (median / trimmed median, circular mean for the
heading). No test-frame GNSS is involved, so the navigation-stage evaluation
stays honest.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import utm

DEFAULT_RATIO_THRESH = 0.75
# Pairs closer than this in metres are dominated by GPS noise, not motion.
DEFAULT_MIN_DISPLACEMENT_M = 4.0
# Only calibrate off pairs that are genuinely adjacent in time.
DEFAULT_MAX_GAP_SEC = 1.5
DEFAULT_MIN_INLIERS = 20


@dataclass
class TrackCalibration:
    """Image->ground transform measured from the georeferenced reference track."""

    scale_per_alt_m_per_px: float
    """Metres per pixel, per metre of reported altitude. Multiply by a frame's
    altitude to get its ground sample distance."""

    headings_deg: dict = field(default_factory=dict)
    """frame_path -> camera heading in degrees (0=N, clockwise)."""

    n_observations: int = 0
    scale_scatter: float = float("nan")
    """Median relative deviation of the per-pair scale estimates. A useful
    health check: > ~0.3 means the fit is not trustworthy."""

    def gsd_for(self, altitude_m: float) -> float:
        return self.scale_per_alt_m_per_px * altitude_m


def _center_shift_px(entry_a, entry_b, ratio_thresh: float, min_inliers: int):
    """Map entry_b's image centre into entry_a's image; return (dx, dy) in px."""
    if entry_a.descriptors is None or entry_b.descriptors is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(entry_b.descriptors, entry_a.descriptors, k=2)
    good = [m for m, n in knn if m.distance < ratio_thresh * n.distance]
    if len(good) < 15:
        return None
    src = np.float32([entry_b.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([entry_a.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None or int(mask.sum()) < min_inliers:
        return None
    center = np.array([[entry_b.image_width / 2, entry_b.image_height / 2]], np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(center, H)[0, 0]
    return float(mapped[0] - entry_a.image_width / 2), float(mapped[1] - entry_a.image_height / 2)


def _circular_mean_deg(values: list[float]) -> float:
    rad = np.radians(values)
    return math.degrees(math.atan2(float(np.mean(np.sin(rad))), float(np.mean(np.cos(rad))))) % 360


def calibrate_from_reference(
    reference_index: list,
    ratio_thresh: float = DEFAULT_RATIO_THRESH,
    min_displacement_m: float = DEFAULT_MIN_DISPLACEMENT_M,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    smoothing_halfwidth: int = 1,
) -> Optional[TrackCalibration]:
    """Measure scale + per-frame camera heading from the reference track's own GPS.

    Returns None when there are too few usable pairs (e.g. a pure hover), in
    which case callers should fall back to the modelled GSD.
    """
    observations = []  # (index, dx_px, dy_px, d_east_m, d_north_m, altitude_m)
    for i in range(1, len(reference_index)):
        a, b = reference_index[i - 1], reference_index[i]
        if b.timestamp_sec - a.timestamp_sec > max_gap_sec:
            continue
        if a.altitude is None or b.altitude is None:
            continue
        shift = _center_shift_px(a, b, ratio_thresh, min_inliers)
        if shift is None:
            continue
        e0, n0, zone_n, zone_l = utm.from_latlon(a.latitude, a.longitude)
        e1, n1, _, _ = utm.from_latlon(b.latitude, b.longitude,
                                       force_zone_number=zone_n, force_zone_letter=zone_l)
        d_east, d_north = e1 - e0, n1 - n0
        if math.hypot(d_east, d_north) < min_displacement_m:
            continue
        if math.hypot(*shift) < 1.0:
            continue
        observations.append((i, shift[0], shift[1], d_east, d_north, b.altitude))

    if len(observations) < 8:
        print(f"[calibrate] only {len(observations)} usable frame pairs -- "
              f"falling back to the modelled camera geometry.", file=sys.stderr)
        return None

    # --- metric scale: robust (trimmed) median of per-pair metres-per-pixel ---
    ratios = np.array([
        math.hypot(o[3], o[4]) / (math.hypot(o[1], o[2]) * o[5]) for o in observations
    ])
    scale = float(np.median(ratios))
    for _ in range(4):
        mad = float(np.median(np.abs(ratios - scale)))
        if mad <= 0:
            break
        keep = np.abs(ratios - scale) <= 1.5 * mad
        if keep.sum() < 5:
            break
        scale = float(np.median(ratios[keep]))
    scatter = float(np.median(np.abs(ratios - scale) / scale))

    # --- per-frame camera heading ---
    # bearing of the GPS step, minus the direction of the image step.
    # Image axes: +x right, +y down, so "up the image" is -y.
    per_pair = {
        o[0]: (math.degrees(math.atan2(o[3], o[4])) - math.degrees(math.atan2(o[1], -o[2]))) % 360
        for o in observations
    }
    headings = [None] * len(reference_index)
    for i in range(len(reference_index)):
        window = [per_pair[j] for j in range(i - smoothing_halfwidth, i + smoothing_halfwidth + 1)
                  if j in per_pair]
        if window:
            headings[i] = _circular_mean_deg(window)
    # hold the nearest measured heading across gaps, forwards then backwards
    for order in (range(len(headings)), range(len(headings) - 1, -1, -1)):
        last = None
        for i in order:
            if headings[i] is None:
                headings[i] = last
            else:
                last = headings[i]

    by_path = {
        reference_index[i].frame_path: headings[i]
        for i in range(len(reference_index)) if headings[i] is not None
    }
    return TrackCalibration(
        scale_per_alt_m_per_px=scale,
        headings_deg=by_path,
        n_observations=len(observations),
        scale_scatter=scatter,
    )


def apply_calibration(reference_index: list, calibration: Optional[TrackCalibration]) -> None:
    """Write the measured scale + heading onto each reference entry, in place.

    Entries keep their modelled fallback: ``gsd_m_per_px`` stays None when there
    is no calibration, and the localizer then uses the FOV/altitude model.
    """
    if calibration is None:
        return
    for entry in reference_index:
        if entry.altitude is not None:
            entry.gsd_m_per_px = calibration.gsd_for(entry.altitude)
        measured_heading = calibration.headings_deg.get(entry.frame_path)
        if measured_heading is not None:
            entry.heading_deg = measured_heading
