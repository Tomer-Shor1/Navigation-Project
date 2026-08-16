"""Steps 3-4: Split extracted frames into reference/test sets, and build an
in-memory ORB feature index over the reference set.

The "reference map" here is intentionally just a list of individually
georeferenced frames with their ORB features -- not a stitched orthomosaic
and not a real database. That matches the MVP scope.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .extract_frames import FrameRecord

DEFAULT_ORB_FEATURES = 2000

# Fallback altitude (meters above takeoff) used only if a frame's telemetry is
# missing an altitude value. This MVP's target flights are flown at ~119m.
FALLBACK_ALTITUDE_M = 119.0


@dataclass
class ReferenceEntry:
    """A single georeferenced, feature-indexed reference view.

    This is deliberately source-agnostic: today every entry is a frame from an
    earlier part of the same flight, but the same structure describes a
    georeferenced map/orthophoto tile (a GIS / Google-Earth patch with a known
    center lat/lon, altitude-equivalent scale, and heading). That is the seam
    the final-project extension plugs into -- swap how this list is populated,
    and the matcher/localizer downstream are unchanged. `source` records where
    the entry came from so mixed indices stay debuggable.
    """

    frame_path: str
    timestamp_sec: float
    latitude: float
    longitude: float
    altitude: float
    gimbal_pitch: Optional[float]
    gimbal_yaw: Optional[float]
    estimated_yaw_deg: Optional[float]
    image_width: int
    image_height: int
    # Best available absolute camera heading (deg, 0=N) for this view: measured
    # gimbal yaw if the telemetry has it, else the GPS travel-bearing estimate.
    # Resolved once here so the localizer has a single heading source to fuse
    # with the visual (homography) rotation.
    heading_deg: Optional[float] = None
    # Metres per pixel measured from the reference track's own GPS during
    # preprocessing (see src/calibrate.py). None -> the localizer falls back to
    # the modelled FOV/altitude/gimbal-pitch ground sample distance.
    gsd_m_per_px: Optional[float] = None
    source: str = "flight_frame"
    keypoints: list = field(default_factory=list)   # list[cv2.KeyPoint]
    descriptors: Optional[np.ndarray] = None


def split_reference_test(
    frames: list[FrameRecord], ref_fraction: float = 0.8
) -> tuple[list[FrameRecord], list[FrameRecord]]:
    """Split chronologically-ordered frames into a reference set (first
    `ref_fraction`) and a test set (the remainder), preserving order.
    """
    assert frames, "Sanity check failed: no frames to split into reference/test sets."
    n_ref = max(1, int(len(frames) * ref_fraction))
    n_ref = min(n_ref, len(frames) - 1) if len(frames) > 1 else len(frames)
    reference_frames = frames[:n_ref]
    test_frames = frames[n_ref:]
    return reference_frames, test_frames


def split_interleaved(
    frames: list[FrameRecord], every: int = 5
) -> tuple[list[FrameRecord], list[FrameRecord]]:
    """Hold out every `every`-th frame as a test frame; the rest are the map.

    Why this split exists alongside `split_reference_test`
    -----------------------------------------------------
    The tail split (first 80% = map, last 20% = test) does not measure
    "can we localize inside a known map". On these flights the last 20% is a
    return leg flown back over the mapped area on the *opposite* heading, and
    ORB simply cannot match a place seen from a reversed oblique viewpoint --
    measured here: matching a test frame against the reference frame nearest to
    its true position (a GPS oracle) still yields only 5-18 good matches and
    0-6 RANSAC inliers, versus 500+ on a same-heading overlap. The tail split
    therefore reports the failure of appearance matching across a 180-degree
    revisit, not the accuracy of the navigator.

    The interleaved hold-out keeps the test frames inside the mapped corridor,
    which is the assignment's actual scenario ("preprocess this flight data,
    then localize a new pass"). Pair it with `exclusion_sec` in the localizer so
    a test frame cannot be matched against its own temporal neighbours.
    """
    assert frames, "Sanity check failed: no frames to split."
    assert every >= 2, "Sanity check failed: --holdout-every must be >= 2."
    reference_frames = [f for i, f in enumerate(frames) if i % every != 0]
    # Drop the very first test frame: nothing precedes it to seed the motion model.
    test_frames = [f for i, f in enumerate(frames) if i % every == 0][1:]
    return reference_frames, test_frames


def compute_orb_features(gray_image: np.ndarray, n_features: int = DEFAULT_ORB_FEATURES):
    """Compute ORB keypoints + descriptors for a single grayscale image."""
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    return keypoints, descriptors


def build_reference_index(
    reference_frames: list[FrameRecord],
    n_features: int = DEFAULT_ORB_FEATURES,
    assumed_gimbal_pitch_deg: Optional[float] = None,
) -> list[ReferenceEntry]:
    """Load each reference frame's image, compute ORB features, and package
    everything (features + known GPS/altitude/gimbal angle) into a simple
    in-memory list -- the "reference map" for later matching.

    `assumed_gimbal_pitch_deg` fills in the gimbal pitch for frames whose SRT has
    no gimbal field (true for every DJI flight tested here), so a per-flight
    camera angle (e.g. -60 for the Mini 3 Pro) drives the slant-range GSD. Left
    as None, downstream falls back to its own module default.
    """
    assert reference_frames, "Sanity check failed: reference frame set is empty."

    index: list[ReferenceEntry] = []
    for fr in reference_frames:
        image = cv2.imread(fr.frame_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[build_reference] WARNING: could not read '{fr.frame_path}', skipping.", file=sys.stderr)
            continue

        keypoints, descriptors = compute_orb_features(image, n_features=n_features)
        if descriptors is None or len(keypoints) == 0:
            print(
                f"[build_reference] WARNING: no ORB features found in '{fr.frame_path}' "
                f"(likely textureless), skipping as a reference frame.",
                file=sys.stderr,
            )
            continue

        altitude = fr.altitude
        if altitude is None:
            print(
                f"[build_reference] WARNING: '{fr.frame_path}' has no altitude telemetry, "
                f"using fallback of {FALLBACK_ALTITUDE_M}m.",
                file=sys.stderr,
            )
            altitude = FALLBACK_ALTITUDE_M

        heading_deg = fr.gimbal_yaw if fr.gimbal_yaw is not None else fr.estimated_yaw_deg
        gimbal_pitch = fr.gimbal_pitch if fr.gimbal_pitch is not None else assumed_gimbal_pitch_deg

        index.append(
            ReferenceEntry(
                frame_path=fr.frame_path,
                timestamp_sec=fr.timestamp_sec,
                latitude=fr.latitude,
                longitude=fr.longitude,
                altitude=altitude,
                gimbal_pitch=gimbal_pitch,
                gimbal_yaw=fr.gimbal_yaw,
                estimated_yaw_deg=fr.estimated_yaw_deg,
                heading_deg=heading_deg,
                image_width=image.shape[1],
                image_height=image.shape[0],
                keypoints=keypoints,
                descriptors=descriptors,
            )
        )

    assert index, "Sanity check failed: reference index is empty after ORB extraction (no usable frames)."
    return index


def _print_sample(index: list[ReferenceEntry]) -> None:
    n_kps = [len(e.keypoints) for e in index]
    print(f"Built reference index with {len(index)} frames.")
    print(f"  keypoints per frame: min={min(n_kps)} max={max(n_kps)} mean={sum(n_kps)/len(n_kps):.1f}")
    for e in index[:3]:
        print(
            f"  {os.path.basename(e.frame_path)} t={e.timestamp_sec:.2f}s "
            f"lat={e.latitude:.6f} lon={e.longitude:.6f} alt={e.altitude} "
            f"n_kp={len(e.keypoints)}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python -m src.build_reference <video_path> <srt_path> <frames_dir> [rate_hz] [max_seconds]")
        sys.exit(1)

    from .parse_srt import parse_srt
    from .extract_frames import extract_frames, attach_telemetry

    video_path, srt_path, frames_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    rate_hz = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    max_seconds = float(sys.argv[5]) if len(sys.argv) > 5 else None

    telemetry = parse_srt(srt_path)
    assert telemetry, "Sanity check failed: parsed telemetry table is empty."
    raw_frames = extract_frames(video_path, frames_dir, rate_hz=rate_hz, max_seconds=max_seconds)
    tagged_frames = attach_telemetry(raw_frames, telemetry)

    ref_frames, test_frames = split_reference_test(tagged_frames, ref_fraction=0.8)
    print(f"Split {len(tagged_frames)} frames -> {len(ref_frames)} reference / {len(test_frames)} test")

    ref_index = build_reference_index(ref_frames)
    _print_sample(ref_index)
