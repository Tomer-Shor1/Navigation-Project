"""Step 2: Extract frames from the flight video at a fixed rate via ffmpeg,
and attach the nearest telemetry entry (lat, lon, altitude, gimbal angle) to
each extracted frame by timestamp.
"""

from __future__ import annotations

import bisect
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from .geo import approx_distance_m, bearing_deg
from .parse_srt import TelemetryRecord, parse_srt


@dataclass
class FrameRecord:
    frame_path: str
    timestamp_sec: float
    latitude: float
    longitude: float
    altitude: Optional[float]
    gimbal_pitch: Optional[float]
    gimbal_yaw: Optional[float]
    estimated_yaw_deg: Optional[float] = None


def _check_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install it (e.g. `brew install ffmpeg` "
            "on macOS) before running frame extraction."
        )


def get_video_duration_sec(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe was not found on PATH (usually installed alongside ffmpeg).")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on '{video_path}': {result.stderr.strip()}")
    return float(result.stdout.strip())


def extract_frames(
    video_path: str,
    output_dir: str,
    rate_hz: float = 1.0,
    max_seconds: Optional[float] = None,
) -> list[tuple[str, float]]:
    """Extract frames from `video_path` at `rate_hz` frames/sec into `output_dir`.

    Returns a list of (frame_path, timestamp_sec) assuming constant frame rate
    output from ffmpeg's fps filter (n-th output frame ~= n / rate_hz seconds).
    If `max_seconds` is set, only that much of the video (from the start) is
    processed -- used for quick stub/sanity runs on placeholder clips.
    """
    _check_ffmpeg_available()

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video file not found at '{video_path}'. Place the flight video there "
            f"(see README) before running the pipeline."
        )

    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.startswith("frame_") and f.endswith(".jpg"):
            os.remove(os.path.join(output_dir, f))

    cmd = ["ffmpeg", "-y"]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds)]
    cmd += [
        "-i", video_path,
        "-vf", f"fps={rate_hz}",
        "-qscale:v", "2",
        os.path.join(output_dir, "frame_%05d.jpg"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed:\n{result.stderr[-2000:]}")

    frame_files = sorted(f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg"))
    if not frame_files:
        raise RuntimeError(
            f"ffmpeg produced no frames from '{video_path}'. Check that the file "
            f"is a valid, non-empty video."
        )

    frames = []
    for i, fname in enumerate(frame_files):
        timestamp_sec = i / rate_hz
        frames.append((os.path.join(output_dir, fname), timestamp_sec))
    return frames


def estimate_travel_bearing_deg(
    telemetry: list[TelemetryRecord],
    idx: int,
    window_sec: float = 0.5,
    min_displacement_m: float = 2.0,
) -> Optional[float]:
    """Estimate a frame's heading from its direction of travel: the compass
    bearing between telemetry samples ~window_sec before and after it.

    This assumes the camera roughly faces the direction of travel -- a
    reasonable default for forward flight, but wrong during hovering,
    orbiting, or independent gimbal panning. Returns None if there isn't
    enough surrounding telemetry, or displacement is too small for a
    reliable bearing (e.g. near-stationary hover), rather than guessing.
    """
    t0 = telemetry[idx].start_sec
    lo = idx
    while lo > 0 and t0 - telemetry[lo].start_sec < window_sec:
        lo -= 1
    hi = idx
    while hi < len(telemetry) - 1 and telemetry[hi].start_sec - t0 < window_sec:
        hi += 1
    if hi <= lo:
        return None

    p1, p2 = telemetry[lo], telemetry[hi]
    if approx_distance_m(p1.latitude, p1.longitude, p2.latitude, p2.longitude) < min_displacement_m:
        return None
    return bearing_deg(p1.latitude, p1.longitude, p2.latitude, p2.longitude)


def attach_telemetry(
    frames: list[tuple[str, float]],
    telemetry: list[TelemetryRecord],
) -> list[FrameRecord]:
    """Attach the nearest-by-timestamp telemetry record to each extracted frame,
    plus a travel-bearing-based yaw estimate (see estimate_travel_bearing_deg)
    for use when the SRT itself has no gimbal yaw reading.
    """
    assert telemetry, "Sanity check failed: telemetry list is empty, cannot attach to frames."

    starts = [r.start_sec for r in telemetry]
    results = []
    for frame_path, ts in frames:
        pos = bisect.bisect_left(starts, ts)
        candidates = [p for p in (pos - 1, pos) if 0 <= p < len(starts)]
        nearest = min(candidates, key=lambda p: abs(starts[p] - ts))
        rec = telemetry[nearest]
        results.append(
            FrameRecord(
                frame_path=frame_path,
                timestamp_sec=ts,
                latitude=rec.latitude,
                longitude=rec.longitude,
                altitude=rec.altitude,
                gimbal_pitch=rec.gimbal_pitch,
                gimbal_yaw=rec.gimbal_yaw,
                estimated_yaw_deg=estimate_travel_bearing_deg(telemetry, nearest),
            )
        )
    return results


def _print_sample(frames: list[FrameRecord], n: int = 5) -> None:
    print(f"Extracted and telemetry-tagged {len(frames)} frames.\n")
    print(f"{'frame':>16} {'t(s)':>8} {'lat':>11} {'lon':>11} {'alt':>7} {'pitch':>7}")
    for fr in frames[:n]:
        print(
            f"{os.path.basename(fr.frame_path):>16} {fr.timestamp_sec:>8.2f} "
            f"{fr.latitude:>11.6f} {fr.longitude:>11.6f} {str(fr.altitude):>7} {str(fr.gimbal_pitch):>7}"
        )
    if len(frames) > n:
        print("...")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python -m src.extract_frames <video_path> <srt_path> <output_dir> [rate_hz] [max_seconds]")
        sys.exit(1)

    video_path, srt_path, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    rate_hz = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    max_seconds = float(sys.argv[5]) if len(sys.argv) > 5 else None

    telemetry = parse_srt(srt_path)
    assert telemetry, "Sanity check failed: parsed telemetry table is empty."

    raw_frames = extract_frames(video_path, output_dir, rate_hz=rate_hz, max_seconds=max_seconds)
    tagged_frames = attach_telemetry(raw_frames, telemetry)
    _print_sample(tagged_frames)
