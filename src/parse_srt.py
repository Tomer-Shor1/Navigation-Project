"""Step 1: Parse DJI-style .srt telemetry files into a clean list of records.

DJI subtitle blocks are not a single rigid format -- it varies across firmware
versions (Mavic/Phantom vs. Mini 3 Pro, older vs. newer app builds). This parser
uses flexible, tolerant regexes rather than assuming one exact layout, and it
logs + skips any block it cannot extract GPS from instead of crashing.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class TelemetryRecord:
    index: int
    start_sec: float
    end_sec: float
    latitude: float
    longitude: float
    altitude: Optional[float]      # meters, prefers rel_alt (above takeoff) over abs_alt
    rel_alt: Optional[float]
    abs_alt: Optional[float]
    gimbal_pitch: Optional[float]
    gimbal_yaw: Optional[float]
    datetime_str: Optional[str]


_TIMECODE_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)

_INDEX_RE = re.compile(r"^\s*(\d+)\s*$")

# GPS as a single tuple: GPS(lat,lon,alt) or "GPS (lat, lon, alt)"
_GPS_TUPLE_RE = re.compile(
    r"GPS\s*\(\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*(?:,\s*([-+]?\d+\.?\d*)\s*)?\)",
    re.IGNORECASE,
)

# Separate latitude / longitude fields, e.g. "[latitude: 32.123456]" or "latitude : 32.1"
_LAT_RE = re.compile(r"latitude\s*[:=]\s*([-+]?\d+\.\d+)", re.IGNORECASE)
_LON_RE = re.compile(r"longitude\s*[:=]\s*([-+]?\d+\.\d+)", re.IGNORECASE)

_REL_ALT_RE = re.compile(r"rel_alt\s*[:=]\s*([-+]?\d+\.?\d*)", re.IGNORECASE)
_ABS_ALT_RE = re.compile(r"abs_alt\s*[:=]\s*([-+]?\d+\.?\d*)", re.IGNORECASE)

# Gimbal pitch: gb_pitch, gimbal_pitch, or "gimbal: pitch"
_GIMBAL_PITCH_RE = re.compile(
    r"(?:gb_pitch|gimbal_pitch|gimbal\.pitch)\s*[:=]\s*([-+]?\d+\.?\d*)",
    re.IGNORECASE,
)
_GIMBAL_YAW_RE = re.compile(
    r"(?:gb_yaw|gimbal_yaw|gimbal\.yaw)\s*[:=]\s*([-+]?\d+\.?\d*)",
    re.IGNORECASE,
)

_DATETIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[.,]?\d*)")


def _timecode_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_block(block: str, block_num: int) -> Optional[TelemetryRecord]:
    """Parse a single SRT block. Returns None (and logs) if GPS can't be found."""
    lines = block.strip("﻿\n\r ").splitlines()
    if not lines:
        return None

    tc_match = _TIMECODE_RE.search(block)
    if not tc_match:
        print(f"[parse_srt] WARNING: block {block_num} has no timecode, skipping.", file=sys.stderr)
        return None

    start_sec = _timecode_to_seconds(*tc_match.groups()[0:4])
    end_sec = _timecode_to_seconds(*tc_match.groups()[4:8])

    index_match = _INDEX_RE.match(lines[0])
    index = int(index_match.group(1)) if index_match else block_num

    text = block

    lat = lon = alt_from_tuple = None
    gps_tuple = _GPS_TUPLE_RE.search(text)
    if gps_tuple:
        lat = float(gps_tuple.group(1))
        lon = float(gps_tuple.group(2))
        if gps_tuple.group(3):
            alt_from_tuple = float(gps_tuple.group(3))
    else:
        lat_match = _LAT_RE.search(text)
        lon_match = _LON_RE.search(text)
        if lat_match and lon_match:
            lat = float(lat_match.group(1))
            lon = float(lon_match.group(1))

    if lat is None or lon is None:
        print(
            f"[parse_srt] WARNING: block {block_num} (index {index}) has no "
            f"parsable GPS fields, skipping.",
            file=sys.stderr,
        )
        return None

    rel_alt_match = _REL_ALT_RE.search(text)
    abs_alt_match = _ABS_ALT_RE.search(text)
    rel_alt = float(rel_alt_match.group(1)) if rel_alt_match else None
    abs_alt = float(abs_alt_match.group(1)) if abs_alt_match else None

    altitude = rel_alt if rel_alt is not None else (abs_alt if abs_alt is not None else alt_from_tuple)

    pitch_match = _GIMBAL_PITCH_RE.search(text)
    yaw_match = _GIMBAL_YAW_RE.search(text)
    gimbal_pitch = float(pitch_match.group(1)) if pitch_match else None
    gimbal_yaw = float(yaw_match.group(1)) if yaw_match else None

    dt_match = _DATETIME_RE.search(text)
    datetime_str = dt_match.group(1) if dt_match else None

    return TelemetryRecord(
        index=index,
        start_sec=start_sec,
        end_sec=end_sec,
        latitude=lat,
        longitude=lon,
        altitude=altitude,
        rel_alt=rel_alt,
        abs_alt=abs_alt,
        gimbal_pitch=gimbal_pitch,
        gimbal_yaw=gimbal_yaw,
        datetime_str=datetime_str,
    )


def parse_srt(path: str) -> list[TelemetryRecord]:
    """Parse a DJI .srt telemetry file into a list of TelemetryRecord.

    Malformed or GPS-less blocks are logged to stderr and skipped rather than
    raising, since real-world DJI SRT files sometimes contain a header block
    or a stray block without full telemetry.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"[parse_srt] ERROR: telemetry file not found at '{path}'.", file=sys.stderr)
        return []

    if not content.strip():
        print(f"[parse_srt] ERROR: telemetry file '{path}' is empty.", file=sys.stderr)
        return []

    raw_blocks = re.split(r"\r?\n\r?\n+", content.strip())

    records: list[TelemetryRecord] = []
    for i, block in enumerate(raw_blocks, start=1):
        record = _parse_block(block, i)
        if record is not None:
            records.append(record)

    if not records:
        print(f"[parse_srt] ERROR: no valid telemetry records parsed from '{path}'.", file=sys.stderr)

    return records


def _print_sample(records: list[TelemetryRecord], n: int = 5) -> None:
    print(f"Parsed {len(records)} telemetry records.\n")
    print(f"{'idx':>4} {'t(s)':>8} {'lat':>11} {'lon':>11} {'alt':>7} {'gimbal_pitch':>12}")
    for r in records[:n]:
        print(
            f"{r.index:>4} {r.start_sec:>8.3f} {r.latitude:>11.6f} {r.longitude:>11.6f} "
            f"{str(r.altitude):>7} {str(r.gimbal_pitch):>12}"
        )
    if len(records) > n:
        print("...")
        r = records[-1]
        print(
            f"{r.index:>4} {r.start_sec:>8.3f} {r.latitude:>11.6f} {r.longitude:>11.6f} "
            f"{str(r.altitude):>7} {str(r.gimbal_pitch):>12}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/parse_srt.py <path_to_srt>")
        sys.exit(1)
    recs = parse_srt(sys.argv[1])
    assert len(recs) > 0, "Sanity check failed: parsed telemetry table is empty."
    _print_sample(recs)
