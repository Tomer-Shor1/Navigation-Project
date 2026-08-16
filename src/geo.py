"""Shared geodesy helpers.

Coordinate math used across the pipeline (distances, bearings, and applying a
local East/North offset in meters to a lat/lon point via UTM) lives here so the
same formulas back both the same-flight MVP and the future GIS/Google-Earth
reference source. Keeping it in one place also avoids the copy of these
formulas that used to live in both `evaluate.py` and `extract_frames.py`.
"""

from __future__ import annotations

import math

import utm

EARTH_RADIUS_M = 6371000.0


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def approx_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Cheap equirectangular-approximation distance, for gating/threshold checks.

    Accurate to well under a meter over the sub-kilometer distances this project
    deals with, and avoids the trig of the full haversine on the hot path.
    """
    avg_lat_rad = math.radians((lat1 + lat2) / 2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    x = dlon * math.cos(avg_lat_rad)
    return EARTH_RADIUS_M * math.hypot(x, dlat)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing (0=N, 90=E, clockwise) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def offset_latlon(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    """Apply an East/North meters offset to a lat/lon point via its UTM zone."""
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    new_lat, new_lon = utm.to_latlon(easting + east_m, northing + north_m, zone_number, zone_letter)
    return new_lat, new_lon
