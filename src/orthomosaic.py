"""Stitch the georeferenced flight frames into one north-up raster map.

This is the bridge between Ex1 and the final project. Ex1's map is a *stack of
oblique frames*, each one usable only from roughly the heading it was shot at.
The final project's map is a *single north-up georeferenced raster*. This module
produces the second thing out of the first, in exactly the `Basemap` format that
`src/gis_reference.py` loads a satellite download into.

Two reasons it earns its place:

* **It makes the GIS pipeline testable without a network.** The satellite path
  and this path produce the same `Basemap`, feed the same `build_gis_index()`,
  and run through the same localizer, so the whole final-project code path can be
  exercised and measured offline. Swapping in real satellite imagery changes one
  argument.
* **It is a real method in its own right.** Rotating every frame to north-up and
  resampling to a common ground scale removes the map's dependence on the
  heading it happened to be flown at -- which is precisely the failure that
  wrecks the return leg in Ex1 (see RESULTS.md).

Each frame is placed by its own GPS position, so placement error does not
accumulate along the flight the way it would in a chained visual mosaic. The
geometry used is the similarity transform measured by `src/calibrate.py`:
metres-per-pixel and camera heading. That is an approximation -- it treats each
frame as if it were nadir -- and the seams show it. Nothing here does bundle
adjustment or terrain correction; a production orthomosaic (ODM, Pix4D) would.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Optional

import cv2
import numpy as np
import utm

from .gis_reference import Basemap

DEFAULT_TARGET_GSD_M = 0.15


def _frame_gsd(entry, fallback: Optional[float]) -> Optional[float]:
    if getattr(entry, "gsd_m_per_px", None):
        return entry.gsd_m_per_px
    return fallback


def build_ortho_basemap(
    reference_index: list,
    target_gsd_m: float = DEFAULT_TARGET_GSD_M,
    fallback_gsd_m: Optional[float] = None,
    margin_m: float = 60.0,
    clahe: bool = False,
    min_altitude_m: float = 15.0,
) -> Basemap:
    """Warp every reference frame to north-up and average them into one raster.

    `target_gsd_m` is the output ground sample distance. 0.15 m/px is chosen to
    sit near what a good satellite basemap delivers (Web-Mercator zoom 20 at this
    latitude is ~0.11 m/px), so the offline experiment and the satellite
    experiment stress the matcher at a comparable resolution.
    """
    # Frames from the take-off / landing roll are useless as map material: at a
    # few metres altitude they cover almost no ground, and at 0 m the scale
    # collapses to nothing and the warp degenerates.
    grounded = [e for e in reference_index if e.altitude is not None and e.altitude < min_altitude_m]
    usable = [e for e in reference_index
              if _frame_gsd(e, fallback_gsd_m) and e.heading_deg is not None
              and (e.altitude is None or e.altitude >= min_altitude_m)]
    if grounded:
        print(f"[ortho] skipping {len(grounded)} frame(s) below {min_altitude_m:.0f} m "
              f"(take-off/landing).", file=sys.stderr)
    if len(usable) < 2:
        raise RuntimeError(
            "Cannot build an orthomosaic: the reference frames have no measured "
            "scale/heading. Run without --no-calibrate (see src/calibrate.py)."
        )

    # Work in UTM metres, then convert the canvas corners back to lat/lon.
    e0, n0, zone_number, zone_letter = utm.from_latlon(usable[0].latitude, usable[0].longitude)
    positions = []
    for entry in usable:
        e, n, _, _ = utm.from_latlon(entry.latitude, entry.longitude,
                                     force_zone_number=zone_number, force_zone_letter=zone_letter)
        positions.append((e, n))

    # Each frame covers roughly half its diagonal in every direction.
    reach = max(
        _frame_gsd(e, fallback_gsd_m) * math.hypot(e.image_width, e.image_height) / 2
        for e in usable
    )
    pad = margin_m + reach
    east_min = min(p[0] for p in positions) - pad
    east_max = max(p[0] for p in positions) + pad
    north_min = min(p[1] for p in positions) - pad
    north_max = max(p[1] for p in positions) + pad

    width = int(math.ceil((east_max - east_min) / target_gsd_m))
    height = int(math.ceil((north_max - north_min) / target_gsd_m))
    if width * height > 80_000_000:
        raise RuntimeError(f"Orthomosaic canvas would be {width}x{height} px; raise --ortho-gsd.")

    # Compositing rule: for each output pixel keep the frame whose *centre* is
    # nearest, rather than averaging everything that covers it. Averaging was
    # tried first and produced an unusably blurred map -- the similarity model
    # below treats each frame as nadir, so overlapping frames disagree by a few
    # pixels and stacking ~100 of them smears away exactly the corners and edges
    # ORB needs. Nearest-centre also means each pixel comes from the least
    # oblique part of some frame, which is where the nadir approximation is best.
    mosaic = np.zeros((height, width), np.uint8)
    best_dist = np.full((height, width), np.inf, np.float32)

    for entry, (east, north) in zip(usable, positions):
        image = cv2.imread(entry.frame_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        gsd = _frame_gsd(entry, fallback_gsd_m)
        scale = gsd / target_gsd_m                      # frame px -> canvas px
        yaw = math.radians(entry.heading_deg)

        # Frame offset (dx right, dy down) -> world (east, north):
        #   east  =  gsd*dx*cos(yaw) - gsd*dy*sin(yaw)
        #   north = -gsd*dx*sin(yaw) - gsd*dy*cos(yaw)
        # then world -> canvas (col right, row down) divides by target_gsd and
        # flips north. Composing gives a plain similarity transform.
        a, b = scale * math.cos(yaw), -scale * math.sin(yaw)
        c, d = scale * math.sin(yaw), scale * math.cos(yaw)
        col0 = (east - east_min) / target_gsd_m
        row0 = (north_max - north) / target_gsd_m
        cx, cy = image.shape[1] / 2, image.shape[0] / 2

        # Work inside this frame's bounding box on the canvas, not the whole
        # canvas -- otherwise every frame costs a full-canvas allocation.
        corners = np.array([[-cx, -cy], [cx, -cy], [cx, cy], [-cx, cy]], np.float32)
        proj = np.stack([a * corners[:, 0] + b * corners[:, 1] + col0,
                         c * corners[:, 0] + d * corners[:, 1] + row0], axis=1)
        x0 = max(0, int(np.floor(proj[:, 0].min()))); x1 = min(width, int(np.ceil(proj[:, 0].max())) + 1)
        y0 = max(0, int(np.floor(proj[:, 1].min()))); y1 = min(height, int(np.ceil(proj[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        M = np.array([[a, b, col0 - (a * cx + b * cy) - x0],
                      [c, d, row0 - (c * cx + d * cy) - y0]], np.float32)
        bw, bh = x1 - x0, y1 - y0
        warped = cv2.warpAffine(image, M, (bw, bh), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(np.full(image.shape, 255, np.uint8), M, (bw, bh),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        cols = np.arange(x0, x1, dtype=np.float32) - col0
        rows = np.arange(y0, y1, dtype=np.float32) - row0
        dist = np.sqrt(cols[None, :] ** 2 + rows[:, None] ** 2)

        window_best = best_dist[y0:y1, x0:x1]
        take = (mask > 0) & (dist < window_best)
        window_best[take] = dist[take]
        mosaic[y0:y1, x0:x1][take] = warped[take]

    covered = np.isfinite(best_dist)

    if not covered.any():
        raise RuntimeError("Orthomosaic came out empty -- no frame warped onto the canvas.")
    if clahe:
        mosaic = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(mosaic)

    top_lat, left_lon = utm.to_latlon(east_min, north_max, zone_number, zone_letter)
    bottom_lat, right_lon = utm.to_latlon(east_max, north_min, zone_number, zone_letter)

    print(f"[ortho] {width}x{height} px at {target_gsd_m:.3f} m/px from {len(usable)} frames; "
          f"{100.0 * covered.mean():.0f}% of the canvas covered.", file=sys.stderr)

    return Basemap(image=mosaic, min_lat=bottom_lat, min_lon=left_lon,
                   max_lat=top_lat, max_lon=right_lon, source="flight_orthomosaic",
                   utm_anchor=(east_min, north_max, target_gsd_m, zone_number, zone_letter))


def save_basemap(basemap: Basemap, image_path: str) -> None:
    """Write a Basemap as .jpg + .json, the format `load_basemap()` reads."""
    os.makedirs(os.path.dirname(image_path) or ".", exist_ok=True)
    cv2.imwrite(image_path, basemap.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(os.path.splitext(image_path)[0] + ".json", "w") as f:
        json.dump({"min_lat": basemap.min_lat, "min_lon": basemap.min_lon,
                   "max_lat": basemap.max_lat, "max_lon": basemap.max_lon,
                   "source": basemap.source,
                   "gsd_m_per_px": basemap.gsd_m_per_px}, f, indent=2)
