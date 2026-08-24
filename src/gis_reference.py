"""GIS / satellite basemap as the reference map (the final-project extension).

Ex1 uses an earlier part of the *same flight* as the map. The final project
replaces that with a **georeferenced raster** -- a Google Earth / satellite
basemap. Everything downstream is unchanged, because both produce the same
`ReferenceEntry` objects behind the same `ReferenceSource` interface.

Why a raster map is structurally better than a stack of flight frames
---------------------------------------------------------------------
The two error sources that had to be *measured* out of the flight-frame map
(see `src/calibrate.py`) simply do not exist here:

* **Scale is exact.** A Web-Mercator tile's ground sample distance is a closed
  form: ``156543.03392 * cos(lat) / 2**zoom`` metres per pixel. No FOV guess, no
  altitude datum, no gimbal angle.
* **Heading is exact.** The raster is north-up, so every reference view has
  heading 0. Nothing to estimate, nothing to drift.

And the map no longer has to have been flown before, which is the entire point:
the drone can be localized over ground it has never itself visited.

Two ways to supply the raster
-----------------------------
1. ``tools/fetch_basemap.py`` downloads XYZ satellite tiles for a bounding box
   and writes ``<name>.jpg`` plus a ``<name>.json`` sidecar. Run it anywhere
   with internet access.
2. Export an image from Google Earth Pro (File > Save Image) and hand-write the
   same sidecar with the image's bounds. Same loader either way.

The sidecar is deliberately trivial -- four numbers -- so this stays dependency
free (no GDAL/rasterio):

    {"min_lat":.., "min_lon":.., "max_lat":.., "max_lon":.., "source":"..."}
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import utm

from .build_reference import ReferenceEntry, compute_orb_features

# Web-Mercator ground resolution at zoom 0, in metres per pixel at the equator
# (earth circumference / 256-pixel tile).
EQUATOR_M_PER_PX_Z0 = 156543.03392804097


def gsd_at(latitude_deg: float, zoom: int) -> float:
    """Metres per pixel of a Web-Mercator basemap at this latitude and zoom."""
    return EQUATOR_M_PER_PX_Z0 * math.cos(math.radians(latitude_deg)) / (2 ** zoom)


def zoom_for_gsd(latitude_deg: float, target_gsd_m: float) -> int:
    """Smallest zoom whose GSD is at least as fine as `target_gsd_m`."""
    z = math.log2(EQUATOR_M_PER_PX_Z0 * math.cos(math.radians(latitude_deg)) / target_gsd_m)
    return max(0, min(22, math.ceil(z)))


def lonlat_to_pixel(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Global Web-Mercator pixel coordinates (256 px tiles) at `zoom`."""
    n = 256 * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * n
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def pixel_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    """Inverse of `lonlat_to_pixel`."""
    n = 256 * (2 ** zoom)
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


@dataclass
class Basemap:
    """A north-up georeferenced raster plus the bounds it covers."""

    image: np.ndarray          # grayscale
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    source: str = "unknown"
    # Exact georeferencing for rasters that were *built* on a metric grid (the
    # orthomosaic). A satellite tile mosaic leaves this None and uses the
    # lat/lon bounds, which is what Web-Mercator tiles are natively indexed by.
    # (easting_of_col0, northing_of_row0, metres_per_px, zone_number, zone_letter)
    utm_anchor: Optional[tuple] = None

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def gsd_m_per_px(self) -> float:
        """Metres per pixel of the raster.

        Exact when the raster was built on a metric grid; otherwise derived from
        the lat/lon bounds, so a Google Earth export with no tile structure works
        exactly like a downloaded tile mosaic.
        """
        if self.utm_anchor is not None:
            return self.utm_anchor[2]
        mid_lat = (self.min_lat + self.max_lat) / 2
        span_m = (self.max_lon - self.min_lon) * math.radians(1.0) * 6378137.0 * math.cos(math.radians(mid_lat))
        return span_m / self.width

    def pixel_to_latlon(self, col: float, row: float) -> tuple[float, float]:
        """Raster pixel -> lat/lon.

        Exact via UTM when the raster was built on a metric grid. Interpolating
        lat/lon linearly instead is *not* good enough for one that was: UTM grid
        north differs from true north by the convergence angle (~1.2 deg here),
        so the metric canvas does not map to a lat/lon rectangle, and reading it
        back linearly stretched this project's orthomosaic by 4% -- several
        metres of position error, on top of a rotation.
        """
        if self.utm_anchor is not None:
            east0, north0, gsd, zone_number, zone_letter = self.utm_anchor
            return utm.to_latlon(east0 + col * gsd, north0 - row * gsd, zone_number, zone_letter)
        lon = self.min_lon + (col / self.width) * (self.max_lon - self.min_lon)
        lat = self.max_lat - (row / self.height) * (self.max_lat - self.min_lat)
        return lat, lon


def load_basemap(image_path: str, bounds_path: Optional[str] = None) -> Basemap:
    """Load a basemap raster and its bounds sidecar."""
    if bounds_path is None:
        bounds_path = os.path.splitext(image_path)[0] + ".json"
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Basemap image not found at '{image_path}'. Run tools/fetch_basemap.py "
            f"(needs internet) or export an image from Google Earth Pro -- see "
            f"src/gis_reference.py for the sidecar format."
        )
    if not os.path.isfile(bounds_path):
        raise FileNotFoundError(
            f"Basemap bounds sidecar not found at '{bounds_path}'. It must contain "
            f'{{"min_lat":..,"min_lon":..,"max_lat":..,"max_lon":..}}.'
        )
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not decode basemap image '{image_path}'.")
    with open(bounds_path) as f:
        b = json.load(f)
    for key in ("min_lat", "min_lon", "max_lat", "max_lon"):
        if key not in b:
            raise ValueError(f"Basemap sidecar '{bounds_path}' is missing '{key}'.")
    return Basemap(image=image, min_lat=b["min_lat"], min_lon=b["min_lon"],
                   max_lat=b["max_lat"], max_lon=b["max_lon"],
                   source=b.get("source", "unknown"))


def normalize_for_matching(image: np.ndarray) -> np.ndarray:
    """Local contrast equalisation, applied identically to map and drone frame.

    Whatever normalisation the map gets, the frame must get too. Equalising only
    one side changes ORB's intensity comparisons on that side alone, which
    weakens true matches and lets confident false ones win -- measured here on
    flight_0024 before this was made symmetric.
    """
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def build_gis_index(
    basemap: Basemap,
    patch_px: int = 512,
    overlap: float = 0.5,
    n_features: int = 2000,
    clahe: bool = True,
) -> list[ReferenceEntry]:
    """Cut the basemap into overlapping north-up patches and ORB-index them.

    Each patch becomes a `ReferenceEntry` with an **exactly** known ground
    sample distance and heading 0, so the localizer's pixel-offset-to-lat/lon
    step needs no calibration at all.

    `overlap` (0..1) is how much neighbouring patches share. Overlap matters:
    a drone frame landing on a patch boundary would otherwise have only half its
    content available in any single patch.
    """
    assert 0.0 <= overlap < 1.0, "Sanity check failed: overlap must be in [0, 1)."
    image = basemap.image
    if clahe:
        # Satellite imagery and drone video are exposed very differently; local
        # contrast equalisation is the cheapest way to make ORB's intensity
        # comparisons mean roughly the same thing on both. The drone frame gets
        # the identical treatment in the localizer.
        image = normalize_for_matching(image)

    stride = max(1, int(patch_px * (1.0 - overlap)))
    gsd = basemap.gsd_m_per_px
    index: list[ReferenceEntry] = []

    for row in range(0, max(1, basemap.height - patch_px + 1), stride):
        for col in range(0, max(1, basemap.width - patch_px + 1), stride):
            patch = image[row:row + patch_px, col:col + patch_px]
            if patch.shape[0] < 32 or patch.shape[1] < 32:
                continue
            keypoints, descriptors = compute_orb_features(patch, n_features=n_features)
            if descriptors is None or len(keypoints) < 8:
                continue
            centre_lat, centre_lon = basemap.pixel_to_latlon(
                col + patch.shape[1] / 2, row + patch.shape[0] / 2
            )
            index.append(ReferenceEntry(
                frame_path=f"gis://{basemap.source}/patch_r{row}_c{col}",
                timestamp_sec=0.0,           # a map tile has no time
                latitude=centre_lat,
                longitude=centre_lon,
                altitude=None,               # not needed: the GSD is already exact
                gimbal_pitch=-90.0,          # orthophoto == nadir
                gimbal_yaw=0.0,
                estimated_yaw_deg=0.0,
                heading_deg=0.0,             # north-up, exactly
                gsd_m_per_px=gsd,            # exact, not modelled
                image_width=patch.shape[1],
                image_height=patch.shape[0],
                source="gis_tile",
                keypoints=keypoints,
                descriptors=descriptors,
            ))

    if not index:
        raise RuntimeError(
            "Basemap produced no usable patches -- is the raster larger than "
            f"patch_px ({patch_px})? Raster is {basemap.width}x{basemap.height}."
        )
    print(f"[gis] indexed {len(index)} map patches of {patch_px}px "
          f"({gsd:.3f} m/px, {patch_px * gsd:.0f} m across, {int(overlap * 100)}% overlap) "
          f"from '{basemap.source}'.", file=sys.stderr)
    return index


def rescale_frame_to_map(image: np.ndarray, frame_gsd_m_per_px: float,
                         map_gsd_m_per_px: float) -> tuple[np.ndarray, float]:
    """Resample a drone frame so one pixel covers the same ground as the map's.

    This matters more than it looks. ORB is rotation invariant but only weakly
    scale invariant, and a 50 m drone frame is typically 3-5x finer than a
    satellite basemap; matching them at native resolution mostly fails. Returns
    the resampled image and the factor applied (map coords = factor * frame).
    """
    if not frame_gsd_m_per_px or not map_gsd_m_per_px:
        return image, 1.0
    factor = frame_gsd_m_per_px / map_gsd_m_per_px
    if not (0.05 < factor < 20):
        return image, 1.0
    new_w = max(32, int(round(image.shape[1] * factor)))
    new_h = max(32, int(round(image.shape[0] * factor)))
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp), factor
