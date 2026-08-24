"""Download a satellite basemap for a lat/lon box and write it with its bounds.

Run this anywhere with internet access:

    python tools/fetch_basemap.py --flight data/raw/flight.srt

or with an explicit box:

    python tools/fetch_basemap.py --bbox 32.1015 35.2080 32.1060 35.2122

It writes two files that `src/gis_reference.load_basemap()` reads:

    data/basemap/<name>.jpg     the stitched, north-up raster
    data/basemap/<name>.json    {"min_lat":.., "min_lon":.., "max_lat":.., "max_lon":..}

Only the Python standard library and OpenCV are used, so there is nothing extra
to install beyond requirements.txt.

Imagery sources
---------------
`--source esri` (default) uses the Esri World Imagery service, which serves
tiles without an API key. `--source google` uses Google's satellite tiles and
`--source custom --url-template ...` takes any XYZ URL. Check the terms of
service of whichever provider you choose before using it for anything beyond
coursework; Esri's World Imagery is the one that is straightforwardly usable
here, and Google Earth imagery is generally intended to be accessed through
Google's own clients.

If a provider blocks you, the alternative needs no network at all: export a
top-down image from Google Earth Pro (File > Save Image), note the bounds it
covers, and hand-write the .json sidecar yourself.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gis_reference import gsd_at, lonlat_to_pixel, pixel_to_lonlat, zoom_for_gsd  # noqa: E402

SOURCES = {
    "esri": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "google": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
}
TILE_PX = 256
# Identifies the project, but browser-prefixed: some tile services reject a
# bare custom agent with a 404 that looks exactly like "no coverage here".
USER_AGENT = "Mozilla/5.0 (compatible; visual-nav-mvp/1.0; university coursework)"

# A tile with almost no contrast is a provider placeholder ("Map data not yet
# available"), not imagery. They return HTTP 200, so without this check a
# basemap of 357 blank tiles is reported as 357 tiles downloaded successfully --
# and the failure only shows up later as a navigator that cannot match anything.
PLACEHOLDER_STD = 12.0


def is_placeholder(tile: np.ndarray) -> bool:
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim == 3 else tile
    return float(gray.std()) < PLACEHOLDER_STD


def tls_context() -> ssl.SSLContext:
    """An SSL context that can actually verify a certificate chain.

    A python.org build on macOS ships without a CA bundle wired up -- it never
    runs the system keychain and never sees `/etc/ssl`, so *every* HTTPS tile
    request dies with CERTIFICATE_VERIFY_FAILED even though `curl` on the same
    machine is fine. That is what stopped this script from ever being run
    before. `certifi` carries the Mozilla root store as a data file, so ask for
    it first and fall back to whatever the interpreter found by itself.

    Verification is never disabled: a basemap is imagery this project's accuracy
    claims rest on, so it is worth knowing it came from who it says it did.
    """
    try:
        import certifi
    except ImportError:
        print("[fetch_basemap] certifi is not installed; falling back to the "
              "interpreter's default CA store. If tiles fail with "
              "CERTIFICATE_VERIFY_FAILED, run: pip install certifi", file=sys.stderr)
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


_TLS = None


def _shared_tls_context() -> ssl.SSLContext:
    """Built once: constructing a context per tile re-parses the whole CA bundle."""
    global _TLS
    if _TLS is None:
        _TLS = tls_context()
    return _TLS


def _import_ok(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def flight_bbox(srt_path: str, margin_m: float = 150.0):
    """Bounding box of a flight's GPS track, padded by `margin_m`."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.parse_srt import parse_srt
    records = parse_srt(srt_path)
    if not records:
        raise SystemExit(f"No telemetry parsed from {srt_path}")
    lats = [r.latitude for r in records]
    lons = [r.longitude for r in records]
    mid_lat = (min(lats) + max(lats)) / 2
    d_lat = margin_m / 111320.0
    d_lon = margin_m / (111320.0 * math.cos(math.radians(mid_lat)))
    return min(lats) - d_lat, min(lons) - d_lon, max(lats) + d_lat, max(lons) + d_lon


def fetch_tile(url: str, retries: int = 3, pause: float = 0.6) -> np.ndarray | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=20,
                                        context=_shared_tls_context()) as response:
                data = response.read()
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ssl.SSLError) as exc:
            if attempt == retries - 1:
                print(f"  ! tile failed ({exc}) {url}", file=sys.stderr)
                return None
            time.sleep(pause * (attempt + 1))
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Download a satellite basemap for a lat/lon box")
    p.add_argument("--flight", help="Derive the box from this flight's .srt telemetry")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"))
    p.add_argument("--margin-m", type=float, default=150.0, help="Padding around the flight track (default: 150 m)")
    p.add_argument("--zoom", type=int, default=None, help="Web-Mercator zoom (default: finest that reaches --target-gsd)")
    p.add_argument("--target-gsd", type=float, default=0.15, help="Desired metres per pixel (default: 0.15)")
    p.add_argument("--source", default="esri", choices=list(SOURCES) + ["custom"])
    p.add_argument("--url-template", help="XYZ template with {z}/{x}/{y}, for --source custom")
    p.add_argument("--out", default=None, help="Output path (default: data/basemap/<name>.jpg)")
    args = p.parse_args()

    if args.bbox:
        min_lat, min_lon, max_lat, max_lon = args.bbox
        name = "bbox"
    elif args.flight:
        min_lat, min_lon, max_lat, max_lon = flight_bbox(args.flight, args.margin_m)
        name = os.path.splitext(os.path.basename(args.flight))[0]
    else:
        raise SystemExit("Give either --flight <srt> or --bbox MIN_LAT MIN_LON MAX_LAT MAX_LON")

    template = args.url_template if args.source == "custom" else SOURCES[args.source]
    if not template:
        raise SystemExit("--source custom requires --url-template")

    mid_lat = (min_lat + max_lat) / 2
    zoom = args.zoom if args.zoom is not None else zoom_for_gsd(mid_lat, args.target_gsd)

    x0, y0 = lonlat_to_pixel(max_lat, min_lon, zoom)   # top-left
    x1, y1 = lonlat_to_pixel(min_lat, max_lon, zoom)   # bottom-right
    tx0, ty0 = int(x0 // TILE_PX), int(y0 // TILE_PX)
    tx1, ty1 = int(x1 // TILE_PX), int(y1 // TILE_PX)
    n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)

    print(f"source : {args.source}  (TLS via "
          f"{'certifi' if _import_ok('certifi') else 'interpreter default'})")
    print(f"box    : {min_lat:.6f},{min_lon:.6f} .. {max_lat:.6f},{max_lon:.6f}")
    print(f"zoom   : {zoom}  ({gsd_at(mid_lat, zoom):.3f} m/px)")
    print(f"tiles  : {n_tiles} ({tx1 - tx0 + 1} x {ty1 - ty0 + 1}) from '{args.source}'")
    if n_tiles > 400:
        raise SystemExit(f"{n_tiles} tiles is more than this script will fetch; lower --zoom or --target-gsd.")

    mosaic = np.zeros(((ty1 - ty0 + 1) * TILE_PX, (tx1 - tx0 + 1) * TILE_PX, 3), np.uint8)
    ok = blank = 0
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            tile = fetch_tile(template.format(z=zoom, x=tx, y=ty))
            if tile is None:
                continue
            if tile.shape[0] != TILE_PX or tile.shape[1] != TILE_PX:
                tile = cv2.resize(tile, (TILE_PX, TILE_PX))
            if is_placeholder(tile):
                blank += 1
                continue
            r, c = (ty - ty0) * TILE_PX, (tx - tx0) * TILE_PX
            mosaic[r:r + TILE_PX, c:c + TILE_PX] = tile
            ok += 1
        print(f"  row {ty - ty0 + 1}/{ty1 - ty0 + 1} ({ok} imagery, {blank} blank, of {n_tiles})",
              flush=True)

    if ok == 0:
        raise SystemExit(
            f"No usable imagery: {blank} of {n_tiles} tiles came back as provider "
            f"placeholders and the rest failed.\n"
            f"'{args.source}' has no coverage at zoom {zoom} for this area. Try a "
            f"coarser --zoom (Esri World Imagery here tops out around 18), or "
            f"--source google, which usually has finer imagery.")
    if blank > ok:
        raise SystemExit(
            f"Refusing to write: {blank} of {ok + blank} downloaded tiles are provider "
            f"placeholders, so most of this basemap would be blank.\n"
            f"Use a coarser --zoom or a different --source.")
    if blank:
        print(f"\nNote: {blank} of {ok + blank} tiles had no imagery and were left blank.")

    # Crop the tile-aligned mosaic down to the requested box, then record the
    # bounds of what we actually kept (not what was asked for).
    ox, oy = tx0 * TILE_PX, ty0 * TILE_PX
    left, top = int(round(x0 - ox)), int(round(y0 - oy))
    right, bottom = int(round(x1 - ox)), int(round(y1 - oy))
    left, top = max(0, left), max(0, top)
    right, bottom = min(mosaic.shape[1], right), min(mosaic.shape[0], bottom)
    cropped = mosaic[top:bottom, left:right]

    kept_max_lat, kept_min_lon = pixel_to_lonlat(ox + left, oy + top, zoom)
    kept_min_lat, kept_max_lon = pixel_to_lonlat(ox + right, oy + bottom, zoom)

    out = args.out or os.path.join("data", "basemap", f"{name}.jpg")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
    sidecar = os.path.splitext(out)[0] + ".json"
    with open(sidecar, "w") as f:
        json.dump({
            "min_lat": kept_min_lat, "min_lon": kept_min_lon,
            "max_lat": kept_max_lat, "max_lon": kept_max_lon,
            "zoom": zoom, "source": args.source,
            "gsd_m_per_px": gsd_at(mid_lat, zoom),
        }, f, indent=2)

    print(f"\nwrote {out}  ({cropped.shape[1]}x{cropped.shape[0]} px, "
          f"{ok}/{n_tiles} tiles with imagery)")
    print(f"wrote {sidecar}")
    print(f"\nNow run:  python run_pipeline.py --video data/raw/{name}.mp4 "
          f"--srt data/raw/{name}.srt --map-source gis --basemap {out}")


if __name__ == "__main__":
    main()
