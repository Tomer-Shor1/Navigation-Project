"""A seekable, lazily-computed record of what the navigator did on each frame.

`run_pipeline.py` answers "how accurate was it". This module answers "what did
it actually look at, and why did it decide that" -- for one frame at a time, in
order, so a UI can step through the run.

Two properties matter and drive the design:

* **It is the real run, not a replay of a summary.** A `NavigationTrace` drives
  `localize_stream` (the same generator `localize_all` collects) and simply
  records the `debug` dict that `localize_frame` fills in. Nothing here
  re-implements matching, gating or acceptance, so what you watch is by
  construction what the report measured.
* **It is cheap to hold.** The navigation loop is forward-only and
  deterministic, so seeking *backwards* is a cache read and seeking *forwards*
  runs the algorithm for real. A `FrameTrace` therefore stores no images -- only
  paths, small coordinate arrays and scalars -- and the render helpers below
  reconstruct the exact pixels on demand.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

import cv2
import numpy as np

from .build_reference import ReferenceEntry
from .geo import haversine_distance_m
from .gis_reference import Basemap, normalize_for_matching
from .localize import RASTER_SOURCES, LocalizationResult, localize_stream
from .motion import MotionState
from .session import NavSession


@dataclass
class CandidateSummary:
    """One map view the frame was compared against, and how it fared.

    `verdict` is the whole point: a candidate can lose because its homography
    was degenerate, because the frame centre projected off the view entirely, or
    simply because it did not clear the inlier bar -- and those are very
    different stories to tell about a frame.
    """

    label: str
    frame_path: str
    n_good: int
    n_inliers: int
    inlier_ratio: float
    verdict: str
    is_best: bool = False
    source: str = "flight_frame"

    @property
    def map_label(self) -> str:
        return "satellite" if self.source in RASTER_SOURCES else "flight"

    @property
    def short_verdict(self) -> str:
        """A few words, for a narrow table column."""
        v = self.verdict
        if v.startswith("below bar"):
            return "below bar"
        if v.startswith("rejected: "):
            return {"degenerate homography": "degenerate H",
                    "centre projects off the reference view": "centre off view",
                    "no homography": "no homography"}.get(v[10:], v[10:])
        return v


@dataclass
class FrameTrace:
    """Everything needed to draw one step of the navigation loop."""

    index: int
    frame_path: str
    timestamp_sec: float
    true_latitude: float
    true_longitude: float
    altitude: Optional[float]
    result: LocalizationResult

    # --- the image the matcher actually saw -------------------------------
    # (h, w) of the matched image, plus how to rebuild it: against a raster map
    # the frame is resampled to the map's ground scale and contrast-equalised,
    # so the raw JPEG's pixel coordinates are *not* the ones the keypoints are in.
    image_shape: tuple = (0, 0)
    clahe: bool = False
    test_kp_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))

    # --- the motion model's prediction and search gate ---------------------
    predicted_latlon: Optional[tuple] = None
    gate_radius_m: Optional[float] = None
    n_in_gate: int = 0
    n_searched: int = 0
    searched_latlon: list = field(default_factory=list)
    reacquisition: bool = False
    bar: dict = field(default_factory=dict)

    # --- the winning match -------------------------------------------------
    matched_source: Optional[str] = None
    # Which map tier produced the fix ("native" = previous flight video,
    # "raster" = GIS), and which tiers were available to try.
    tier_used: Optional[str] = None
    tiers_available: list = field(default_factory=list)
    best_entry: Optional[ReferenceEntry] = None
    src_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    dst_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    is_inlier: np.ndarray = field(default_factory=lambda: np.zeros((0,), bool))
    homography: Optional[np.ndarray] = None
    projected_outline: Optional[np.ndarray] = None   # 4x2, reference pixel coords
    projected_centre: Optional[np.ndarray] = None    # 2,  reference pixel coords

    candidates: list[CandidateSummary] = field(default_factory=list)
    # Wall-clock cost of this frame's decision, by phase (see localize_frame).
    timing_ms: dict = field(default_factory=dict)
    error_m: Optional[float] = None
    motion_after: Optional[MotionState] = None

    @property
    def name(self) -> str:
        return os.path.basename(self.frame_path)

    @property
    def mode(self) -> str:
        return self.result.mode

    @property
    def n_inliers(self) -> int:
        return int(self.is_inlier.sum()) if self.is_inlier.size else 0

    @property
    def compute_ms(self) -> float:
        return float(self.timing_ms.get("total_ms", 0.0))

    @property
    def used_gis(self) -> bool:
        """Did this fix come from the GIS raster rather than previous video?"""
        return self.matched_source in RASTER_SOURCES

    @property
    def gis_was_fallback(self) -> bool:
        """...and was the previous-flight map available but unable to explain it?"""
        return self.used_gis and "native" in self.tiers_available


def entry_label(entry: ReferenceEntry) -> str:
    """Short human-readable name for a reference view (a flight frame or a map tile)."""
    if entry.frame_path.startswith("gis://"):
        m = re.search(r"patch_r(\d+)_c(\d+)", entry.frame_path)
        return f"tile r{m.group(1)} c{m.group(2)}" if m else "tile"
    return os.path.basename(entry.frame_path)


def _summarise_candidates(debug: dict) -> list[CandidateSummary]:
    """Turn the localizer's debug record into an ordered, explainable list."""
    accepted = debug.get("accepted") or []
    accepted_ids = {id(d) for d in accepted}
    best_id = id(accepted[0]) if accepted else None
    bar = debug.get("bar") or {}

    out: list[CandidateSummary] = []
    for d in debug.get("verified") or []:
        if id(d) in accepted_ids:
            verdict = "accepted"
        else:
            # Say which threshold it missed -- "below bar" alone explains nothing.
            missed = []
            if d["n_in"] < bar.get("min_inliers", 0):
                missed.append(f"inliers < {bar['min_inliers']}")
            if d["ratio"] < bar.get("min_inlier_ratio", 0):
                missed.append(f"ratio < {bar['min_inlier_ratio']:.2f}")
            if len(d["good"]) < bar.get("min_good_matches", 0):
                missed.append(f"good < {bar['min_good_matches']}")
            verdict = "below bar: " + ", ".join(missed) if missed else "below bar"
        out.append(CandidateSummary(
            label=entry_label(d["entry"]), frame_path=d["entry"].frame_path,
            n_good=len(d["good"]), n_inliers=d["n_in"], inlier_ratio=d["ratio"],
            verdict=verdict, is_best=(id(d) == best_id), source=d["entry"].source,
        ))
    for entry, n_good, reason in debug.get("rejected") or []:
        out.append(CandidateSummary(
            label=entry_label(entry), frame_path=entry.frame_path,
            n_good=n_good, n_inliers=0, inlier_ratio=0.0,
            verdict=f"rejected: {reason}", source=entry.source,
        ))
    return out


def _build_frame_trace(index, test_frame, result, debug, motion_state) -> FrameTrace:
    trace = FrameTrace(
        index=index,
        frame_path=test_frame.frame_path,
        timestamp_sec=test_frame.timestamp_sec,
        true_latitude=test_frame.latitude,
        true_longitude=test_frame.longitude,
        altitude=test_frame.altitude,
        result=result,
        motion_after=copy.copy(motion_state) if motion_state is not None else None,
    )
    if result.has_estimate:
        trace.error_m = haversine_distance_m(
            result.true_latitude, result.true_longitude,
            result.estimated_latitude, result.estimated_longitude,
        )
    if not debug:
        return trace

    trace.image_shape = tuple(debug.get("image_shape", (0, 0)))
    trace.clahe = bool(debug.get("clahe", False))
    trace.test_kp_xy = debug.get("test_kp_xy", trace.test_kp_xy)
    trace.predicted_latlon = debug.get("predicted_latlon")
    trace.gate_radius_m = debug.get("gate_radius_m")
    trace.n_in_gate = debug.get("n_in_gate", 0)
    trace.n_searched = debug.get("n_searched", 0)
    trace.searched_latlon = debug.get("searched_latlon", [])
    trace.reacquisition = bool(debug.get("reacquisition", False))
    trace.bar = debug.get("bar") or {}
    trace.timing_ms = debug.get("timing") or {}
    trace.tier_used = debug.get("tier_used")
    trace.tiers_available = debug.get("tiers_available") or []
    trace.candidates = _summarise_candidates(debug)
    trace.matched_source = result.matched_source

    best = debug.get("best")
    if best is not None:
        entry, good = best["entry"], best["good"]
        trace.best_entry = entry
        trace.homography = best["H"]
        trace.src_xy = np.float32([trace.test_kp_xy[m.queryIdx] for m in good]) \
            if len(good) and trace.test_kp_xy.size else np.zeros((0, 2), np.float32)
        trace.dst_xy = np.float32([entry.keypoints[m.trainIdx].pt for m in good]) \
            if len(good) else np.zeros((0, 2), np.float32)
        mask = best.get("inlier_mask")
        trace.is_inlier = mask if mask is not None else np.zeros(len(good), bool)

        h, w = trace.image_shape if trace.image_shape[0] else (0, 0)
        if h and w:
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            trace.projected_outline = cv2.perspectiveTransform(
                corners, trace.homography).reshape(-1, 2)
            trace.projected_centre = cv2.perspectiveTransform(
                np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2), trace.homography).reshape(2)
    return trace


class NavigationTrace:
    """The navigation run as a random-access sequence of `FrameTrace`.

    Frames are computed on demand and kept, so `ensure(i)` is instant for any
    `i` already reached and costs one real localization per new frame beyond it.
    """

    def __init__(self, session: NavSession):
        self.session = session
        self._frames: list[FrameTrace] = []
        self._stream = localize_stream(
            session.test_frames, session.reference_index,
            collect_debug=True, verbose=False, **session.localize_kwargs,
        )
        self._exhausted = False

    @property
    def n_frames(self) -> int:
        return len(self.session.test_frames)

    @property
    def n_computed(self) -> int:
        return len(self._frames)

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, index: int) -> FrameTrace:
        self.ensure(index)
        return self._frames[index]

    def computed_frames(self) -> list[FrameTrace]:
        return self._frames

    def step(self) -> Optional[FrameTrace]:
        """Advance the navigator by exactly one frame. None once it is done."""
        if self._exhausted:
            return None
        try:
            result, debug, motion_state = next(self._stream)
        except StopIteration:
            self._exhausted = True
            return None
        index = len(self._frames)
        trace = _build_frame_trace(
            index, self.session.test_frames[index], result, debug, motion_state)
        self._frames.append(trace)
        return trace

    def ensure(self, index: int, progress: Optional[Callable[[int, int], None]] = None) -> None:
        """Run the navigator forward until `index` has been computed.

        `progress(done, target)` is called before each new frame, so a UI can say
        what it is waiting on instead of freezing silently.
        """
        if index < 0 or index >= self.n_frames:
            raise IndexError(f"frame {index} out of range (0..{self.n_frames - 1})")
        while len(self._frames) <= index:
            if progress is not None:
                progress(len(self._frames), index)
            if self.step() is None:
                raise RuntimeError(
                    f"navigation stream ended after {len(self._frames)} frames, "
                    f"but frame {index} was requested")


# ---------------------------------------------------------------------------
# Rendering helpers: rebuild, on demand, exactly the pixels the matcher saw.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _read_gray(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


@lru_cache(maxsize=16)
def _read_rgb(path: str) -> Optional[np.ndarray]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


_NORMALISED_BASEMAPS: dict = {}


def _normalised_basemap(basemap: Basemap) -> np.ndarray:
    """`build_gis_index` equalises the whole raster once and *then* cuts patches,
    so a tile must be cropped from the equalised raster, not equalised after
    cropping -- CLAHE is tile-local and the two are not the same image.
    """
    cached = _NORMALISED_BASEMAPS.get(id(basemap))
    if cached is None:
        cached = normalize_for_matching(basemap.image)
        _NORMALISED_BASEMAPS[id(basemap)] = cached
    return cached


def render_test_image(trace: FrameTrace) -> Optional[np.ndarray]:
    """The current frame as RGB, in the same pixel coordinates as `test_kp_xy`.

    Colour when the matcher used the frame at native resolution; the resampled,
    contrast-equalised grayscale when it did not -- otherwise the keypoints
    drawn on top would not line up with what they were computed from.
    """
    h, w = trace.image_shape if trace.image_shape and trace.image_shape[0] else (0, 0)
    if not trace.clahe and h:
        image = _read_rgb(trace.frame_path)
        if image is not None and image.shape[:2] == (h, w):
            return image
    gray = _read_gray(trace.frame_path)
    if gray is None:
        return None
    if h and gray.shape[:2] != (h, w):
        interp = cv2.INTER_AREA if w < gray.shape[1] else cv2.INTER_LINEAR
        gray = cv2.resize(gray, (w, h), interpolation=interp)
    if trace.clahe:
        gray = normalize_for_matching(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def render_ref_image(entry: ReferenceEntry, basemap: Optional[Basemap] = None) -> Optional[np.ndarray]:
    """The matched reference view as RGB: a flight frame from disk, or the tile
    cut out of the basemap raster."""
    if not entry.frame_path.startswith("gis://"):
        return _read_rgb(entry.frame_path)
    if basemap is None:
        return None
    m = re.search(r"patch_r(\d+)_c(\d+)", entry.frame_path)
    if not m:
        return None
    row, col = int(m.group(1)), int(m.group(2))
    patch = _normalised_basemap(basemap)[row:row + entry.image_height,
                                         col:col + entry.image_width]
    return cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
