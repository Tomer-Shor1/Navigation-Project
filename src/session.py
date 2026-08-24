"""Preprocessing, packaged once so every front-end sees the same map.

`run_pipeline.py` (batch: localize everything, write a report and a PNG) and
`nav_player.py` (interactive: step through the same navigation one frame at a
time) need identical setup -- parse the telemetry, extract/reuse frames, split
map vs. test, ORB-index the map, self-calibrate, and optionally swap the map for
an orthomosaic or a GIS raster. Doing that twice would let the thing you *watch*
drift from the thing you *measure*, which would make the visualization worthless
as evidence. So it lives here, and both front-ends call `prepare_session`.

The command-line flags live here too (`add_pipeline_args`), for the same reason:
one definition, so `--map-source ortho` means exactly the same thing in the
player as it does in the pipeline.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

from .build_reference import ReferenceEntry, build_reference_index, split_interleaved, split_reference_test
from .calibrate import TrackCalibration, apply_calibration, calibrate_from_reference
from .extract_frames import FrameRecord, attach_telemetry, extract_frames
from .gis_reference import Basemap, build_gis_index, load_basemap
from .localize import ground_sample_distance
from .orthomosaic import build_ortho_basemap, save_basemap
from .parse_srt import TelemetryRecord, parse_srt
from .reference_source import CompositeReferenceSource

DEFAULT_VIDEO = "data/raw/flight.mp4"
DEFAULT_SRT = "data/raw/flight.srt"

# Anything ffmpeg will decode. Kept broad on purpose: the app is meant to open
# whatever flight the user points it at, not only the files this repo shipped.
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mpg", ".mpeg", ".webm", ".ts")


class SessionInputError(Exception):
    """The inputs are missing or unusable, with a message meant for a human.

    Preprocessing used to print and `sys.exit` here. That is right for a
    command-line run and useless inside a window, where the same condition needs
    to become a dialog and leave the app running -- so it is raised instead, and
    each front-end decides how to show it.
    """


@dataclass(frozen=True)
class FlightSource:
    """A flight video and the telemetry that goes with it."""

    video_path: str
    srt_path: Optional[str]

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.video_path))[0]

    @property
    def has_telemetry(self) -> bool:
        return bool(self.srt_path) and os.path.isfile(self.srt_path)

    @property
    def size_mb(self) -> float:
        try:
            return os.path.getsize(self.video_path) / 1e6
        except OSError:
            return 0.0


def find_telemetry_for(video_path: str) -> Optional[str]:
    """The `.srt` that belongs to a video: same directory, same stem.

    DJI writes `FLIGHT.MP4` alongside `FLIGHT.SRT`, and case varies between
    firmware versions and between filesystems, so the match is case-insensitive
    rather than a plain `os.path.isfile` on a constructed name.
    """
    directory = os.path.dirname(os.path.abspath(video_path))
    wanted = os.path.splitext(os.path.basename(video_path))[0].lower()
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    for name in sorted(entries):
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".srt" and stem.lower() == wanted:
            return os.path.join(directory, name)
    return None


def discover_flights(root: str = "data") -> list[FlightSource]:
    """Every video under `root`, each paired with its telemetry if it has any.

    Extracted frames live under `data/frames/` and are not flights, so that
    subtree is skipped -- otherwise a cached run would show up as a candidate.
    """
    found: list[FlightSource] = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in ("frames", "basemap")]
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in VIDEO_EXTENSIONS:
                continue
            path = os.path.join(directory, name)
            found.append(FlightSource(video_path=path, srt_path=find_telemetry_for(path)))
    return sorted(found, key=lambda f: f.video_path)


def add_pipeline_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the flags that describe *which* flight and *what map* to use.

    Shared by run_pipeline.py and nav_player.py so the two can never disagree
    about what a flag means.
    """
    p.add_argument("--video", default=DEFAULT_VIDEO, help=f"Path to flight video (default: {DEFAULT_VIDEO})")
    p.add_argument("--srt", default=DEFAULT_SRT, help=f"Path to telemetry SRT file (default: {DEFAULT_SRT})")
    p.add_argument("--frames-dir", default=None, help="Directory to extract frames into (default: data/frames/<video-stem>)")
    p.add_argument("--gimbal-pitch", type=float, default=-60.0,
                   help="Assumed gimbal pitch in degrees (0=horizontal, -90=nadir) used when the SRT has no gimbal field (default: -60, the Mini 3 Pro 60-degree angle)")
    p.add_argument("--rate", type=float, default=1.0, help="Frame extraction rate in Hz (default: 1.0)")
    p.add_argument("--split", choices=("interleave", "tail"), default="interleave",
                   help="How to divide frames into map vs. test. 'interleave' (default) holds out "
                        "every Nth frame so the test frames lie inside the mapped corridor -- the "
                        "assignment's scenario. 'tail' uses the first --ref-fraction as the map and "
                        "the rest as test; on these flights that return leg is flown back on the "
                        "opposite heading, which ORB cannot match (see README).")
    p.add_argument("--holdout-every", type=int, default=5,
                   help="With --split interleave: hold out every Nth frame as a test frame (default: 5)")
    p.add_argument("--exclusion-sec", type=float, default=2.0,
                   help="Refuse to localize against map frames recorded within this many seconds of "
                        "the test frame, so a hold-out frame cannot trivially match its own "
                        "neighbours (default: 2.0; use 0 to disable)")
    p.add_argument("--map-source", choices=("flight", "ortho", "gis", "hybrid"), default="flight",
                   help="What the reference map is made of. 'flight' (default, Ex1) indexes the "
                        "reference flight frames themselves. 'ortho' stitches those frames into one "
                        "north-up georeferenced raster and indexes that. 'gis' indexes an external "
                        "satellite/Google-Earth basemap given by --basemap. 'hybrid' indexes both "
                        "the previous flight's frames AND the basemap, and lets them compete per "
                        "frame -- the final project's 'previous videos and GIS datasets'.")
    p.add_argument("--hybrid-policy", choices=("fallback", "compete"), default="fallback",
                   help="With --map-source hybrid: 'fallback' (default) trusts the previous "
                        "flight's frames and only consults the GIS raster for frames they "
                        "could not explain; 'compete' matches both every frame and keeps "
                        "whichever gathered more inliers.")
    p.add_argument("--basemap", default=None,
                   help="Basemap image for --map-source gis (expects a matching .json sidecar with "
                        "min_lat/min_lon/max_lat/max_lon). See tools/fetch_basemap.py.")
    p.add_argument("--patch-px", type=int, default=512, help="Raster map patch size in pixels (default: 512)")
    p.add_argument("--patch-overlap", type=float, default=0.5, help="Overlap between raster map patches (default: 0.5)")
    p.add_argument("--ortho-gsd", type=float, default=0.15, help="Ground sample distance of the built orthomosaic, m/px (default: 0.15)")
    p.add_argument("--save-basemap", default=None, help="Also write the built orthomosaic here as .jpg + .json")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Skip the preprocessing self-calibration and use the modelled "
                        "FOV/altitude/gimbal-pitch ground sample distance instead (see src/calibrate.py)")
    p.add_argument("--ref-fraction", type=float, default=0.8, help="With --split tail: fraction of frames used as reference set (default: 0.8)")
    p.add_argument("--orb-features", type=int, default=2000, help="Max ORB features per frame (default: 2000)")
    p.add_argument("--ratio-thresh", type=float, default=0.75, help="Lowe's ratio test threshold (default: 0.75)")
    p.add_argument("--min-good-matches", type=int, default=15, help="Min good matches to accept a localization (default: 15)")
    p.add_argument("--min-inliers", type=int, default=6, help="Min RANSAC inliers to accept a localization (default: 6)")
    p.add_argument("--min-inlier-ratio", type=float, default=0.12, help="Min RANSAC inlier ratio to accept a localization (default: 0.12)")
    p.add_argument("--top-k", type=int, default=5, help="Top match candidates to verify by homography inliers (default: 5)")
    p.add_argument("--max-evidence-scale", type=float, default=1.0,
                   help="Make the match-acceptance bar stricter as the motion gate widens after a "
                        "dropout. Default 1.0 = off: measured a wash-to-negative once degenerate "
                        "homographies were rejected on geometry instead. See src/localize.py.")
    p.add_argument("--no-motion", action="store_true", help="Disable the motion model / spatial gating (pure image matching)")
    p.add_argument("--fuse", action="store_true", help="Enable multi-reference fusion (inlier-weighted average of all accepted matches). OFF by default: measured wash-to-negative across flights; see README.")
    p.add_argument("--use-cached-frames", action="store_true",
                   help="Reuse the JPEGs already in --frames-dir instead of re-running ffmpeg. "
                        "Lets the whole experiment be reproduced from the committed frames without "
                        "the multi-GB source videos.")
    p.add_argument("--max-seconds", type=float, default=None, help="Only process the first N seconds of the video (for quick/stub runs)")
    p.add_argument("--results-dir", default=None, help="Directory to write outputs into (default: results/<video-stem>)")
    return p


@dataclass
class NavSession:
    """Everything preprocessing produced, ready for the navigation stage."""

    telemetry: list[TelemetryRecord]
    all_frames: list[FrameRecord]
    ref_frames: list[FrameRecord]
    test_frames: list[FrameRecord]
    # A plain list for a single-source map, or a CompositeReferenceSource when
    # two maps are in play; the localizer accepts either.
    reference_index: object
    calibration: Optional[TrackCalibration]
    # Kwargs for localize_all / localize_stream, resolved once so the batch
    # pipeline and the live player navigate under identical settings.
    localize_kwargs: dict = field(default_factory=dict)
    # Set only when the map is a raster (ortho / GIS); the player draws it as the
    # backdrop of its map panel.
    basemap: Optional[Basemap] = None
    map_source: str = "flight"
    # The source video, when there is one. Live playback reads it directly; a
    # run from cached frames may not have it, and then live mode is unavailable.
    video_path: Optional[str] = None
    # Which kinds of map view the navigator can draw on, e.g. {"flight_frame",
    # "gis_tile"} for a hybrid map. Reported so a source that won nothing is
    # still visible in the results.
    source_names: set = field(default_factory=set)
    stem: str = ""
    frames_dir: str = ""
    results_dir: str = ""


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Namespace frames/results by the video stem so multiple flights don't
    clobber each other's outputs (results/<stem>/, data/frames/<stem>/)."""
    stem = os.path.splitext(os.path.basename(args.video))[0]
    if getattr(args, "frames_dir", None) is None:
        args.frames_dir = os.path.join("data/frames", stem)
    if getattr(args, "results_dir", None) is None:
        args.results_dir = os.path.join("results", stem)
    return args


def prepare_session(args: argparse.Namespace, log: Callable[[str], None] = print,
                    cancel_event=None) -> NavSession:
    """Steps 1-4 (plus the optional raster-map swap): turn a flight video +
    telemetry into a georeferenced, ORB-indexed map and a held-out test set.

    Raises `SessionInputError` when the inputs are missing or unusable, so a
    command-line caller can print and exit while a window can show a dialog.
    `cancel_event` is forwarded to frame extraction (see `src/extract_frames.py`).
    """
    resolve_paths(args)
    stem = os.path.splitext(os.path.basename(args.video))[0]

    cached = sorted(glob.glob(os.path.join(args.frames_dir, "frame_*.jpg"))) if args.use_cached_frames else []
    if args.use_cached_frames and not cached:
        raise SessionInputError(
            f"--use-cached-frames given but no frame_*.jpg found in {args.frames_dir}")

    if not os.path.isfile(args.srt) or (not cached and not os.path.isfile(args.video)):
        raise SessionInputError(
            "input flight data not found.\n"
            f"  Expected video:    {args.video}\n"
            f"  Expected telemetry: {args.srt}\n\n"
            "Please place your DJI flight video and matching .srt telemetry\n"
            f"file at {DEFAULT_VIDEO} and {DEFAULT_SRT} (or pass --video/--srt\n"
            "to point at different files), then re-run this script.")

    log(f"[1/6] Parsing telemetry from {args.srt} ...")
    telemetry = parse_srt(args.srt)
    if not telemetry:
        raise SessionInputError(
            f"no usable telemetry could be parsed from {args.srt}.\n"
            "The navigator needs a DJI-style .srt whose subtitle blocks carry GPS\n"
            "coordinates -- check that this is really the telemetry for this video.")
    log(f"      -> {len(telemetry)} telemetry records parsed.")

    if cached:
        log(f"[2/6] Reusing {len(cached)} cached frames from {args.frames_dir} (extracted at {args.rate} Hz) ...")
        raw_frames = [(path, i / args.rate) for i, path in enumerate(cached)]
    else:
        log(f"[2/6] Extracting frames from {args.video} at {args.rate} Hz ...")
        raw_frames = extract_frames(args.video, args.frames_dir, rate_hz=args.rate,
                                    max_seconds=args.max_seconds, cancel_event=cancel_event)
    tagged_frames = attach_telemetry(raw_frames, telemetry)
    log(f"      -> {len(tagged_frames)} frames extracted and tagged.")

    if args.split == "interleave":
        log(f"[3/6] Splitting into map/test sets (interleaved hold-out, every {args.holdout_every}th frame) ...")
        ref_frames, test_frames = split_interleaved(tagged_frames, every=args.holdout_every)
    else:
        log(f"[3/6] Splitting into map/test sets (chronological tail, ref_fraction={args.ref_fraction}) ...")
        ref_frames, test_frames = split_reference_test(tagged_frames, ref_fraction=args.ref_fraction)
    log(f"      -> {len(ref_frames)} reference frames, {len(test_frames)} test frames.")
    if len(test_frames) == 0:
        raise SessionInputError(
            "test set is empty -- this video and sample rate produced too few frames.\n"
            "Try a higher sample rate, or a longer clip.")

    log("[4/6] Building ORB reference index ...")
    reference_index = build_reference_index(
        ref_frames, n_features=args.orb_features, assumed_gimbal_pitch_deg=args.gimbal_pitch
    )
    log(f"      -> reference index built with {len(reference_index)} frames.")

    calibration = None
    if args.no_calibrate:
        log("      -> self-calibration DISABLED; using the modelled camera geometry.")
    else:
        calibration = calibrate_from_reference(reference_index)
        apply_calibration(reference_index, calibration)
        if calibration is not None:
            median_alt = sorted(e.altitude for e in reference_index)[len(reference_index) // 2]
            log(f"      -> self-calibrated from {calibration.n_observations} reference pairs: "
                f"{calibration.gsd_for(median_alt):.4f} m/px at {median_alt:.0f} m "
                f"(modelled: {ground_sample_distance(median_alt, reference_index[0].image_width, reference_index[0].image_height, args.gimbal_pitch)[0]:.4f} m/px), "
                f"scale scatter {calibration.scale_scatter:.2f}, "
                f"{len(calibration.headings_deg)} frame headings measured.")

    # ---- swap the map representation, if asked ------------------------------
    # Everything downstream is untouched: a raster map still arrives as a list of
    # georeferenced, ORB-indexed ReferenceEntry objects.
    frame_gsd = map_gsd = None
    seed_latlon = None
    basemap = None
    raster_index: list[ReferenceEntry] = []
    exclusion_sec = args.exclusion_sec
    if args.map_source != "flight":
        median_alt = sorted(e.altitude for e in reference_index if e.altitude)[len(reference_index) // 2] \
            if any(e.altitude for e in reference_index) else None
        frame_gsd = (calibration.gsd_for(median_alt) if (calibration and median_alt)
                     else ground_sample_distance(median_alt or 100.0,
                                                 reference_index[0].image_width,
                                                 reference_index[0].image_height,
                                                 args.gimbal_pitch)[0])
        if args.map_source == "ortho":
            # Stitching needs a measured scale and heading per frame. Self-
            # calibration provides them, but it needs frame pairs with real
            # displacement -- a clip that is mostly hover, or too short, yields
            # none. Say that plainly instead of failing deep inside the stitcher.
            if calibration is None:
                raise SessionInputError(
                    "this flight could not be self-calibrated, and the orthomosaic map "
                    "is built from the measured scale and heading that calibration "
                    "provides.\n\n"
                    "Calibration needs frame pairs where the drone actually moved. A clip "
                    "that is mostly hover, shot very low, or just too short will not give "
                    "enough of them.\n\n"
                    "Try a longer stretch of the flight, or use the flight's own frames as "
                    "the map -- that path does not need calibration.")
            log(f"[4b/6] Building a north-up orthomosaic from the {len(reference_index)} reference frames ...")
            try:
                basemap = build_ortho_basemap(reference_index, target_gsd_m=args.ortho_gsd,
                                              fallback_gsd_m=frame_gsd)
            except RuntimeError as exc:
                raise SessionInputError(f"the orthomosaic could not be built: {exc}") from exc
        else:
            if not args.basemap:
                raise SessionInputError(
                    "--map-source gis requires --basemap <image>. "
                    "Run tools/fetch_basemap.py to download one.")
            log(f"[4b/6] Loading GIS basemap {args.basemap} ...")
            basemap = load_basemap(args.basemap)
        if args.save_basemap:
            save_basemap(basemap, args.save_basemap)
            log(f"      -> basemap written to {args.save_basemap}")
        map_gsd = basemap.gsd_m_per_px
        raster_index = build_gis_index(basemap, patch_px=args.patch_px,
                                       overlap=args.patch_overlap, n_features=args.orb_features)
        log(f"      -> map: {len(raster_index)} patches at {map_gsd:.3f} m/px; "
            f"drone frames resampled from {frame_gsd:.3f} m/px to match.")
        if args.map_source == "hybrid":
            # Both maps, behind one radius query. The flight frames keep their
            # timestamps so the temporal exclusion still applies to them; raster
            # patches carry timestamp 0, so they are never excluded by it -- which
            # is right, a satellite tile has no "too recent" to be.
            log(f"      -> hybrid map: {len(reference_index)} flight frames + "
                f"{len(raster_index)} raster patches, competing per frame.")
            reference_index = CompositeReferenceSource([reference_index, raster_index])
        else:
            reference_index = raster_index
            # A raster-only map carries no track to seed the motion model from, so
            # the navigator is given its starting position explicitly: the last
            # preprocessing fix before the first test frame.
            prior = [f for f in ref_frames if f.timestamp_sec <= test_frames[0].timestamp_sec] or ref_frames[:1]
            seed_latlon = (prior[-1].latitude, prior[-1].longitude, prior[-1].timestamp_sec)
            exclusion_sec = 0.0   # map patches have no timestamp; nothing to exclude

    # Tier order for a hybrid map: previously-flown video first, GIS raster as
    # the safety net. "native"/"raster" are the frame preparations each map needs
    # (see _view_kind in src/localize.py).
    fallback_order = None
    if args.map_source == "hybrid" and args.hybrid_policy == "fallback":
        fallback_order = ("native", "raster")

    localize_kwargs = dict(
        n_features=args.orb_features,
        ratio_thresh=args.ratio_thresh,
        min_good_matches=args.min_good_matches,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        top_k=args.top_k,
        use_motion=not args.no_motion,
        fuse_references=args.fuse,
        exclusion_sec=exclusion_sec,
        frame_gsd_m_per_px=frame_gsd,
        map_gsd_m_per_px=map_gsd,
        frame_scale_per_alt=(calibration.scale_per_alt_m_per_px if calibration else None),
        max_evidence_scale=args.max_evidence_scale,
        fallback_order=fallback_order,
        seed_latlon=seed_latlon,
    )

    from .reference_source import as_reference_source
    source_names = {e.source for e in as_reference_source(reference_index).all()}

    return NavSession(
        telemetry=telemetry,
        all_frames=tagged_frames,
        ref_frames=ref_frames,
        test_frames=test_frames,
        reference_index=reference_index,
        calibration=calibration,
        localize_kwargs=localize_kwargs,
        basemap=basemap,
        map_source=args.map_source,
        video_path=args.video if os.path.isfile(args.video) else None,
        source_names=source_names,
        stem=stem,
        frames_dir=args.frames_dir,
        results_dir=args.results_dir,
    )
