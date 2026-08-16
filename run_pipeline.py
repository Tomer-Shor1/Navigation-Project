"""End-to-end GPS-denied visual navigation MVP pipeline.

Usage (from a fresh clone, with a virtualenv activated and requirements installed):

    python run_pipeline.py

By default this expects:
    data/raw/flight.mp4
    data/raw/flight.srt

If those files aren't present, this script prints a clear error and exits
rather than crashing -- see --video/--srt to point at a placeholder clip for
a quick sanity run (e.g. --max-seconds 10 to only use the first 10s).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from src.parse_srt import parse_srt
from src.extract_frames import extract_frames, attach_telemetry
from src.build_reference import split_reference_test, split_interleaved, build_reference_index
from src.calibrate import calibrate_from_reference, apply_calibration
from src.localize import localize_all, ground_sample_distance
from src.evaluate import write_error_report
from src.visualize import plot_trajectory

DEFAULT_VIDEO = "data/raw/flight.mp4"
DEFAULT_SRT = "data/raw/flight.srt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPS-denied visual navigation MVP pipeline")
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
    p.add_argument("--no-motion", action="store_true", help="Disable the motion model / spatial gating (pure image matching)")
    p.add_argument("--fuse", action="store_true", help="Enable multi-reference fusion (inlier-weighted average of all accepted matches). OFF by default: measured wash-to-negative across flights; see README.")
    p.add_argument("--use-cached-frames", action="store_true",
                   help="Reuse the JPEGs already in --frames-dir instead of re-running ffmpeg. "
                        "Lets the whole experiment be reproduced from the committed frames without "
                        "the multi-GB source videos.")
    p.add_argument("--max-seconds", type=float, default=None, help="Only process the first N seconds of the video (for quick/stub runs)")
    p.add_argument("--results-dir", default=None, help="Directory to write outputs into (default: results/<video-stem>)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Namespace frames/results by the video stem so multiple flights don't
    # clobber each other's outputs (results/<stem>/, data/frames/<stem>/).
    stem = os.path.splitext(os.path.basename(args.video))[0]
    if args.frames_dir is None:
        args.frames_dir = os.path.join("data/frames", stem)
    if args.results_dir is None:
        args.results_dir = os.path.join("results", stem)

    cached = sorted(glob.glob(os.path.join(args.frames_dir, "frame_*.jpg"))) if args.use_cached_frames else []
    if args.use_cached_frames and not cached:
        print(f"ERROR: --use-cached-frames given but no frame_*.jpg found in {args.frames_dir}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.srt) or (not cached and not os.path.isfile(args.video)):
        print("=" * 70)
        print("ERROR: input flight data not found.")
        print(f"  Expected video:    {args.video}")
        print(f"  Expected telemetry: {args.srt}")
        print()
        print("Please place your DJI flight video and matching .srt telemetry")
        print(f"file at {DEFAULT_VIDEO} and {DEFAULT_SRT} (or pass --video/--srt")
        print("to point at different files), then re-run this script.")
        print("=" * 70)
        sys.exit(1)

    print(f"[1/6] Parsing telemetry from {args.srt} ...")
    telemetry = parse_srt(args.srt)
    assert telemetry, "Sanity check failed: parsed telemetry table is empty."
    print(f"      -> {len(telemetry)} telemetry records parsed.")

    if cached:
        print(f"[2/6] Reusing {len(cached)} cached frames from {args.frames_dir} (extracted at {args.rate} Hz) ...")
        raw_frames = [(path, i / args.rate) for i, path in enumerate(cached)]
    else:
        print(f"[2/6] Extracting frames from {args.video} at {args.rate} Hz ...")
        raw_frames = extract_frames(args.video, args.frames_dir, rate_hz=args.rate, max_seconds=args.max_seconds)
    tagged_frames = attach_telemetry(raw_frames, telemetry)
    print(f"      -> {len(tagged_frames)} frames extracted and tagged.")

    if args.split == "interleave":
        print(f"[3/6] Splitting into map/test sets (interleaved hold-out, every {args.holdout_every}th frame) ...")
        ref_frames, test_frames = split_interleaved(tagged_frames, every=args.holdout_every)
    else:
        print(f"[3/6] Splitting into map/test sets (chronological tail, ref_fraction={args.ref_fraction}) ...")
        ref_frames, test_frames = split_reference_test(tagged_frames, ref_fraction=args.ref_fraction)
    print(f"      -> {len(ref_frames)} reference frames, {len(test_frames)} test frames.")
    if len(test_frames) == 0:
        print("ERROR: test set is empty -- video/rate combination produced too few frames.", file=sys.stderr)
        sys.exit(1)

    print("[4/6] Building ORB reference index ...")
    reference_index = build_reference_index(
        ref_frames, n_features=args.orb_features, assumed_gimbal_pitch_deg=args.gimbal_pitch
    )
    print(f"      -> reference index built with {len(reference_index)} frames.")

    if args.no_calibrate:
        print("      -> self-calibration DISABLED; using the modelled camera geometry.")
    else:
        calibration = calibrate_from_reference(reference_index)
        apply_calibration(reference_index, calibration)
        if calibration is not None:
            median_alt = sorted(e.altitude for e in reference_index)[len(reference_index) // 2]
            print(f"      -> self-calibrated from {calibration.n_observations} reference pairs: "
                  f"{calibration.gsd_for(median_alt):.4f} m/px at {median_alt:.0f} m "
                  f"(modelled: {ground_sample_distance(median_alt, reference_index[0].image_width, reference_index[0].image_height, args.gimbal_pitch)[0]:.4f} m/px), "
                  f"scale scatter {calibration.scale_scatter:.2f}, "
                  f"{len(calibration.headings_deg)} frame headings measured.")

    motion_desc = "pure image matching" if args.no_motion else "motion-gated + inlier confidence"
    print(f"[5/6] Localizing test frames against the reference index ({motion_desc}) ...")
    results = localize_all(
        test_frames, reference_index,
        n_features=args.orb_features,
        ratio_thresh=args.ratio_thresh,
        min_good_matches=args.min_good_matches,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        top_k=args.top_k,
        use_motion=not args.no_motion,
        fuse_references=args.fuse,
        exclusion_sec=args.exclusion_sec,
    )
    n_map = sum(1 for r in results if r.mode == "map_fix")
    n_dr = sum(1 for r in results if r.mode == "dead_reckon")
    print(f"      -> {n_map} visual map fixes, {n_dr} dead-reckoned, out of {len(results)} frames.")

    print("[6/6] Evaluating and visualizing ...")
    report_path = os.path.join(args.results_dir, "error_report.txt")
    summary = write_error_report(results, report_path)
    plot_path = os.path.join(args.results_dir, "trajectory_comparison.png")
    plot_trajectory(telemetry, results, plot_path)

    print()
    print("=" * 40)
    print("DONE")
    print(f"  Frames:       {summary['n_total']}")
    print(f"  Map fixes:    {summary['n_map_fix']}   dead-reckoned: {summary['n_dead_reckon']}   failed: {summary['n_failed']}")
    if summary["median_error_m"] is not None:
        print(f"  Map-fix accuracy (trusted):")
        print(f"    median: {summary['median_error_m']:.2f} m   mean: {summary['mean_error_m']:.2f} m   "
              f"rmse: {summary['rmse_error_m']:.2f} m   p90: {summary['p90_error_m']:.2f} m   max: {summary['max_error_m']:.2f} m")
    else:
        print("  No trusted map fixes -- no accuracy stats.")
    print(f"  Report:      {report_path}")
    print(f"  Plot:        {plot_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
