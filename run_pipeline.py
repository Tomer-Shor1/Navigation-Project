"""End-to-end GPS-denied visual navigation MVP pipeline.

Usage (from a fresh clone, with a virtualenv activated and requirements installed):

    python run_pipeline.py

By default this expects:
    data/raw/flight.mp4
    data/raw/flight.srt

If those files aren't present, this script prints a clear error and exits
rather than crashing -- see --video/--srt to point at a placeholder clip for
a quick sanity run (e.g. --max-seconds 10 to only use the first 10s).

To *watch* this same navigation run frame by frame instead of just reading its
report, use `python nav_player.py` -- it takes the same flags and drives the
same code.
"""

from __future__ import annotations

import argparse
import os
import sys

from src.session import SessionInputError, add_pipeline_args, prepare_session
from src.localize import localize_all
from src.evaluate import write_error_report
from src.visualize import plot_trajectory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPS-denied visual navigation MVP pipeline")
    add_pipeline_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        session = prepare_session(args)
    except SessionInputError as exc:
        # Preprocessing raises rather than exiting, so the window front-end can
        # show a dialog; on the command line it is still a message and exit 1.
        print("=" * 70, file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(1)

    motion_desc = "pure image matching" if args.no_motion else "motion-gated + inlier confidence"
    print(f"[5/6] Localizing test frames against the reference index ({motion_desc}) ...")
    results = localize_all(session.test_frames, session.reference_index, **session.localize_kwargs)
    n_map = sum(1 for r in results if r.mode == "map_fix")
    n_dr = sum(1 for r in results if r.mode == "dead_reckon")
    print(f"      -> {n_map} visual map fixes, {n_dr} dead-reckoned, out of {len(results)} frames.")

    print("[6/6] Evaluating and visualizing ...")
    report_path = os.path.join(session.results_dir, "error_report.txt")
    summary = write_error_report(results, report_path, available_sources=session.source_names)
    plot_path = os.path.join(session.results_dir, "trajectory_comparison.png")
    plot_trajectory(session.telemetry, results, plot_path)

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
