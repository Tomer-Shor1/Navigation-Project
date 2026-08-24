"""Step 8: evaluate localization accuracy in meters against SRT ground truth.

Reports errors broken down by *mode*: a trusted visual **map fix** is a very
different thing from a **dead-reckoned** position coasted on the motion model, so
lumping them into one number would be misleading. For each mode it reports
median / mean / RMSE / p90 / max (the error distribution is bimodal, so a single
mean hides the story).
"""

from __future__ import annotations

import os

from typing import Optional

from .geo import haversine_distance_m
from .localize import LocalizationResult


def _result_error_m(r: LocalizationResult) -> float:
    return haversine_distance_m(r.true_latitude, r.true_longitude, r.estimated_latitude, r.estimated_longitude)


def compute_errors(results: list[LocalizationResult]) -> list[float]:
    """Per-frame error in meters for trusted visual map fixes."""
    return [_result_error_m(r) for r in results if r.success]


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,1]) of an already-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _stats(errors: list[float]) -> dict:
    if not errors:
        return {"n": 0, "median_m": None, "mean_m": None, "rmse_m": None, "p90_m": None, "max_m": None}
    s = sorted(errors)
    n = len(s)
    return {
        "n": n,
        "median_m": _percentile(s, 0.5),
        "mean_m": sum(s) / n,
        "rmse_m": (sum(e * e for e in s) / n) ** 0.5,
        "p90_m": _percentile(s, 0.9),
        "max_m": s[-1],
    }


def _fmt(stats: dict) -> str:
    if not stats["n"]:
        return "none"
    return (f"median {stats['median_m']:.2f}  mean {stats['mean_m']:.2f}  rmse {stats['rmse_m']:.2f}  "
            f"p90 {stats['p90_m']:.2f}  max {stats['max_m']:.2f}")


SOURCE_LABELS = {"flight_frame": "previous flight video", "gis_tile": "GIS / satellite raster"}


def source_split(results: list[LocalizationResult]) -> dict:
    """How many trusted fixes each map contributed, and how good they were.

    On a hybrid map the two sources compete per frame, so this is the number
    that says whether the satellite raster is actually pulling its weight or
    just riding along.
    """
    split: dict = {}
    for r in results:
        if r.mode != "map_fix":
            continue
        entry = split.setdefault(r.matched_source or "unknown", {"n": 0, "errors": []})
        entry["n"] += 1
        entry["errors"].append(_result_error_m(r))
    for entry in split.values():
        entry["stats"] = _stats(entry["errors"])
    return split


def write_error_report(results: list[LocalizationResult], output_path: str,
                       available_sources: Optional[set] = None) -> dict:
    """Write a plain-text error report and return summary stats.

    `available_sources` is the set of map kinds the navigator could have drawn
    on. Pass it when more than one was offered (a hybrid map) and the report
    gains a per-source breakdown -- including a source that was available and
    won *nothing*, which is exactly the result worth seeing.
    """
    n_total = len(results)
    map_errs = [_result_error_m(r) for r in results if r.mode == "map_fix"]
    dr_errs = [_result_error_m(r) for r in results if r.mode == "dead_reckon"]
    all_errs = [_result_error_m(r) for r in results if r.has_estimate]

    map_stats = _stats(map_errs)
    dr_stats = _stats(dr_errs)
    all_stats = _stats(all_errs)

    n_map = map_stats["n"]
    n_dr = dr_stats["n"]
    n_failed = sum(1 for r in results if not r.has_estimate)

    by_source = source_split(results)

    summary = {
        "n_total": n_total,
        "source_split": {k: {"n": v["n"], "median_m": v["stats"]["median_m"]}
                         for k, v in by_source.items()},
        "n_map_fix": n_map,
        "n_dead_reckon": n_dr,
        "n_failed": n_failed,
        # Headline accuracy = trusted map fixes.
        "median_error_m": map_stats["median_m"],
        "mean_error_m": map_stats["mean_m"],
        "rmse_error_m": map_stats["rmse_m"],
        "p90_error_m": map_stats["p90_m"],
        "max_error_m": map_stats["max_m"],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("GPS-denied localization -- error report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Test frames total:       {n_total}\n")
        f.write(f"Trusted map fixes:       {n_map}\n")
        f.write(f"Dead-reckoned (coasted): {n_dr}\n")
        f.write(f"No estimate at all:      {n_failed}\n\n")

        f.write("Error (m) vs true GPS, by mode:\n")
        f.write(f"  map fixes ({n_map}):     {_fmt(map_stats)}\n")
        f.write(f"  dead-reckon ({n_dr}):    {_fmt(dr_stats)}\n")
        f.write(f"  all reported ({all_stats['n']}):  {_fmt(all_stats)}\n\n")

        if available_sources and len(available_sources) > 1:
            f.write("Which map produced each fix:\n")
            for name in sorted(available_sources,
                               key=lambda n: -by_source.get(n, {}).get("n", 0)):
                entry = by_source.get(name)
                label = SOURCE_LABELS.get(name, name)
                if not entry:
                    f.write(f"  {label:<24}   0 fixes (  0%)  offered, never won\n")
                    continue
                share = 100.0 * entry["n"] / max(n_map, 1)
                f.write(f"  {label:<24} {entry['n']:>3} fixes ({share:>4.0f}%)  {_fmt(entry['stats'])}\n")
            f.write("\n")

        f.write("Per-frame detail:\n")
        for r in results:
            name = os.path.basename(r.frame_path)
            if r.mode == "map_fix":
                tag = "gate" if r.gated else "reacq"
                # Only worth naming when more than one map was in play; on a
                # single-source run every line would carry the same tag.
                origin = ""
                if r.matched_source and available_sources and len(available_sources) > 1:
                    origin = f" src={r.matched_source}"
                f.write(
                    f"  {name} t={r.timestamp_sec:.2f}s MAP  [{tag}] "
                    f"good={r.num_good_matches} inliers={r.num_inliers} ratio={r.inlier_ratio:.2f} "
                    f"error={_result_error_m(r):.2f}m matched_ref={os.path.basename(r.matched_ref_frame)}"
                    f"{origin}\n"
                )
            elif r.mode == "dead_reckon":
                f.write(
                    f"  {name} t={r.timestamp_sec:.2f}s DR   [coast #{r.coast_steps}] "
                    f"error={_result_error_m(r):.2f}m ({r.failure_reason})\n"
                )
            else:
                f.write(f"  {name} t={r.timestamp_sec:.2f}s FAILED ({r.failure_reason})\n")

    return summary


if __name__ == "__main__":
    print("This module is invoked by run_pipeline.py after localize.py produces results.")
