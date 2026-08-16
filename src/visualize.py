"""Step 9: plot the true GPS trajectory (full flight) against the estimated
positions (test-set only) in local x/y meters (via UTM), and save a PNG.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import utm

from .parse_srt import TelemetryRecord
from .localize import LocalizationResult


def _to_local_xy(lat: float, lon: float, origin_easting: float, origin_northing: float, zone_number: int, zone_letter: str):
    easting, northing, _, _ = utm.from_latlon(lat, lon, force_zone_number=zone_number, force_zone_letter=zone_letter)
    return easting - origin_easting, northing - origin_northing


def plot_trajectory(
    telemetry: list[TelemetryRecord],
    results: list[LocalizationResult],
    output_path: str,
) -> None:
    """Plot the true full-flight trajectory (from SRT) and estimated test-set
    positions on a common local x/y (meters) frame, and save to `output_path`.
    """
    assert telemetry, "Sanity check failed: no telemetry to plot."

    origin_easting, origin_northing, zone_number, zone_letter = utm.from_latlon(
        telemetry[0].latitude, telemetry[0].longitude
    )

    true_xy = [
        _to_local_xy(r.latitude, r.longitude, origin_easting, origin_northing, zone_number, zone_letter)
        for r in telemetry
    ]
    true_x, true_y = zip(*true_xy)

    map_x, map_y = [], []      # trusted visual map fixes
    dr_x, dr_y = [], []        # dead-reckoned (coasted) positions
    true_test_x, true_test_y = [], []
    for r in results:
        tx, ty = _to_local_xy(r.true_latitude, r.true_longitude, origin_easting, origin_northing, zone_number, zone_letter)
        true_test_x.append(tx)
        true_test_y.append(ty)
        if r.has_estimate:
            ex, ey = _to_local_xy(
                r.estimated_latitude, r.estimated_longitude, origin_easting, origin_northing, zone_number, zone_letter
            )
            (map_x if r.mode == "map_fix" else dr_x).append(ex)
            (map_y if r.mode == "map_fix" else dr_y).append(ey)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(true_x, true_y, "-", color="tab:blue", linewidth=1.5, label="True GPS trajectory (full flight)")
    ax.scatter(true_test_x, true_test_y, color="tab:blue", marker="o", s=30, label="True position (test frames)")
    if map_x:
        ax.scatter(map_x, map_y, color="tab:red", marker="x", s=50, label="Estimated (visual map fix)")
    if dr_x:
        ax.scatter(dr_x, dr_y, color="tab:orange", marker="^", s=45, label="Estimated (dead-reckoned)")
    # error links from each true test position to its estimate
    for r, tx, ty in zip(results, true_test_x, true_test_y):
        if r.has_estimate:
            ex, ey = _to_local_xy(
                r.estimated_latitude, r.estimated_longitude, origin_easting, origin_northing, zone_number, zone_letter
            )
            ax.plot([tx, ex], [ty, ey], color="gray", linewidth=0.5, alpha=0.6)

    n_map = sum(1 for r in results if r.mode == "map_fix")
    n_dr = sum(1 for r in results if r.mode == "dead_reckon")
    ax.set_title(f"GPS-denied visual navigation ({n_map} map fixes, {n_dr} dead-reckoned / {len(results)} frames)")
    ax.set_xlabel("Local East offset (m)")
    ax.set_ylabel("Local North offset (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("This module is invoked by run_pipeline.py after evaluate.py produces results.")
