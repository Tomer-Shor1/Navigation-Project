"""Measure whether the navigator is actually real-time, and what it costs.

    python tools/benchmark.py --use-cached-frames
    python tools/benchmark.py --use-cached-frames --map-source hybrid --basemap data/basemap/flight.jpg

"Real-time" for this system means one thing: a frame must be localized in less
than the interval between frames. At the default 1 Hz that budget is 1000 ms.
This reports the per-phase cost of the decision, the margin against that budget,
and -- the part that actually matters for a map that grows -- how the cost scales
with the number of map views the motion gate lets through.

That last column is the argument for gating. Matching is linear in candidates,
so an ungated search over a whole city is hopeless while a gated search over the
drone's own neighbourhood stays flat no matter how large the map gets.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reference_source import as_reference_source          # noqa: E402
from src.session import add_pipeline_args, prepare_session     # noqa: E402
from src.trace import NavigationTrace                          # noqa: E402

PHASES = (("decode_ms", "decode JPEG"), ("prepare_ms", "ORB on the frame"),
          ("gate_ms", "motion gate"), ("match_ms", "descriptor matching"),
          ("verify_ms", "RANSAC verify"))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    return ordered[low] + (pos - low) * (ordered[min(low + 1, len(ordered) - 1)] - ordered[low])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_pipeline_args(parser)
    args = parser.parse_args()

    session = prepare_session(args, log=lambda msg: None)
    n_map = len(as_reference_source(session.reference_index).all())
    budget_ms = 1000.0 / args.rate

    trace = NavigationTrace(session)
    trace.ensure(trace.n_frames - 1)
    frames = trace.computed_frames()
    timings = [f.timing_ms for f in frames if f.timing_ms]
    if not timings:
        raise SystemExit("No timing recorded -- nothing to report.")

    totals = [t["total_ms"] for t in timings]
    searched = [t.get("n_searched", 0) for t in timings]

    print(f"map          : {n_map} views ({session.map_source}, "
          f"{'+'.join(sorted(session.source_names))})")
    print(f"frames       : {len(frames)} localized at {args.rate} Hz "
          f"-> {budget_ms:.0f} ms budget per frame")
    print(f"gate         : {statistics.median(searched):.0f} views searched (median), "
          f"{max(searched)} worst case, out of {n_map}\n")

    print(f"{'phase':<22}{'median':>10}{'p90':>10}{'share':>9}")
    print("-" * 51)
    total_median = statistics.median(totals)
    for key, label in PHASES:
        values = [t.get(key, 0.0) for t in timings]
        median = statistics.median(values)
        print(f"{label:<22}{median:>9.1f}{percentile(values, 0.9):>10.1f}"
              f"{100 * median / max(total_median, 1e-9):>8.0f}%")
    print("-" * 51)
    print(f"{'total':<22}{total_median:>9.1f}{percentile(totals, 0.9):>10.1f}\n")

    worst = max(totals)
    print(f"real-time    : {budget_ms / total_median:.1f}x faster than real time (median), "
          f"{budget_ms / percentile(totals, 0.9):.1f}x at p90")
    print(f"               worst frame {worst:.0f} ms "
          f"{'fits' if worst < budget_ms else 'OVERRUNS'} the {budget_ms:.0f} ms budget")
    print(f"               sustained throughput {1000.0 / total_median:.1f} frames/s\n")

    # Cost against gate size: the claim is that the gate, not the map, sets the bill.
    buckets: dict[int, list[float]] = {}
    for n, total in zip(searched, totals):
        buckets.setdefault(round(n / 20) * 20, []).append(total)
    if len(buckets) > 1:
        print(f"{'views searched':<22}{'frames':>8}{'median ms':>12}{'ms per view':>14}")
        print("-" * 56)
        for n in sorted(buckets):
            values = buckets[n]
            median = statistics.median(values)
            per = median / n if n else float("nan")
            print(f"{n:<22}{len(values):>8}{median:>12.1f}{per:>14.2f}")
        print("\nCost tracks views *searched*, not the size of the map -- which is what")
        print("the motion gate buys: the map can grow without the per-frame bill growing.")


if __name__ == "__main__":
    main()
