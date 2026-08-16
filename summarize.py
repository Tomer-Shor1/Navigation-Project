"""Print a one-screen summary table of every error_report.txt under results/.

    python summarize.py

Reads only the committed reports, so it is instant and needs no video, no
telemetry and no OpenCV -- the fastest way for a reviewer to see the headline
numbers.
"""

from __future__ import annotations

import glob
import os
import re
import sys

HEADER_RE = re.compile(r"^\s*Test frames total:\s*(\d+)", re.M)
MAP_RE = re.compile(r"^\s*Trusted map fixes:\s*(\d+)", re.M)
DR_RE = re.compile(r"^\s*Dead-reckoned \(coasted\):\s*(\d+)", re.M)
STAT_RE = re.compile(
    r"^\s*(map fixes|dead-reckon|all reported)\s*\((\d+)\):\s*"
    r"(?:none|median\s+([\d.]+)\s+mean\s+([\d.]+)\s+rmse\s+([\d.]+)\s+p90\s+([\d.]+)\s+max\s+([\d.]+))",
    re.M,
)


def summarize(path: str) -> dict:
    text = open(path).read()
    if text.lstrip().startswith("STALE"):
        return {"flight": os.path.basename(os.path.dirname(path)), "stale": True}
    row = {
        "flight": os.path.basename(os.path.dirname(path)),
        "n": int(HEADER_RE.search(text).group(1)) if HEADER_RE.search(text) else 0,
        "map": int(MAP_RE.search(text).group(1)) if MAP_RE.search(text) else 0,
        "dr": int(DR_RE.search(text).group(1)) if DR_RE.search(text) else 0,
    }
    for m in STAT_RE.finditer(text):
        if m.group(1) == "map fixes" and m.group(3):
            row.update(median=float(m.group(3)), mean=float(m.group(4)),
                       rmse=float(m.group(5)), p90=float(m.group(6)), max=float(m.group(7)))
    return row


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    reports = sorted(glob.glob(os.path.join(root, "*", "error_report.txt")))
    if not reports:
        print(f"No error_report.txt found under {root}/ -- run ./reproduce.sh first.")
        sys.exit(1)

    print(f"Localization accuracy vs. SRT ground truth  ({root}/)")
    print(f"{'flight':<14}{'frames':>7}{'map fix':>9}{'coasted':>9}"
          f"{'median':>9}{'mean':>8}{'rmse':>8}{'p90':>8}{'max':>8}   (metres)")
    print("-" * 88)
    for path in reports:
        r = summarize(path)
        if r.get("stale"):
            print(f"{r['flight']:<14}{'  stale -- run ./reproduce.sh to regenerate':>50}")
        elif "median" in r:
            print(f"{r['flight']:<14}{r['n']:>7}{r['map']:>9}{r['dr']:>9}"
                  f"{r['median']:>9.1f}{r['mean']:>8.1f}{r['rmse']:>8.1f}{r['p90']:>8.1f}{r['max']:>8.1f}")
        else:
            print(f"{r['flight']:<14}{r['n']:>7}{r['map']:>9}{r['dr']:>9}{'  no trusted map fixes':>41}")
    print("-" * 88)
    print("median/mean/rmse/p90/max are over the trusted visual map fixes only;")
    print("see each results/<flight>/error_report.txt for the per-frame breakdown.")


if __name__ == "__main__":
    main()
