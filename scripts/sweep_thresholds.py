"""Try a grid of (low, high) thresholds and record how the edge maps change.

There is no ground truth, so the sweep looks at things that are easy to judge:
how many pixels are edges, how many separate fragments there are, and how
many open ends are left after linking inside the slice.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ctedges import CannyParams, canny, link_edges_2d  # noqa: E402
from ctedges.io import list_slices, read_sequence  # noqa: E402
from ctedges.metrics import components, edge_density, endpoint_count  # noqa: E402

LOWS = [0.05, 0.10, 0.15, 0.20, 0.30]
RATIOS = [1.5, 2.0, 2.5, 3.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="results/threshold_sweep.csv")
    args = ap.parse_args()

    images = read_sequence(list_slices(args.folder))
    rows = []
    for lo in LOWS:
        for r in RATIOS:
            hi = round(lo * r, 3)
            p = CannyParams(t_low=lo, t_high=hi)
            dens, comp, ends = [], [], []
            for img in images:
                e = link_edges_2d(canny(img, p))
                dens.append(edge_density(e))
                comp.append(components(e)[0])
                ends.append(endpoint_count(e))
            rows.append({"low": lo, "high": hi, "ratio": r,
                         "density_pct": round(float(100 * np.mean(dens)), 3),
                         "components": round(float(np.mean(comp)), 1),
                         "open_ends": round(float(np.mean(ends)), 1)})
            print(rows[-1])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
