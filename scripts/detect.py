"""Detect edges in consecutive CT slices and link them between slices.

    python scripts/detect.py data/233x694 0001-233x694.png 0002-233x694.png ...
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ctedges.canny import canny  # noqa: E402
from ctedges.io import read_sequence, save_binary  # noqa: E402
from ctedges.linking3d import link_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    paths = [os.path.join(args.folder, n) for n in args.images]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        ap.error(f"not found: {', '.join(missing)}")

    images = read_sequence(paths)
    edges = [canny(img) for img in images]
    linked = link_sequence(edges)
    save_binary(linked, paths, args.out)
    print(f"{len(paths)} slices -> {args.out}/")


if __name__ == "__main__":
    main()
