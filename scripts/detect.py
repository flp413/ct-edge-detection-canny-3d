"""Detect and link edges in a sequence of consecutive CT slices.

    python scripts/detect.py data/233x694 0001-233x694.png 0002-233x694.png ...
    python scripts/detect.py data/233x694 --all --show

Writes the final binary images (after linking between slices) as .png files
with the same names into --out.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ctedges import CannyParams, process_sequence  # noqa: E402
from ctedges.io import list_slices, read_sequence, save_binary  # noqa: E402
from ctedges.viz import show_sequences  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder with the slices of one patient / body part")
    ap.add_argument("images", nargs="*", help="consecutive slice file names, in order")
    ap.add_argument("--all", action="store_true", help="use every .png in the folder")
    ap.add_argument("--out", default="output")
    ap.add_argument("--low", type=float, default=CannyParams.t_low)
    ap.add_argument("--high", type=float, default=CannyParams.t_high)
    ap.add_argument("--sigma", type=float, default=CannyParams.sigma)
    ap.add_argument("--show", action="store_true", help="display the three sequences")
    ap.add_argument("--save-figure", help="also save the sequence figure to this file")
    args = ap.parse_args()

    if args.all:
        paths = list_slices(args.folder)
    else:
        paths = [os.path.join(args.folder, n) for n in args.images]
    if len(paths) < 2:
        ap.error("need at least two consecutive slices")
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        ap.error(f"not found: {', '.join(missing)}")

    images = read_sequence(paths)
    params = CannyParams(sigma=args.sigma, t_low=args.low, t_high=args.high)
    _, linked2d, linked3d = process_sequence(images, params)
    save_binary(linked3d, paths, args.out)
    print(f"{len(paths)} slices -> {args.out}/")

    if args.show or args.save_figure:
        names = [os.path.basename(p) for p in paths]
        show_sequences(images, linked2d, linked3d, names, max_cols=len(paths),
                       save=args.save_figure, show=args.show)


if __name__ == "__main__":
    main()
