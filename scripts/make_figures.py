"""Figures and per-slice numbers used in the README."""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ctedges import CannyParams, process_sequence  # noqa: E402
from ctedges.canny import canny  # noqa: E402
from ctedges.io import list_slices, read_sequence  # noqa: E402
from ctedges.metrics import components, endpoint_count, slice_agreement  # noqa: E402
from ctedges.viz import overlay, show_sequences  # noqa: E402

INK = "#333333"
MUTED = "#888888"
BLUES = ["#b7d3f6", "#86b6ef", "#3987e5", "#256abf", "#104281"]
ADDED = (0.92, 0.41, 0.20)   # pixels added by a linking step


def pipeline_figure(images, edges, linked2d, linked3d, i, path):
    _, steps = canny(images[i], CannyParams(), return_steps=True)
    panels = [
        ("CT slice", images[i], "gray"),
        ("gradient magnitude", steps["magnitude"], "magma"),
        ("Canny edges", edges[i], "gray"),
        ("linked in slice", overlay(edges[i], linked2d[i], ADDED), None),
        ("linked to previous slice", overlay(linked2d[i], linked3d[i], ADDED), None),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10, color=INK)
        ax.axis("off")
    fig.text(0.5, 0.02, "orange = pixels added by the linking step", ha="center",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def zoom_figure(edges, linked2d, linked3d, i, box, path):
    y0, y1, x0, x1 = box
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(overlay(edges[i], linked2d[i], ADDED)[y0:y1, x0:x1], interpolation="nearest")
    axes[0].set_title("gaps closed inside the slice", fontsize=10, color=INK)
    axes[1].imshow(overlay(linked2d[i], linked3d[i], ADDED)[y0:y1, x0:x1], interpolation="nearest")
    axes[1].set_title("links from the previous slice (24-connectivity)", fontsize=10, color=INK)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def sweep_figure(csv_path, chosen, path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    lows = sorted({float(r["low"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for metric, ax, label in [("density_pct", axes[0], "edge pixels (% of image)"),
                              ("components", axes[1], "separate edge fragments per slice")]:
        for lo, colour in zip(lows, BLUES):
            rs = [r for r in rows if float(r["low"]) == lo]
            ax.plot([float(r["high"]) for r in rs], [float(r[metric]) for r in rs],
                    color=colour, lw=2, marker="o", ms=5, label=f"low = {lo:g}")
            sel = [r for r in rs if abs(float(r["high"]) - chosen[1]) < 1e-9 and lo == chosen[0]]
            for r in sel:
                ax.plot(float(r["high"]), float(r[metric]), "o", ms=11, mfc="none",
                        mec=INK, mew=1.5)
        ax.set_xlabel("high threshold", color=INK)
        ax.set_title(label, fontsize=10, color=INK)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)
    fig.text(0.5, 0.01, "circle = thresholds used (low 0.15, high 0.375)", ha="center",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def per_slice_stats(names, edges, linked2d, linked3d, path):
    rows = []
    for k, name in enumerate(names):
        rows.append({
            "slice": name,
            "canny_px": int(edges[k].sum()),
            "canny_fragments": components(edges[k])[0],
            "canny_open_ends": endpoint_count(edges[k]),
            "linked2d_px": int(linked2d[k].sum()),
            "linked2d_fragments": components(linked2d[k])[0],
            "linked2d_open_ends": endpoint_count(linked2d[k]),
            "linked3d_px": int(linked3d[k].sum()),
            "added_by_3d_px": int(((linked3d[k] > 0) & ~(linked2d[k] > 0)).sum()),
            "agree_with_prev_1px": round(slice_agreement(linked2d[k], linked2d[k - 1], 1), 3) if k else "",
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--slice", type=int, default=6)
    args = ap.parse_args()

    paths = list_slices(args.folder)
    names = [os.path.basename(p) for p in paths]
    images = read_sequence(paths)
    edges, linked2d, linked3d = process_sequence(images)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    pipeline_figure(images, edges, linked2d, linked3d, args.slice, "figures/pipeline.png")
    zoom_figure(edges, linked2d, linked3d, args.slice, (300, 440, 160, 360), "figures/linking_zoom.png")
    step = max(1, len(images) // 8)
    idx = list(range(0, len(images), step))[:8]
    show_sequences([images[k] for k in idx], [linked2d[k] for k in idx], [linked3d[k] for k in idx],
                   [names[k] for k in idx], save="figures/sequences.png", show=False)
    if os.path.exists("results/threshold_sweep.csv"):
        p = CannyParams()
        sweep_figure("results/threshold_sweep.csv", (p.t_low, p.t_high), "figures/threshold_sweep.png")

    rows = per_slice_stats(names, edges, linked2d, linked3d, "results/per_slice.csv")
    mean = lambda key: np.mean([r[key] for r in rows])  # noqa: E731
    print(f"fragments  {mean('canny_fragments'):.1f} -> {mean('linked2d_fragments'):.1f}")
    print(f"open ends  {mean('canny_open_ends'):.1f} -> {mean('linked2d_open_ends'):.1f}")
    print(f"edge px    {mean('canny_px'):.0f} -> {mean('linked2d_px'):.0f} -> {mean('linked3d_px'):.0f}")
    agree = [r["agree_with_prev_1px"] for r in rows[1:]]
    print(f"agreement with previous slice (1 px): {np.mean(agree):.3f}  min {np.min(agree):.3f}")


if __name__ == "__main__":
    main()
