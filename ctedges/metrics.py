"""Simple measures for edge maps (there is no ground truth in CTMRI DB)."""

import cv2
import numpy as np


def edge_density(edges):
    return float((edges > 0).mean())


def components(edges):
    n, _, stats, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    return n - 1, sizes


def endpoint_count(edges):
    from .linking2d import endpoints
    return len(endpoints((edges > 0).astype(np.uint8)))


def slice_agreement(a, b, radius=1):
    """Share of edge pixels in a that have an edge in b within a (2r+1)^2 window."""
    a = a > 0
    if not a.any():
        return float("nan")
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    near = cv2.dilate((b > 0).astype(np.uint8), k) > 0
    return float((a & near).sum() / a.sum())


def summarize(seq):
    rows = []
    for i, e in enumerate(seq):
        n, sizes = components(e)
        rows.append({
            "slice": i,
            "edge_px": int((e > 0).sum()),
            "density": edge_density(e),
            "components": n,
            "mean_component": float(sizes.mean()) if n else 0.0,
            "endpoints": endpoint_count(e),
        })
    return rows
