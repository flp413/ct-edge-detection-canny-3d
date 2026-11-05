"""Edge linking inside a single slice.

Works only on the binary edge map (no gradient magnitude or angle needed).
Every open end of an edge is extended towards the nearest edge pixel that
lies roughly in front of it, as long as the gap is short. Afterwards tiny
leftover fragments are removed.
"""

import cv2
import numpy as np
from skimage.draw import line
from skimage.morphology import remove_small_objects, skeletonize

_KERNEL8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)


def endpoints(edges):
    """Edge pixels with at most one 8-neighbour."""
    nbrs = cv2.filter2D(edges.astype(np.float32), -1, _KERNEL8, borderType=cv2.BORDER_CONSTANT)
    return np.argwhere((edges > 0) & (nbrs <= 1))


def _outward_direction(edges, y, x, reach=3):
    """Unit vector pointing out of the edge at an endpoint."""
    y0, y1 = max(0, y - reach), y + reach + 1
    x0, x1 = max(0, x - reach), x + reach + 1
    pts = np.argwhere(edges[y0:y1, x0:x1] > 0) + [y0, x0]
    if len(pts) <= 1:
        return None  # isolated pixel, any direction is fine
    v = np.array([y, x], dtype=float) - pts.mean(axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else None


def link_edges_2d(edges, max_gap=6, min_cos=0.5, min_size=10):
    """Close short gaps between edge fragments in one binary edge image.

    max_gap   longest gap (in pixels) that may be bridged
    min_cos   target has to be within ~60 degrees of the edge direction
    min_size  components smaller than this are dropped at the end
    """
    edges = skeletonize(edges > 0).astype(np.uint8)
    _, labels = cv2.connectedComponents(edges, connectivity=8)
    h, w = edges.shape
    out = edges.copy()
    ends = endpoints(edges)
    end_set = {(int(a), int(b)) for a, b in ends}

    for y, x in ends:
        y, x = int(y), int(x)
        y0, y1 = max(0, y - max_gap), min(h, y + max_gap + 1)
        x0, x1 = max(0, x - max_gap), min(w, x + max_gap + 1)
        cand = np.argwhere(edges[y0:y1, x0:x1] > 0) + [y0, x0]
        if len(cand) == 0:
            continue
        lab = labels[y, x]
        same = labels[cand[:, 0], cand[:, 1]] == lab
        # same fragment only counts if it is another open end (closing a contour)
        other_end = np.array([(int(a), int(b)) in end_set and (a, b) != (y, x) for a, b in cand])
        ok = ~same | other_end
        d = cand - [y, x]
        dist = np.hypot(d[:, 0], d[:, 1])
        ok &= (dist > 1.5) & (dist <= max_gap)
        direction = _outward_direction(edges, y, x)
        if direction is not None:
            cos = (d @ direction) / np.maximum(dist, 1e-9)
            ok &= cos >= min_cos
        if not ok.any():
            continue
        ty, tx = cand[ok][np.argmin(dist[ok])]
        rr, cc = line(y, x, int(ty), int(tx))
        out[rr, cc] = 1

    if min_size > 1:
        out = remove_small_objects(out.astype(bool), max_size=min_size - 1, connectivity=2).astype(np.uint8)
    return out
