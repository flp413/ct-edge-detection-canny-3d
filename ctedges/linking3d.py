"""Edge linking between consecutive slices with 24-connectivity.

For every edge pixel p(x, y) in slice n:
  1. if slice n+1 has an edge at (x, y)            -> nothing to do
  2. if it has one in the 3x3 neighbourhood         -> nothing to do
  3. if the 5x5 neighbourhood in n+1 is empty       -> nothing to do
  4. otherwise every edge pixel (x+a, y+b) of the 5x5 neighbourhood in n+1
     is connected to (x, y) by walking a and b towards 0 one step at a time
     and switching on every visited pixel of slice n+1.
"""

import cv2
import numpy as np


def _shortest_path(a, b):
    """Offsets visited when (a, b) converges to (0, 0), both ends included."""
    path = [(a, b)]
    while a != 0 or b != 0:
        a -= int(np.sign(a))
        b -= int(np.sign(b))
        path.append((a, b))
    return path


def link_consecutive(current, nxt):
    """Return slice n+1 with links to slice n added."""
    cur = current > 0
    nb = (nxt > 0).astype(np.uint8)
    has3 = cv2.dilate(nb, np.ones((3, 3), np.uint8)) > 0
    has5 = cv2.dilate(nb, np.ones((5, 5), np.uint8)) > 0
    # steps 1-3 in one go: edge in n, nothing in the 3x3 of n+1, something in the 5x5
    todo = np.argwhere(cur & ~has3 & has5)

    h, w = nb.shape
    out = nb.copy()
    for y, x in todo:
        for b in range(-2, 3):
            for a in range(-2, 3):
                yy, xx = y + b, x + a
                if not (0 <= yy < h and 0 <= xx < w) or not nb[yy, xx]:
                    continue
                for da, db in _shortest_path(a, b):
                    out[y + db, x + da] = 1
    return out


def link_sequence(edge_images):
    """Apply the 24-connectivity linking to every consecutive pair.

    The first slice has no predecessor and is returned as is. Links into slice
    n+1 are made from slice n as it was before inter-slice linking, so pixels
    added in one step don't keep pulling edges further through the stack.
    """
    if not edge_images:
        return []
    linked = [edge_images[0].astype(np.uint8).copy()]
    for prev, nxt in zip(edge_images[:-1], edge_images[1:]):
        linked.append(link_consecutive(prev, nxt))
    return linked
