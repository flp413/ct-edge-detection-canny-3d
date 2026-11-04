"""Edge linking between consecutive slices (24-connectivity)."""

import numpy as np


def _draw_path(image, x1, y1, x2, y2):
    cx, cy = x1, y1
    while cx != x2 or cy != y2:
        image[cy, cx] = 1
        cx += int(np.sign(x2 - cx))
        cy += int(np.sign(y2 - cy))
    image[cy, cx] = 1


def link_consecutive(current, nxt):
    padded = np.pad(nxt, 2)
    for y, x in np.argwhere(current == 1):
        py, px = y + 2, x + 2
        if padded[py, px] == 1:
            continue
        if np.any(padded[py - 1:py + 2, px - 1:px + 2] == 1):
            continue
        hood = padded[py - 2:py + 3, px - 2:px + 3]
        if hood.sum() == 0:
            continue
        for ly, lx in np.argwhere(hood == 1):
            _draw_path(padded, px, py, px - 2 + lx, py - 2 + ly)
    return padded[2:-2, 2:-2]


def link_sequence(edge_images):
    if len(edge_images) < 2:
        return list(edge_images)
    linked = [edge_images[0].copy()]
    for prev, nxt in zip(edge_images[:-1], edge_images[1:]):
        linked.append(link_consecutive(prev, nxt.copy()))
    return linked
