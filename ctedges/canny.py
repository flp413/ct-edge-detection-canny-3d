"""Canny edge detector written out step by step.

Coordinates: arrays are indexed [row, col] = [y, x] and y grows downwards,
which matters for the diagonal cases in non-maximum suppression.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CannyParams:
    sigma: float = 1.4
    # Thresholds are absolute values of the Sobel gradient magnitude computed
    # on an image scaled to [0, 1], so they mean the same thing on every slice.
    t_low: float = 0.10
    t_high: float = 0.25


def smooth(img, sigma):
    if sigma <= 0:
        return img
    k = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def gradients(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def non_max_suppression(mag, angle):
    """Keep pixels that are a local maximum along the gradient direction."""
    deg = np.rad2deg(angle) % 180
    p = np.pad(mag, 1)
    c = p[1:-1, 1:-1]
    left, right = p[1:-1, :-2], p[1:-1, 2:]
    up, down = p[:-2, 1:-1], p[2:, 1:-1]
    up_left, up_right = p[:-2, :-2], p[:-2, 2:]
    down_left, down_right = p[2:, :-2], p[2:, 2:]

    horiz = (deg < 22.5) | (deg >= 157.5)
    diag45 = (deg >= 22.5) & (deg < 67.5)     # gradient towards +x, +y (down-right)
    vert = (deg >= 67.5) & (deg < 112.5)
    diag135 = (deg >= 112.5) & (deg < 157.5)  # gradient towards -x, +y (down-left)

    n1 = np.select([horiz, diag45, vert, diag135], [left, up_left, up, up_right])
    n2 = np.select([horiz, diag45, vert, diag135], [right, down_right, down, down_left])
    keep = (c >= n1) & (c >= n2) & (c > 0)
    return np.where(keep, mag, 0.0)


def hysteresis(mag, t_low, t_high):
    """Strong pixels plus every weak pixel 8-connected to a strong one."""
    candidates = (mag >= t_low).astype(np.uint8)
    n, labels = cv2.connectedComponents(candidates, connectivity=8)
    strong_labels = np.unique(labels[mag >= t_high])
    keep = np.zeros(n, dtype=bool)
    keep[strong_labels[strong_labels > 0]] = True
    return keep[labels].astype(np.uint8)


def canny(img, params=None, return_steps=False):
    params = params or CannyParams()
    s = smooth(img, params.sigma)
    mag, ang = gradients(s)
    nms = non_max_suppression(mag, ang)
    edges = hysteresis(nms, params.t_low, params.t_high)
    if return_steps:
        return edges, {"smoothed": s, "magnitude": mag, "nms": nms}
    return edges
