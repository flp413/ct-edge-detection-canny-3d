"""Canny edge detector for CT slices."""

import cv2
import numpy as np
from skimage import morphology

DEFAULT_CONFIG = {
    "gaussian_sigma": None,
    "t_low_factor": 0.6,
    "t_high_factor": 1.2,
    "morphology_cleanup": True,
}


def smooth(img, sigma=None):
    sigma = sigma or max(min(img.shape) * 0.003, 0.5)
    k = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def gradients(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def non_max_suppression(mag, angle):
    deg = np.rad2deg(angle) % 180
    p = np.pad(mag, 1)
    c = p[1:-1, 1:-1]
    left, right = p[1:-1, :-2], p[1:-1, 2:]
    up, down = p[:-2, 1:-1], p[2:, 1:-1]
    up_left, up_right = p[:-2, :-2], p[:-2, 2:]
    down_left, down_right = p[2:, :-2], p[2:, 2:]

    horiz = (deg < 22.5) | (deg >= 157.5)
    diag45 = (deg >= 22.5) & (deg < 67.5)
    vert = (deg >= 67.5) & (deg < 112.5)
    diag135 = (deg >= 112.5) & (deg < 157.5)

    n1 = np.select([horiz, diag45, vert, diag135], [left, up_right, up, up_left])
    n2 = np.select([horiz, diag45, vert, diag135], [right, down_left, down, down_right])
    keep = (c >= n1) & (c >= n2)
    return np.where(keep, mag, 0.0)


def adaptive_thresholds(mag, config):
    nz = mag[mag > 0]
    if len(nz) == 0:
        return 0.05, 0.15
    t_high = np.percentile(nz, 70) * config["t_high_factor"]
    t_low = np.percentile(nz, 30) * config["t_low_factor"]
    return t_low, t_high


def hysteresis(mag, t_low, t_high):
    if mag.max() > 0:
        mag = mag / mag.max()
    strong = mag >= t_high
    weak = (mag >= t_low) & (mag < t_high)
    edges = np.zeros(mag.shape, dtype=np.uint8)
    edges[strong] = 1
    _, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    for lab in np.unique(labels[strong]):
        if lab > 0:
            edges[labels == lab] = 1
    return edges


def cleanup(edges):
    cleaned = morphology.remove_small_objects(edges.astype(bool), max_size=2, connectivity=2)
    return morphology.closing(cleaned.astype(np.uint8), morphology.disk(1)).astype(np.uint8)


def canny(img, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    s = smooth(img, config["gaussian_sigma"])
    mag, ang = gradients(s)
    nms = non_max_suppression(mag, ang)
    t_low, t_high = adaptive_thresholds(nms, config)
    edges = hysteresis(nms, t_low, t_high)
    if config["morphology_cleanup"]:
        edges = cleanup(edges)
    return edges
