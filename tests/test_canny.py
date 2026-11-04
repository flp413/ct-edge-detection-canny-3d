import cv2
import numpy as np

from ctedges.canny import CannyParams, canny, hysteresis, non_max_suppression


def test_hysteresis_keeps_weak_pixels_touching_strong_ones():
    mag = np.zeros((5, 9))
    mag[2, 1:8] = 0.2      # weak line
    mag[2, 4] = 1.0        # one strong pixel in the middle
    mag[0, 0] = 0.2        # weak and alone
    out = hysteresis(mag, 0.1, 0.5)
    assert out[2, 1:8].all()
    assert out[0, 0] == 0


def test_nms_diagonal_uses_pixels_along_the_gradient():
    # gradient pointing down-right (45 deg with y going down)
    mag = np.zeros((3, 3))
    mag[1, 1] = 1.0
    mag[0, 0] = mag[2, 2] = 2.0   # bigger neighbours along the gradient
    angle = np.full((3, 3), np.deg2rad(45))
    assert non_max_suppression(mag, angle)[1, 1] == 0

    mag[0, 0] = mag[2, 2] = 0.5   # now the centre is the maximum
    mag[0, 2] = mag[2, 0] = 2.0   # across the edge, should not matter
    assert non_max_suppression(mag, angle)[1, 1] == 1.0


def test_matches_opencv_canny():
    rng = np.random.default_rng(0)
    img = np.zeros((128, 128), np.float32)
    cv2.circle(img, (64, 64), 35, 0.8, -1)
    cv2.rectangle(img, (10, 10), (40, 50), 0.5, -1)
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)

    p = CannyParams(sigma=1.4, t_low=0.1, t_high=0.3)
    ours, steps = canny(img, p, return_steps=True)

    s = steps["smoothed"] * 1000
    dx = cv2.Sobel(s, cv2.CV_16S, 1, 0, ksize=3)
    dy = cv2.Sobel(s, cv2.CV_16S, 0, 1, ksize=3)
    ref = cv2.Canny(dx, dy, p.t_low * 1000, p.t_high * 1000, L2gradient=True) > 0

    ours = ours > 0
    near_ref = cv2.dilate(ref.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    near_ours = cv2.dilate(ours.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert (ours & near_ref).sum() / ours.sum() > 0.98
    assert (ref & near_ours).sum() / ref.sum() > 0.98
