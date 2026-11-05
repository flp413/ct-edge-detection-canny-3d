import numpy as np

from ctedges.linking2d import endpoints, link_edges_2d
from ctedges.linking3d import _shortest_path, link_consecutive, link_sequence


def test_shortest_path_example_from_the_assignment():
    # (x+2, y-1) -> (x+1, y) -> (x, y)
    assert _shortest_path(2, -1) == [(2, -1), (1, 0), (0, 0)]


def test_no_link_when_slice_below_already_has_edge_nearby():
    a = np.zeros((9, 9), np.uint8); a[4, 4] = 1
    b = np.zeros((9, 9), np.uint8); b[5, 5] = 1     # inside the 3x3
    assert (link_consecutive(a, b) == b).all()


def test_no_link_when_5x5_is_empty():
    a = np.zeros((9, 9), np.uint8); a[4, 4] = 1
    b = np.zeros((9, 9), np.uint8); b[0, 0] = 1
    assert (link_consecutive(a, b) == b).all()


def test_link_draws_path_to_centre():
    a = np.zeros((9, 9), np.uint8); a[4, 4] = 1     # (x, y) = (4, 4)
    b = np.zeros((9, 9), np.uint8); b[3, 6] = 1     # (x+2, y-1)
    out = link_consecutive(a, b)
    expected = b.copy(); expected[4, 5] = 1; expected[4, 4] = 1
    assert (out == expected).all()


def test_link_sequence_keeps_first_slice():
    s = [np.eye(6, dtype=np.uint8), np.zeros((6, 6), np.uint8)]
    out = link_sequence(s)
    assert len(out) == 2 and (out[0] == s[0]).all()


def test_2d_linking_bridges_small_gap():
    e = np.zeros((20, 30), np.uint8)
    e[10, 2:13] = 1
    e[10, 16:28] = 1          # 3 pixel gap in a straight line
    out = link_edges_2d(e, max_gap=6, min_size=1)
    assert out[10, 2:28].all()
    assert len(endpoints(out)) == 2


def test_2d_linking_leaves_big_gap_alone():
    e = np.zeros((20, 40), np.uint8)
    e[10, 2:12] = 1
    e[10, 25:38] = 1
    out = link_edges_2d(e, max_gap=6, min_size=1)
    assert not out[10, 12:25].any()
