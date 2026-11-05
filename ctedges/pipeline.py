from .canny import CannyParams, canny
from .linking2d import link_edges_2d
from .linking3d import link_sequence


def process_sequence(images, params=None, link2d=None):
    """Canny -> linking inside each slice -> linking between slices."""
    params = params or CannyParams()
    link2d = link2d or {}
    edges = [canny(img, params) for img in images]
    linked2d = [link_edges_2d(e, **link2d) for e in edges]
    linked3d = link_sequence(linked2d)
    return edges, linked2d, linked3d
