import os
from pathlib import Path

import cv2
import numpy as np


def read_slice(path):
    """Read one CT slice as float32 in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read {path}")
    return img.astype(np.float32) / 255.0


def read_sequence(paths):
    return [read_slice(p) for p in paths]


def list_slices(folder, pattern="*.png"):
    """Sorted list of slice files in a folder (CTMRI DB names sort correctly)."""
    return sorted(str(p) for p in Path(folder).glob(pattern))


def save_binary(images, names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for img, name in zip(images, names):
        cv2.imwrite(os.path.join(out_dir, os.path.basename(name)), (img > 0).astype(np.uint8) * 255)
