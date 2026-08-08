"""
Loads thermal images and puts them on a consistent [0, 1] scale.

Rule: uint8 -> divide by 255. float -> used as-is (clipped to [0, 1]).
Never independently min-max stretch GT and prediction before comparing -
that would rescale each with a different window and break the relationship
between them.
"""
import os

import cv2
import numpy as np


class ImageLoadError(Exception):
    pass


def load_raw(path):
    if not path or not os.path.isfile(path):
        raise ImageLoadError(f"File not found: {path}")
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ImageLoadError(f"Could not decode: {path}")
    return image


def to_grayscale(image):
    """Collapse to a single channel. If R/G/B are identical (grayscale
    saved as RGB), this is lossless; otherwise falls back to luma."""
    if image.ndim == 2:
        return image

    if image.ndim == 3:
        if image.shape[2] == 1:
            return image[:, :, 0]
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        if np.array_equal(b, g) and np.array_equal(g, r):
            return b
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

    raise ImageLoadError(f"Unsupported image shape: {image.shape}")


def normalize_to_unit_range(image):
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(image.astype(np.float32), 0.0, 1.0)
    raise ImageLoadError(f"Unsupported dtype: {image.dtype}")


def load_thermal_image(path):
    """Returns a float32 (H, W) array normalized to [0, 1]."""
    return normalize_to_unit_range(to_grayscale(load_raw(path)))


def resize_to_match(image, target_shape):
    target_h, target_w = target_shape
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image
    downscaling = image.shape[0] > target_h or image.shape[1] > target_w
    interp = cv2.INTER_AREA if downscaling else cv2.INTER_CUBIC
    return cv2.resize(image, (target_w, target_h), interpolation=interp)
