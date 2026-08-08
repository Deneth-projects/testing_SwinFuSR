"""
metrics/ssim.py
================
Contains only `calculate_ssim()`.
"""

from typing import Optional

import numpy as np
from skimage.metrics import structural_similarity as _sk_ssim


def calculate_ssim(gt_gray: np.ndarray, pred_gray: np.ndarray, win_size: Optional[int] = None) -> float:
    """Compute the Structural Similarity Index (SSIM) between two grayscale images.

    Parameters
    ----------
    gt_gray : np.ndarray
        Ground-truth grayscale image, uint8.
    pred_gray : np.ndarray
        Predicted grayscale image, uint8, same shape as gt_gray.
    win_size : int, optional
        Sliding window size for SSIM. Defaults to scikit-image's default
        (7) when None. A smaller odd value must be supplied for very thin
        images/slices where the default window does not fit.

    Returns
    -------
    float
        SSIM value in [-1, 1] (typically [0, 1] for natural images).
        Returns NaN if the metric could not be computed, so that a single
        failed metric never interrupts the overall evaluation.
    """
    try:
        if win_size is not None:
            return float(_sk_ssim(gt_gray, pred_gray, data_range=255, win_size=win_size))
        return float(_sk_ssim(gt_gray, pred_gray, data_range=255))
    except Exception:
        return float("nan")
