"""
metrics/max_error.py
=====================
Contains only `calculate_max_error()`.
"""

import numpy as np


def calculate_max_error(gt_gray: np.ndarray, pred_gray: np.ndarray) -> float:
    """Compute the Maximum Absolute Error between two grayscale images.

    Parameters
    ----------
    gt_gray : np.ndarray
        Ground-truth grayscale image.
    pred_gray : np.ndarray
        Predicted grayscale image, same shape as gt_gray.

    Returns
    -------
    float
        Largest single-pixel absolute intensity difference.
    """
    gt_f = gt_gray.astype(np.float64)
    pred_f = pred_gray.astype(np.float64)
    return float(np.max(np.abs(gt_f - pred_f)))
