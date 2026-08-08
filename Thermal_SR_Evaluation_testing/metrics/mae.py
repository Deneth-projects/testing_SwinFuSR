"""
metrics/mae.py
===============
Contains only `calculate_mae()`.
"""

import numpy as np


def calculate_mae(gt_gray: np.ndarray, pred_gray: np.ndarray) -> float:
    """Compute the Mean Absolute Error (MAE) between two grayscale images.

    Parameters
    ----------
    gt_gray : np.ndarray
        Ground-truth grayscale image.
    pred_gray : np.ndarray
        Predicted grayscale image, same shape as gt_gray.

    Returns
    -------
    float
        Mean absolute pixel-intensity difference.
    """
    gt_f = gt_gray.astype(np.float64)
    pred_f = pred_gray.astype(np.float64)
    return float(np.mean(np.abs(gt_f - pred_f)))
