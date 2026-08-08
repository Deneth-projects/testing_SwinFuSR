"""
metrics/rmse.py
================
Contains only `calculate_rmse()`.
"""

import numpy as np


def calculate_rmse(gt_gray: np.ndarray, pred_gray: np.ndarray) -> float:
    """Compute the Root Mean Squared Error (RMSE) between two grayscale images.

    Parameters
    ----------
    gt_gray : np.ndarray
        Ground-truth grayscale image.
    pred_gray : np.ndarray
        Predicted grayscale image, same shape as gt_gray.

    Returns
    -------
    float
        Root mean squared pixel-intensity difference.
    """
    gt_f = gt_gray.astype(np.float64)
    pred_f = pred_gray.astype(np.float64)
    return float(np.sqrt(np.mean((gt_f - pred_f) ** 2)))
