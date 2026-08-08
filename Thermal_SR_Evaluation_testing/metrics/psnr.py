"""
metrics/psnr.py
================
Contains only `calculate_psnr()`.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _sk_psnr


def calculate_psnr(gt_gray: np.ndarray, pred_gray: np.ndarray) -> float:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two grayscale images.

    Parameters
    ----------
    gt_gray : np.ndarray
        Ground-truth grayscale image, uint8.
    pred_gray : np.ndarray
        Predicted grayscale image, uint8, same shape as gt_gray.

    Returns
    -------
    float
        PSNR value in decibels (dB). Returns NaN if the metric could not
        be computed (e.g. mismatched shapes), so that a single failed
        metric never interrupts the overall evaluation.
    """
    try:
        return float(_sk_psnr(gt_gray, pred_gray, data_range=255))
    except Exception:
        return float("nan")
