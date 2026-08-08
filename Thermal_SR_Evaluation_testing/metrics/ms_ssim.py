"""
metrics/ms_ssim.py
===================
Contains only `calculate_ms_ssim()`.
"""

import numpy as np
import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

import config
from helpers import grayscale_to_torch_tensor


def calculate_ms_ssim(gt_gray: np.ndarray, pred_gray: np.ndarray) -> float:
    """Compute the Multi-Scale SSIM (MS-SSIM) between two grayscale images.

    MS-SSIM requires a minimum spatial resolution (a function of the number
    of scales and the Gaussian kernel size). If the input images are too
    small for MS-SSIM to be computed, NaN is returned instead of raising,
    so that the overall evaluation is never interrupted.

    Parameters
    ----------
    gt_gray : np.ndarray
        Ground-truth grayscale image, uint8.
    pred_gray : np.ndarray
        Predicted grayscale image, uint8, same shape as gt_gray.

    Returns
    -------
    float
        MS-SSIM score in [0, 1], or NaN if it could not be computed.
    """
    try:
        gt_tensor = grayscale_to_torch_tensor(gt_gray, config.DEVICE)
        pred_tensor = grayscale_to_torch_tensor(pred_gray, config.DEVICE)

        metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(config.DEVICE)
        with torch.no_grad():
            score = metric(pred_tensor, gt_tensor)
        return float(score.item())
    except Exception:
        return float("nan")
