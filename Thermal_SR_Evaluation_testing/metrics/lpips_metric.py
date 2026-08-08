"""
metrics/lpips_metric.py
========================
Contains only `calculate_lpips()` (plus the private lazy-loading cache it
needs for its pretrained backbone).
"""

import logging

import numpy as np
import torch

import config

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    _LPIPS_IMPORT_OK = True
except Exception:  # pragma: no cover - defensive import guard
    _LPIPS_IMPORT_OK = False

_LPIPS_METRIC_CACHE = {"metric": None, "attempted": False}


def _get_lpips_metric(logger: logging.Logger):
    """Lazily construct and cache the LPIPS metric (loads pretrained weights
    only once per process).

    Returns None if LPIPS cannot be initialized (e.g. no internet access
    to download the backbone weights), in which case `calculate_lpips()`
    reports NaN instead of raising.
    """
    if _LPIPS_METRIC_CACHE["attempted"]:
        return _LPIPS_METRIC_CACHE["metric"]

    _LPIPS_METRIC_CACHE["attempted"] = True
    if not _LPIPS_IMPORT_OK:
        logger.warning("LPIPS module could not be imported. LPIPS scores will be reported as NaN.")
        return None

    try:
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(config.DEVICE)
        metric.eval()
        _LPIPS_METRIC_CACHE["metric"] = metric
        logger.info("LPIPS (AlexNet backbone) initialized successfully.")
        return metric
    except Exception as exc:
        logger.warning(
            f"LPIPS could not be initialized ({exc}). This usually means the pretrained "
            f"backbone weights could not be downloaded. LPIPS scores will be reported as NaN."
        )
        return None


def calculate_lpips(gt_gray: np.ndarray, pred_gray: np.ndarray, logger: logging.Logger) -> float:
    """Compute the LPIPS perceptual distance between two grayscale images.

    LPIPS is defined for 3-channel images, so the grayscale inputs are
    replicated across the RGB channels before evaluation, as recommended
    practice for evaluating single-channel (thermal) imagery with LPIPS.

    Parameters
    ----------
    gt_gray : np.ndarray
        Ground-truth grayscale image, uint8.
    pred_gray : np.ndarray
        Predicted grayscale image, uint8, same shape as gt_gray.
    logger : logging.Logger
        Logger used to record initialization warnings (only logged once).

    Returns
    -------
    float
        LPIPS distance (lower = more similar), or NaN if unavailable.
    """
    metric = _get_lpips_metric(logger)
    if metric is None:
        return float("nan")

    try:
        gt_rgb = np.stack([gt_gray] * 3, axis=-1).astype(np.float32) / 255.0
        pred_rgb = np.stack([pred_gray] * 3, axis=-1).astype(np.float32) / 255.0

        gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(config.DEVICE)
        pred_tensor = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            score = metric(pred_tensor, gt_tensor)
        return float(score.item())
    except Exception:
        return float("nan")
