"""
visualization/error_map.py
============================
Absolute error map computation and saving. No normalization is applied —
error maps are saved exactly as computed.
"""

import logging

import cv2
import numpy as np
import pandas as pd

import config
from image_loader import load_grayscale_image
from matcher import build_sample_paths


def compute_error_map(gt_gray: np.ndarray, pred_gray: np.ndarray) -> np.ndarray:
    """Compute the (non-normalized) absolute error map between GT and prediction.

    Error Map = |Prediction - Ground Truth|

    Parameters
    ----------
    gt_gray : np.ndarray
        Ground-truth grayscale image.
    pred_gray : np.ndarray
        Predicted grayscale image, same shape as gt_gray.

    Returns
    -------
    np.ndarray
        uint8 absolute-error map, saved exactly without any contrast
        stretching or normalization.
    """
    return np.abs(pred_gray.astype(np.int16) - gt_gray.astype(np.int16)).astype(np.uint8)


def save_global_error_maps(metrics_df: pd.DataFrame, folders: dict, logger: logging.Logger) -> None:
    """Save the absolute error map for every successfully evaluated sample
    into the top-level Error_Maps output folder.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Full per-image metrics table. Must contain an 'Index' column.
    folders : dict
        Output folder mapping produced by helpers.create_output_folders().
    logger : logging.Logger
        Logger used to record warnings.
    """
    if not config.SAVE_ERROR_MAPS:
        return

    for _, row in metrics_df.iterrows():
        index = int(row["Index"])
        sample = build_sample_paths(index)
        gt_gray = load_grayscale_image(sample.gt_path, target_size=config.INPUT_IMAGE_SIZE)
        pred_gray = load_grayscale_image(sample.pred_path, target_size=config.INPUT_IMAGE_SIZE)

        if gt_gray is None or pred_gray is None:
            logger.warning(f"Error map skipped for sample {index}: image could not be loaded.")
            continue

        error_map = compute_error_map(gt_gray, pred_gray)
        cv2.imwrite(str(folders["error_maps"] / f"error_map_sample_{index}.png"), error_map)
