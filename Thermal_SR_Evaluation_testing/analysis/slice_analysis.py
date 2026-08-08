"""
analysis/slice_analysis.py
============================
Works ONLY on the selected Worst Samples. Extracts horizontal and vertical
centre slices and computes PSNR, SSIM, MAE, RMSE for each. Returns a
DataFrame. No plotting logic itself (delegates to visualization/graphs.py),
no exporting.
"""

import logging

import cv2
import numpy as np
import pandas as pd

import config
from image_loader import load_grayscale_image
from matcher import build_sample_paths
from metrics import calculate_mae, calculate_psnr, calculate_rmse, calculate_ssim
from visualization.graphs import save_slice_profile_plot


def extract_center_slices(image: np.ndarray, slice_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract the horizontal and vertical centre slices of an image.

    The horizontal slice is a thin strip of rows taken around the vertical
    centre of the image, spanning the full image width. The vertical slice
    is a thin strip of columns taken around the horizontal centre of the
    image, spanning the full image height.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W).
    slice_width : int
        Thickness (in pixels) of the extracted centre strip.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (horizontal_slice, vertical_slice)
    """
    height, width = image.shape[:2]
    half = max(slice_width // 2, 1)

    row_center = height // 2
    row_start = max(row_center - half, 0)
    row_end = min(row_center + half, height)
    horizontal_slice = image[row_start:row_end, :]

    col_center = width // 2
    col_start = max(col_center - half, 0)
    col_end = min(col_center + half, width)
    vertical_slice = image[:, col_start:col_end]

    return horizontal_slice, vertical_slice


def compute_slice_metrics(gt_slice: np.ndarray, pred_slice: np.ndarray) -> dict:
    """Compute PSNR, SSIM, MAE and RMSE for one pair of image slices.

    Parameters
    ----------
    gt_slice : np.ndarray
        Ground-truth image slice.
    pred_slice : np.ndarray
        Predicted image slice, same shape as gt_slice.

    Returns
    -------
    dict
        Dictionary with keys 'PSNR', 'SSIM', 'MAE', 'RMSE'.
    """
    win_size = _adaptive_ssim_window(gt_slice)
    return {
        "PSNR": calculate_psnr(gt_slice, pred_slice),
        "SSIM": calculate_ssim(gt_slice, pred_slice, win_size=win_size) if win_size else float("nan"),
        "MAE": calculate_mae(gt_slice, pred_slice),
        "RMSE": calculate_rmse(gt_slice, pred_slice),
    }


def _adaptive_ssim_window(slice_image: np.ndarray) -> int | None:
    """Pick an odd SSIM window size that fits the (typically very thin)
    slice dimensions, or None if the slice is too small for SSIM at all."""
    smaller_dim = min(slice_image.shape[0], slice_image.shape[1])
    if smaller_dim < 3:
        return None
    win_size = smaller_dim if smaller_dim % 2 == 1 else smaller_dim - 1
    return max(win_size, 3)


def generate_slice_analysis(
    worst_samples_df: pd.DataFrame,
    folders: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Generate centre-slice images and slice-level metrics for the Worst samples.

    Slice analysis is intentionally restricted to the Worst samples only
    (never for the full dataset), as these are the cases of greatest
    diagnostic interest.

    Parameters
    ----------
    worst_samples_df : pd.DataFrame
        Rows (from the main metrics DataFrame) corresponding to the
        selected worst-performing samples. Must contain an 'Index' column.
    folders : dict
        Output folder mapping produced by helpers.create_output_folders().
    logger : logging.Logger
        Logger used to record progress and warnings.

    Returns
    -------
    pd.DataFrame
        One row per (sample, orientation) containing the slice metrics.
    """
    slice_rows = []

    for _, row in worst_samples_df.iterrows():
        index = int(row["Index"])
        sample_rows = _process_worst_sample(index, folders, logger)
        slice_rows.extend(sample_rows)

    return pd.DataFrame(slice_rows)


def _process_worst_sample(index: int, folders: dict, logger: logging.Logger) -> list[dict]:
    """Run slice extraction, metric computation, image saving and profile
    plotting for a single worst-ranked sample. Returns the slice metric
    rows for this sample (empty list if the sample could not be loaded)."""
    sample = build_sample_paths(index)
    gt_gray = load_grayscale_image(sample.gt_path, target_size=config.INPUT_IMAGE_SIZE)
    pred_gray = load_grayscale_image(sample.pred_path, target_size=config.INPUT_IMAGE_SIZE)

    if gt_gray is None or pred_gray is None:
        logger.warning(f"Slice analysis skipped for sample {index}: image could not be loaded.")
        return []

    sample_slice_dir = folders["slices"] / f"sample_{index}"
    sample_slice_dir.mkdir(parents=True, exist_ok=True)

    gt_h_slice, gt_v_slice = extract_center_slices(gt_gray, config.CENTER_SLICE_WIDTH)
    pred_h_slice, pred_v_slice = extract_center_slices(pred_gray, config.CENTER_SLICE_WIDTH)

    if config.SAVE_SLICES:
        cv2.imwrite(str(sample_slice_dir / "horizontal_slice_gt.png"), gt_h_slice)
        cv2.imwrite(str(sample_slice_dir / "horizontal_slice_pred.png"), pred_h_slice)
        cv2.imwrite(str(sample_slice_dir / "vertical_slice_gt.png"), gt_v_slice)
        cv2.imwrite(str(sample_slice_dir / "vertical_slice_pred.png"), pred_v_slice)

    h_metrics = compute_slice_metrics(gt_h_slice, pred_h_slice)
    v_metrics = compute_slice_metrics(gt_v_slice, pred_v_slice)

    save_slice_profile_plot(index, gt_h_slice, pred_h_slice, gt_v_slice, pred_v_slice, sample_slice_dir, logger)

    logger.info(f"Slice analysis completed for worst sample {index}.")

    return [
        {"Image ID": f"sample_{index}", "Orientation": "Horizontal", **h_metrics},
        {"Image ID": f"sample_{index}", "Orientation": "Vertical", **v_metrics},
    ]
