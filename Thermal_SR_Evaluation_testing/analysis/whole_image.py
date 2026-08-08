"""
analysis/whole_image.py
=========================
Loops through all verified samples, calls the metric functions in the
metrics/ package, and collects the results into a DataFrame.

No plotting. No exporting.
"""

import logging
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from helpers import timer
from image_loader import load_grayscale_image
from matcher import SamplePaths
from metrics import (
    calculate_lpips,
    calculate_mae,
    calculate_max_error,
    calculate_ms_ssim,
    calculate_normalized_error,
    calculate_psnr,
    calculate_rmse,
    calculate_ssim,
)

# Column order matches the required CSV/Excel schema exactly.
_COLUMN_ORDER = [
    "Image ID", "PSNR", "SSIM", "MS-SSIM", "LPIPS", "MAE", "RMSE",
    "Maximum Error", "Normalized Error %", "Processing Time (s)",
]


def evaluate_all_images(valid_samples: list[SamplePaths], logger: logging.Logger) -> pd.DataFrame:
    """Compute whole-image metrics for every verified sample.

    Parameters
    ----------
    valid_samples : list[SamplePaths]
        Samples that passed dataset verification (see image_loader.verify_dataset).
    logger : logging.Logger
        Logger used to record per-image progress and warnings.

    Returns
    -------
    pd.DataFrame
        One row per successfully evaluated image, columns ordered per the
        project's CSV/Excel schema, plus an 'Index' column used internally
        by ranking/export steps.
    """
    per_image_rows = []
    progress_bar = tqdm(valid_samples, desc="Evaluating images", unit="image", total=len(valid_samples))

    for sample in progress_bar:
        progress_bar.set_postfix_str(f"Current: sample_{sample.index}")
        row = _evaluate_single_sample(sample, logger)
        if row is not None:
            per_image_rows.append(row)

    progress_bar.close()

    if len(per_image_rows) == 0:
        return pd.DataFrame(columns=_COLUMN_ORDER + ["Index"])

    metrics_df = pd.DataFrame(per_image_rows)
    ordered_columns = [c for c in _COLUMN_ORDER if c in metrics_df.columns] + ["Index"]
    return metrics_df[ordered_columns]


def _evaluate_single_sample(sample: SamplePaths, logger: logging.Logger) -> dict | None:
    """Load, evaluate and time a single sample. Returns None (and logs a
    warning) if the sample could not be evaluated, so that one damaged
    image never stops the overall evaluation."""
    logger.info(f"Current Image: sample_{sample.index}")

    try:
        with timer() as t:
            gt_gray = load_grayscale_image(sample.gt_path, target_size=config.INPUT_IMAGE_SIZE)
            pred_gray = load_grayscale_image(sample.pred_path, target_size=config.INPUT_IMAGE_SIZE)

            if gt_gray is None or pred_gray is None:
                raise ValueError("Ground truth or prediction image could not be decoded.")
            if gt_gray.shape != pred_gray.shape:
                raise ValueError(f"Shape mismatch after loading: GT={gt_gray.shape}, Pred={pred_gray.shape}.")

            logger.info(f"Metric Calculation: computing metrics for sample_{sample.index}.")
            metrics = _compute_metrics_for_pair(gt_gray, pred_gray, logger)

        return {
            "Image ID": f"sample_{sample.index}",
            "Index": sample.index,
            **metrics,
            "Processing Time (s)": t.elapsed_seconds,
        }

    except Exception as exc:
        logger.error(f"Evaluation failed for sample_{sample.index}: {exc}\n{traceback.format_exc()}")
        print(f"File sample_{sample.index} could not be evaluated ({exc}). Skipping...")
        return None


def _compute_metrics_for_pair(gt_gray: np.ndarray, pred_gray: np.ndarray, logger: logging.Logger) -> dict:
    """Compute the full set of whole-image metrics for one GT/Prediction pair."""
    rmse = calculate_rmse(gt_gray, pred_gray)
    return {
        "PSNR": calculate_psnr(gt_gray, pred_gray),
        "SSIM": calculate_ssim(gt_gray, pred_gray),
        "MS-SSIM": calculate_ms_ssim(gt_gray, pred_gray),
        "LPIPS": calculate_lpips(gt_gray, pred_gray, logger),
        "MAE": calculate_mae(gt_gray, pred_gray),
        "RMSE": rmse,
        "Maximum Error": calculate_max_error(gt_gray, pred_gray),
        "Normalized Error %": calculate_normalized_error(rmse),
    }
