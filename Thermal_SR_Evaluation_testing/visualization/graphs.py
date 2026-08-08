"""
visualization/graphs.py
=========================
General-purpose figure generation that does not belong in histograms.py or
error_map.py. Currently contains the slice intensity-profile plot; future
figure types (e.g. per-model comparison plots) should be added here as new
functions.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless execution
import matplotlib.pyplot as plt
import numpy as np


def save_slice_profile_plot(
    index: int,
    gt_h_slice: np.ndarray,
    pred_h_slice: np.ndarray,
    gt_v_slice: np.ndarray,
    pred_v_slice: np.ndarray,
    destination_folder: Path,
    logger: logging.Logger,
) -> None:
    """Save a side-by-side plot comparing GT vs prediction intensity profiles
    along the horizontal and vertical centre slices of one sample.

    Parameters
    ----------
    index : int
        Sample index, used in plot titles.
    gt_h_slice, pred_h_slice : np.ndarray
        Ground-truth and predicted horizontal centre slices.
    gt_v_slice, pred_v_slice : np.ndarray
        Ground-truth and predicted vertical centre slices.
    destination_folder : Path
        Folder the resulting PNG is saved into.
    logger : logging.Logger
        Logger used to record warnings.
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(np.mean(gt_h_slice, axis=0), label="Ground Truth")
        axes[0].plot(np.mean(pred_h_slice, axis=0), label="Prediction")
        axes[0].set_title(f"Sample {index} — Horizontal Centre Profile")
        axes[0].set_xlabel("Pixel Position")
        axes[0].set_ylabel("Mean Intensity")
        axes[0].legend()

        axes[1].plot(np.mean(gt_v_slice, axis=1), label="Ground Truth")
        axes[1].plot(np.mean(pred_v_slice, axis=1), label="Prediction")
        axes[1].set_title(f"Sample {index} — Vertical Centre Profile")
        axes[1].set_xlabel("Pixel Position")
        axes[1].set_ylabel("Mean Intensity")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(destination_folder / "slice_profile_plot.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"Slice profile plot failed for sample {index}: {exc}")
