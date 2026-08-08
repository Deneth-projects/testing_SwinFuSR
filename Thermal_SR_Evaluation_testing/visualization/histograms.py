"""
visualization/histograms.py
=============================
Generates PSNR, SSIM, RMSE and MAE histograms across all evaluated images.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless execution
import matplotlib.pyplot as plt
import pandas as pd

import config

_HISTOGRAM_SPECS = [
    ("PSNR", "PSNR (dB)", "PSNR Distribution Across Evaluated Images"),
    ("SSIM", "SSIM", "SSIM Distribution Across Evaluated Images"),
    ("RMSE", "RMSE", "RMSE Distribution Across Evaluated Images"),
    ("MAE", "MAE", "MAE Distribution Across Evaluated Images"),
]


def generate_histograms(metrics_df: pd.DataFrame, graphs_folder: Path, logger: logging.Logger) -> None:
    """Generate and save histogram plots for PSNR, SSIM, RMSE and MAE.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Full per-image metrics table.
    graphs_folder : Path
        Destination folder for the PNG histogram files.
    logger : logging.Logger
        Logger used to record progress and warnings.
    """
    if not config.SAVE_GRAPHS:
        return

    for column, x_label, title in _HISTOGRAM_SPECS:
        _save_single_histogram(metrics_df, column, x_label, title, graphs_folder, logger)


def _save_single_histogram(
    metrics_df: pd.DataFrame,
    column: str,
    x_label: str,
    title: str,
    graphs_folder: Path,
    logger: logging.Logger,
) -> None:
    """Render and save one histogram for a single metric column."""
    try:
        values = metrics_df[column].dropna().values
        if len(values) == 0:
            logger.warning(f"No valid data to plot histogram for {column}.")
            return

        fig, ax = plt.subplots(figsize=(7, 5))
        bin_count = min(max(len(values), 3), 20)
        ax.hist(values, bins=bin_count, color="#3B7DD8", edgecolor="black", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Number of Images")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()

        output_path = graphs_folder / f"{column.replace(' ', '_')}_histogram.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info(f"Histogram saved: {output_path}")
    except Exception as exc:
        logger.warning(f"Failed to generate histogram for {column}: {exc}")
