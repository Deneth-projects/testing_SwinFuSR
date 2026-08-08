"""
analysis/statistics.py
========================
Computes Mean, Median, Standard Deviation, Minimum and Maximum for every
metric. No plotting, no exporting.
"""

import pandas as pd

_SUMMARY_STATISTIC_NAMES = ["Mean", "Median", "Standard Deviation", "Minimum", "Maximum"]

_NUMERIC_COLUMNS = [
    "PSNR", "SSIM", "MS-SSIM", "LPIPS", "MAE", "RMSE",
    "Maximum Error", "Normalized Error %", "Processing Time (s)",
]


def compute_summary_statistics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Mean / Median / Std / Min / Max for every numeric metric column.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Full per-image metrics table.

    Returns
    -------
    pd.DataFrame
        Summary table with a 'Statistic' column plus one column per metric.
    """
    numeric_columns = [c for c in _NUMERIC_COLUMNS if c in metrics_df.columns]

    summary = metrics_df[numeric_columns].agg(["mean", "median", "std", "min", "max"])
    summary.index = _SUMMARY_STATISTIC_NAMES
    summary.index.name = "Statistic"
    return summary.reset_index()
