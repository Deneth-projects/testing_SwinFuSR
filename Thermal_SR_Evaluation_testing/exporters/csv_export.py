"""
exporters/csv_export.py
=========================
Writes every CSV file produced by the project. Nothing else.
"""

import logging
from pathlib import Path

import pandas as pd


def save_dataframe_csv(df: pd.DataFrame, path: Path, logger: logging.Logger, label: str) -> None:
    """Save a DataFrame to CSV, logging success or failure without raising.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    path : Path
        Destination CSV file path.
    logger : logging.Logger
        Logger used to record the outcome.
    label : str
        Human-readable label for the log message (e.g. 'Image Metrics').
    """
    try:
        df.to_csv(path, index=False)
        logger.info(f"{label} CSV saved: {path}")
    except Exception as exc:
        logger.error(f"Failed to save {label} CSV: {exc}")
