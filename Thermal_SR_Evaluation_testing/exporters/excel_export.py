"""
exporters/excel_export.py
===========================
Writes every Excel workbook produced by the project, with auto-adjusted
column widths and bold headers. Nothing else.
"""

import logging
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter


def save_dataframe_excel(df: pd.DataFrame, path: Path, logger: logging.Logger, label: str) -> None:
    """Save a DataFrame to a neatly formatted Excel (.xlsx) file.

    Column widths are auto-adjusted based on the longest value in each
    column, and the header row is bolded for readability.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    path : Path
        Destination XLSX file path.
    logger : logging.Logger
        Logger used to record the outcome.
    label : str
        Human-readable label for the log message.
    """
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            sheet_name = "Results"
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]

            _bold_header_row(worksheet)
            _auto_adjust_column_widths(worksheet, df)

        logger.info(f"{label} Excel file saved: {path}")
    except Exception as exc:
        logger.error(f"Failed to save {label} Excel file: {exc}")


def _bold_header_row(worksheet) -> None:
    """Apply a bold font to the first (header) row of a worksheet."""
    for cell in worksheet[1]:
        cell.font = Font(bold=True)


def _auto_adjust_column_widths(worksheet, df: pd.DataFrame) -> None:
    """Resize every column to fit the longest value it contains."""
    for col_idx, column in enumerate(df.columns, start=1):
        max_length = max([len(str(column))] + [len(str(v)) for v in df[column].tolist()])
        worksheet.column_dimensions[get_column_letter(col_idx)].width = max_length + 4
