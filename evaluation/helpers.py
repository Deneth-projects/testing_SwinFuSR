import os

import config


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def round4(value):
    """Round for reporting/export only - internal math stays full precision."""
    if value is None:
        return None
    try:
        return round(float(value), config.DECIMAL_PLACES)
    except (TypeError, ValueError):
        return value


def format_metric(value, suffix=""):
    if value is None:
        return "N/A"
    return f"{float(value):.{config.DECIMAL_PLACES}f}{suffix}"
