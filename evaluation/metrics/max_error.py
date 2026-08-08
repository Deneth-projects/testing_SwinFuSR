import numpy as np


def compute_max_error(abs_error):
    """Per-image maximum. See analysis/statistics.py for the separate
    whole-dataset maximum (max across every sample, not an average)."""
    max_error = float(np.max(abs_error))
    return max_error, max_error * 100.0
