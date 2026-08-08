import numpy as np


def compute_rmse(abs_error):
    """Computed on normalized [0,1] values. Percentage form (x100) is kept
    alongside the raw value, never replacing it."""
    rmse = float(np.sqrt(np.mean(np.square(abs_error))))
    return rmse, rmse * 100.0
