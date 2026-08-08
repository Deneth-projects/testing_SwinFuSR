import numpy as np


def compute_mae(abs_error):
    """Reported as Normalized Thermal Intensity MAE - no calibrated
    temperature values are available, so this is never in degrees."""
    mae = float(np.mean(abs_error))
    return mae, mae * 100.0
