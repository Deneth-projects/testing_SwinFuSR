"""Absolute error, computed once and reused by MAE/RMSE/Max-Error and the
error-map visualization, instead of being recomputed in each place."""
import numpy as np


def compute_absolute_error(ground_truth, prediction):
    return np.abs(prediction.astype(np.float64) - ground_truth.astype(np.float64))
