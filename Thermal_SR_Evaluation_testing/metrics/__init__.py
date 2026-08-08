"""
metrics package
================
One file per metric. Each module contains exactly one `calculate_*`
function and nothing else — no plotting, no CSV writing, no ranking, no
folder operations.

This __init__.py only re-exports the metric functions for convenient
importing (e.g. `from metrics import calculate_psnr`). To add a new
metric, create a new module in this package and add it to the imports
and `__all__` list below — no other project file needs to change.
"""

from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from metrics.ms_ssim import calculate_ms_ssim
from metrics.lpips_metric import calculate_lpips
from metrics.mae import calculate_mae
from metrics.rmse import calculate_rmse
from metrics.max_error import calculate_max_error
from metrics.normalized_error import calculate_normalized_error

__all__ = [
    "calculate_psnr",
    "calculate_ssim",
    "calculate_ms_ssim",
    "calculate_lpips",
    "calculate_mae",
    "calculate_rmse",
    "calculate_max_error",
    "calculate_normalized_error",
]
