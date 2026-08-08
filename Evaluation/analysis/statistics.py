"""Dataset-level statistics across all pairs.

Mean Maximum Error (average of each pair's own max error) is kept distinct
from Whole-Dataset Maximum Error (the single largest pixel error across
every pair) - these are never the same thing."""
import numpy as np

METRIC_KEYS = [
    ("psnr", "PSNR"),
    ("ssim", "SSIM"),
    ("ms_ssim", "MS-SSIM"),
    ("lpips", "LPIPS"),
    ("mae", "MAE"),
    ("mae_percent", "MAE (%)"),
    ("rmse", "RMSE"),
    ("rmse_percent", "RMSE (%)"),
    ("max_error", "Maximum Error"),
    ("max_error_percent", "Maximum Error (%)"),
]


def _stats_for(values):
    clean = np.array([v for v in values if v is not None], dtype=np.float64)
    if clean.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=0)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def compute_summary_statistics(results):
    return {label: _stats_for([r[key] for r in results]) for key, label in METRIC_KEYS}


def compute_whole_dataset_max_error(results):
    if not results:
        return None, None
    whole_max = max(float(np.max(r["_abs_error"])) for r in results)
    return whole_max, whole_max * 100.0
