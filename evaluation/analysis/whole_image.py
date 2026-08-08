"""Whole-image quantitative evaluation for a single pair: load GT/predicted,
normalize to [0,1], compute every metric. This is the only analysis this
project performs - no slices/ROI/hotspot/frequency/best-worst selection."""
import image_loader
from metrics.lpips_metric import compute_lpips
from metrics.mae import compute_mae
from metrics.max_error import compute_max_error
from metrics.ms_ssim import compute_ms_ssim
from metrics.normalized_error import compute_absolute_error
from metrics.psnr import compute_psnr
from metrics.rmse import compute_rmse
from metrics.ssim import compute_ssim


def evaluate_pair(sample, logger=None):
    ground_truth = image_loader.load_thermal_image(sample.hr_gt_path)
    prediction = image_loader.load_thermal_image(sample.pred_path)

    if ground_truth.shape != prediction.shape:
        if logger:
            logger.info(
                f"  Resizing predicted {prediction.shape} -> ground truth "
                f"{ground_truth.shape} for {sample.pair_id}."
            )
        prediction = image_loader.resize_to_match(prediction, ground_truth.shape)

    abs_error = compute_absolute_error(ground_truth, prediction)
    mae, mae_percent = compute_mae(abs_error)
    rmse, rmse_percent = compute_rmse(abs_error)
    max_error, max_error_percent = compute_max_error(abs_error)

    return {
        "pair_id": sample.pair_id,
        "index": sample.index,
        "hr_gt_path": sample.hr_gt_path,
        "pred_path": sample.pred_path,
        "image_shape": ground_truth.shape,
        "psnr": compute_psnr(ground_truth, prediction),
        "ssim": compute_ssim(ground_truth, prediction),
        "ms_ssim": compute_ms_ssim(ground_truth, prediction),
        "lpips": compute_lpips(ground_truth, prediction),
        "mae": mae,
        "mae_percent": mae_percent,
        "rmse": rmse,
        "rmse_percent": rmse_percent,
        "max_error": max_error,
        "max_error_percent": max_error_percent,
        # reused by error-map visualization and whole-dataset max error
        "_abs_error": abs_error,
    }
