import os

import pandas as pd

import config
import helpers

IMAGE_METRICS_COLUMNS = [
    "Pair_ID", "PSNR_dB", "SSIM", "MS_SSIM", "LPIPS",
    "MAE", "MAE_percent", "RMSE", "RMSE_percent",
    "Maximum_Error", "Maximum_Error_percent",
]
SUMMARY_COLUMNS = ["Metric", "Mean", "Std_Dev", "Min", "Max"]


def export_image_metrics_csv(results):
    if not config.SAVE_CSV:
        return None

    csv_folder = helpers.ensure_dir(os.path.join(config.OUTPUT_FOLDER, "CSV"))
    out_path = os.path.join(csv_folder, "image_metrics.csv")

    rows = [{
        "Pair_ID": r["pair_id"],
        "PSNR_dB": helpers.round4(r["psnr"]),
        "SSIM": helpers.round4(r["ssim"]),
        "MS_SSIM": helpers.round4(r["ms_ssim"]),
        "LPIPS": helpers.round4(r["lpips"]) if r["lpips"] is not None else "N/A",
        "MAE": helpers.round4(r["mae"]),
        "MAE_percent": helpers.round4(r["mae_percent"]),
        "RMSE": helpers.round4(r["rmse"]),
        "RMSE_percent": helpers.round4(r["rmse_percent"]),
        "Maximum_Error": helpers.round4(r["max_error"]),
        "Maximum_Error_percent": helpers.round4(r["max_error_percent"]),
    } for r in results]

    pd.DataFrame(rows, columns=IMAGE_METRICS_COLUMNS).to_csv(out_path, index=False)
    return out_path


def export_summary_statistics_csv(summary, whole_dataset_max, whole_dataset_max_percent):
    if not config.SAVE_CSV:
        return None

    csv_folder = helpers.ensure_dir(os.path.join(config.OUTPUT_FOLDER, "CSV"))
    out_path = os.path.join(csv_folder, "summary_statistics.csv")

    rows = [{
        "Metric": label,
        "Mean": helpers.round4(stats["mean"]),
        "Std_Dev": helpers.round4(stats["std"]),
        "Min": helpers.round4(stats["min"]),
        "Max": helpers.round4(stats["max"]),
    } for label, stats in summary.items()]

    rows.append({"Metric": "Whole-Dataset Maximum Error", "Mean": "-", "Std_Dev": "-",
                 "Min": "-", "Max": helpers.round4(whole_dataset_max)})
    rows.append({"Metric": "Whole-Dataset Maximum Error (%)", "Mean": "-", "Std_Dev": "-",
                 "Min": "-", "Max": helpers.round4(whole_dataset_max_percent)})

    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(out_path, index=False)
    return out_path
