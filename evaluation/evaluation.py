"""
Entry point. Reads config.py, discovers GT/predicted pairs, evaluates each,
exports CSV/Excel/graphs/error maps.

Run: python evaluation.py
"""
import os
import traceback

import config
import helpers
import matcher
import logger as logger_module
from analysis.statistics import compute_summary_statistics, compute_whole_dataset_max_error
from analysis.whole_image import evaluate_pair
from exporters.csv_export import export_image_metrics_csv, export_summary_statistics_csv
from exporters.excel_export import export_excel, export_summary_only_excel
from visualization.error_map import save_error_map

try:
    from visualization.graphs import save_summary_graphs
    GRAPHS_AVAILABLE = True
except Exception:
    GRAPHS_AVAILABLE = False


def evaluate_all_pairs(samples, log):
    results, skipped, lpips_checked = [], 0, False

    for sample in samples:
        try:
            log.info(f"Evaluating {sample.pair_id} "
                      f"(GT: {os.path.basename(sample.hr_gt_path)}, "
                      f"Pred: {os.path.basename(sample.pred_path)}) ...")
            result = evaluate_pair(sample, logger=log)
            results.append(result)

            if not lpips_checked:
                lpips_checked = True
                log.info("  LPIPS OK" if result["lpips"] is not None else "  LPIPS unavailable (see warning below)")

            if config.SAVE_ERROR_MAPS:
                save_error_map(result["_abs_error"], sample.pair_id)

        except Exception:
            skipped += 1
            filename = os.path.basename(sample.pred_path or sample.hr_gt_path or sample.pair_id)
            log.warning(f"File {filename} cannot be opened. Skipping...")
            log.debug(traceback.format_exc())

    return results, skipped


def print_summary(log, results, summary, whole_max, whole_max_percent, n_found, skipped):
    fmt = helpers.format_metric
    log.info("")
    log.info("=" * 78)
    log.info("EVALUATION COMPLETE")
    log.info("=" * 78)
    log.info("")
    log.info(f"Pairs evaluated         : {len(results)} / {n_found}")
    log.info(f"Pairs skipped           : {skipped}")
    log.info("")
    log.info(f"Mean PSNR               : {fmt(summary['PSNR']['mean'], ' dB')}")
    log.info(f"Mean SSIM               : {fmt(summary['SSIM']['mean'])}")
    log.info(f"Mean MS-SSIM            : {fmt(summary['MS-SSIM']['mean'])}")
    log.info(f"Mean LPIPS              : {fmt(summary['LPIPS']['mean'])}")
    log.info("")
    log.info(f"Mean MAE                : {fmt(summary['MAE']['mean'])}")
    log.info(f"Mean MAE (%)            : {fmt(summary['MAE (%)']['mean'], '%')}")
    log.info("")
    log.info(f"Mean RMSE               : {fmt(summary['RMSE']['mean'])}")
    log.info(f"Mean RMSE (%)           : {fmt(summary['RMSE (%)']['mean'], '%')}")
    log.info("")
    log.info(f"Mean Max Error          : {fmt(summary['Maximum Error']['mean'])}")
    log.info(f"Mean Max Error (%)      : {fmt(summary['Maximum Error (%)']['mean'], '%')}")
    log.info("")
    log.info(f"Whole Dataset Max Error     : {fmt(whole_max)}")
    log.info(f"Whole Dataset Max Error (%) : {fmt(whole_max_percent, '%')}")
    log.info("")


def run():
    helpers.ensure_dir(config.OUTPUT_FOLDER)
    log, log_path = logger_module.setup_logger()

    log.info("=" * 78)
    log.info("THERMAL SUPER-RESOLUTION EVALUATION")
    log.info("=" * 78)
    log.info(f"HR Ground Truth Folder : {config.HR_GT_FOLDER}")
    log.info(f"Predicted Folder       : {config.PRED_FOLDER}")
    log.info(f"Output Folder          : {config.OUTPUT_FOLDER}")
    log.info("")

    samples = matcher.discover_samples()
    n_found = len(samples)
    log.info(f"Pairs discovered (matched GT + Predicted): {n_found}")
    for s in samples:
        if s.lr_path is None:
            log.warning(f"  [{s.pair_id}] LR image not found (not required for metrics).")
        if s.rgb_path is None:
            log.warning(f"  [{s.pair_id}] RGB image not found (not required for metrics).")

    results, skipped = evaluate_all_pairs(samples, log)
    if not results:
        log.error("No pairs were successfully evaluated. Exiting.")
        return

    if any(r["lpips"] is None for r in results):
        log.warning(
            "LPIPS could not be computed for one or more pairs (pretrained "
            "backbone weights unavailable - internet access is required on "
            "first use). Reported as N/A for those pairs; other metrics unaffected."
        )

    summary = compute_summary_statistics(results)
    whole_max, whole_max_percent = compute_whole_dataset_max_error(results)

    csv_path = export_image_metrics_csv(results)
    summary_csv_path = export_summary_statistics_csv(summary, whole_max, whole_max_percent)
    excel_path = export_excel(results, summary, whole_max, whole_max_percent,
                               n_found, len(results), skipped, log_path=log_path)
    summary_excel_path = export_summary_only_excel(summary, whole_max, whole_max_percent)

    graph_paths = []
    if config.SAVE_GRAPHS and GRAPHS_AVAILABLE:
        try:
            graph_paths = save_summary_graphs(results)
        except Exception:
            log.debug(traceback.format_exc())

    print_summary(log, results, summary, whole_max, whole_max_percent, n_found, skipped)

    log.info(f"Output folder           : {config.OUTPUT_FOLDER}")
    log.info(f"  Image CSV              : {csv_path}")
    log.info(f"  Summary CSV            : {summary_csv_path}")
    log.info(f"  Excel report           : {excel_path}")
    log.info(f"  Summary Excel          : {summary_excel_path}")
    if graph_paths:
        log.info(f"  Graphs                 : {os.path.join(config.OUTPUT_FOLDER, 'Graphs')}")
    if config.SAVE_ERROR_MAPS:
        log.info(f"  Error maps             : {os.path.join(config.OUTPUT_FOLDER, 'Error_Maps')}")
    log.info(f"  Log file               : {log_path}")
    log.info("=" * 78)


if __name__ == "__main__":
    run()
