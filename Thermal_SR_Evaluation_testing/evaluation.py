#!/usr/bin/env python3
"""
evaluation.py
==============
Entry point for the Thermal-Guided Super-Resolution evaluation project.

This file contains ONLY the main execution flow. It performs almost no
calculation itself — it reads configuration, then calls into the
project's modules in order:

    Read configuration
        -> Verify dataset
        -> Evaluate images
        -> Rank samples
        -> Export results
        -> Generate plots
        -> Finish

See README.md for setup instructions and a description of every module.
Edit config.py to point at your own dataset — no other file should need
to change for a standard evaluation run.
"""

import config
from logger import setup_logging
from helpers import create_output_folders, export_sample_group
from matcher import discover_sample_indices
from image_loader import verify_dataset
from analysis.whole_image import evaluate_all_images
from analysis.ranking import rank_and_select_samples
from analysis.slice_analysis import generate_slice_analysis
from analysis.statistics import compute_summary_statistics
from visualization.error_map import save_global_error_maps
from visualization.histograms import generate_histograms
from exporters.csv_export import save_dataframe_csv
from exporters.excel_export import save_dataframe_excel


def main() -> None:
    """Run the complete thermal super-resolution evaluation pipeline."""

    # 1. Prepare output folders and logging.
    folders = create_output_folders(config.OUTPUT_FOLDER)
    logger = setup_logging(folders["logs"])
    logger.info("=" * 78)
    logger.info("Evaluation Started")
    logger.info(f"Device selected for torch-based metrics: {config.DEVICE}")
    logger.info("=" * 78)

    # 2. Discover and verify the dataset.
    candidate_indices = discover_sample_indices(config.LR_FOLDER, config.LR_PREFIX, config.LR_EXTENSION)
    logger.info(f"Images Found: {len(candidate_indices)} candidate sample(s) in '{config.LR_FOLDER}'.")

    valid_samples = verify_dataset(candidate_indices, logger)
    skipped_count = len(candidate_indices) - len(valid_samples)
    logger.info(f"Images Skipped: {skipped_count}")

    if len(valid_samples) == 0:
        logger.error(
            "No valid samples found. Evaluation cannot proceed. Please check config.py "
            "(folder paths / prefixes / extensions)."
        )
        print("No valid samples found. Please verify your folder paths and filename settings.")
        return

    # 3. Evaluate every valid sample (whole-image metrics).
    metrics_df = evaluate_all_images(valid_samples, logger)
    if metrics_df.empty:
        logger.error("No samples were successfully evaluated. Aborting output generation.")
        print("No samples were successfully evaluated. Please check evaluation.log for details.")
        return

    export_df = metrics_df.drop(columns=["Index"])
    if config.SAVE_CSV:
        save_dataframe_csv(export_df, folders["csv"] / "image_metrics.csv", logger, "Image Metrics")
    if config.SAVE_EXCEL:
        save_dataframe_excel(export_df, folders["excel"] / "image_metrics.xlsx", logger, "Image Metrics")

    # 4. Summary statistics.
    summary_df = compute_summary_statistics(metrics_df)
    if config.SAVE_CSV:
        save_dataframe_csv(summary_df, folders["csv"] / "summary_statistics.csv", logger, "Summary Statistics")
    if config.SAVE_EXCEL:
        save_dataframe_excel(summary_df, folders["excel"] / "summary_statistics.xlsx", logger, "Summary Statistics")

    # 5. Rank samples and export Best / Worst groups.
    best_samples_df, worst_samples_df = rank_and_select_samples(metrics_df)
    logger.info(
        f"Ranking complete. Selected {len(best_samples_df)} best sample(s) and "
        f"{len(worst_samples_df)} worst sample(s) (primary key: lowest RMSE, "
        f"secondary key: highest SSIM)."
    )
    export_sample_group(best_samples_df, folders["best"], logger)
    export_sample_group(worst_samples_df, folders["worst"], logger)

    # 6. Global error maps (every evaluated sample).
    save_global_error_maps(metrics_df, folders, logger)

    # 7. Slice analysis (Worst samples only).
    if config.SAVE_SLICES:
        slice_metrics_df = generate_slice_analysis(worst_samples_df, folders, logger)
        if not slice_metrics_df.empty:
            if config.SAVE_CSV:
                save_dataframe_csv(slice_metrics_df, folders["csv"] / "slice_metrics.csv", logger, "Slice Metrics")
            if config.SAVE_EXCEL:
                save_dataframe_excel(slice_metrics_df, folders["excel"] / "slice_metrics.xlsx", logger, "Slice Metrics")

    # 8. Graph generation.
    generate_histograms(metrics_df, folders["graphs"], logger)
    if config.SAVE_GRAPHS:
        logger.info("Graphs Saved")

    # 9. Finish.
    logger.info("Evaluation Completed")
    logger.info("=" * 78)
    _print_final_summary(metrics_df, candidate_indices, skipped_count, folders)


def _print_final_summary(metrics_df, candidate_indices: list[int], skipped_count: int, folders: dict) -> None:
    """Print a short human-readable summary of the completed run to the console."""
    print("\n" + "=" * 78)
    print("EVALUATION COMPLETE")
    print("=" * 78)
    print(f"Samples evaluated : {len(metrics_df)} / {len(candidate_indices)}")
    print(f"Samples skipped   : {skipped_count}")
    print(f"Mean PSNR         : {metrics_df['PSNR'].mean():.4f} dB")
    print(f"Mean SSIM         : {metrics_df['SSIM'].mean():.4f}")
    print(f"Mean MS-SSIM      : {metrics_df['MS-SSIM'].mean():.4f}")
    print(f"Mean LPIPS        : {metrics_df['LPIPS'].mean():.4f}")
    print(f"Mean RMSE         : {metrics_df['RMSE'].mean():.4f}")
    print(f"Output folder     : {folders['root'].resolve()}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
