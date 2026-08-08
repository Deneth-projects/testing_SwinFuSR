"""
helpers.py
==========
Small, generic, reusable utilities that do not belong to any single
pipeline stage: output-folder creation, timing, and copying the files of a
selected sample group (Best / Worst) into their destination folders.

Nothing metric-related lives here.
"""

import logging
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import pandas as pd
import torch

import config
from image_loader import load_grayscale_image
from matcher import build_sample_paths
from visualization.error_map import compute_error_map


def create_output_folders(output_root: str) -> dict:
    """Create (or reuse) the full output folder hierarchy.

    Parameters
    ----------
    output_root : str
        Root name/path of the evaluation output directory (config.OUTPUT_FOLDER).

    Returns
    -------
    dict
        Mapping of logical folder names to their Path objects, e.g.
        {'root': ..., 'csv': ..., 'excel': ..., 'best': ..., ...}
    """
    root = Path(output_root)
    folders = {
        "root": root,
        "csv": root / "CSV",
        "excel": root / "Excel",
        "best": root / "Best_Samples",
        "worst": root / "Worst_Samples",
        "error_maps": root / "Error_Maps",
        "graphs": root / "Graphs",
        "slices": root / "Slices",
        "logs": root / "Logs",
    }
    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)
    return folders


@contextmanager
def timer():
    """Context manager that measures wall-clock elapsed time in seconds.

    Usage
    -----
    with timer() as t:
        do_work()
    print(t.elapsed_seconds)

    Yields
    ------
    _Timer
        Object exposing `elapsed_seconds` after the `with` block exits.
    """

    class _Timer:
        elapsed_seconds: float = 0.0

    state = _Timer()
    start = time.perf_counter()
    try:
        yield state
    finally:
        state.elapsed_seconds = time.perf_counter() - start


def grayscale_to_torch_tensor(gray_image, device: torch.device) -> torch.Tensor:
    """Convert a single-channel uint8 grayscale image to a (1, 1, H, W)
    torch tensor in the [0, 1] range, on the requested device.

    Shared by MS-SSIM and other torch-based single-channel metrics to avoid
    duplicating the same conversion logic in every metric module.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale image (H, W), uint8.
    device : torch.device
        Device the resulting tensor should live on.

    Returns
    -------
    torch.Tensor
        Tensor of shape (1, 1, H, W), dtype float32, values in [0, 1].
    """
    tensor = torch.from_numpy(gray_image.astype("float32") / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def export_sample_group(
    samples_df: pd.DataFrame,
    destination_folder: Path,
    logger: logging.Logger,
) -> None:
    """Copy RGB / LR / Ground-Truth / Prediction / Error-Map files for a
    group of selected samples (used for both Best_Samples and Worst_Samples).

    Parameters
    ----------
    samples_df : pd.DataFrame
        Rows of the metrics table corresponding to the selected samples.
        Must contain an 'Index' column identifying each sample.
    destination_folder : Path
        Target folder (Best_Samples or Worst_Samples) to copy files into.
    logger : logging.Logger
        Logger used to record progress and warnings.
    """
    for rank, (_, row) in enumerate(samples_df.iterrows(), start=1):
        index = int(row["Index"])
        sample = build_sample_paths(index)

        sample_dir = destination_folder / f"rank_{rank}_sample_{index}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(sample.rgb_path, sample_dir / f"rgb{sample.rgb_path.suffix}")
            shutil.copy2(sample.lr_path, sample_dir / f"lr{sample.lr_path.suffix}")
            shutil.copy2(sample.gt_path, sample_dir / f"ground_truth{sample.gt_path.suffix}")
            shutil.copy2(sample.pred_path, sample_dir / f"prediction{sample.pred_path.suffix}")

            if config.SAVE_ERROR_MAPS:
                gt_gray = load_grayscale_image(sample.gt_path, target_size=config.INPUT_IMAGE_SIZE)
                pred_gray = load_grayscale_image(sample.pred_path, target_size=config.INPUT_IMAGE_SIZE)
                if gt_gray is not None and pred_gray is not None:
                    error_map = compute_error_map(gt_gray, pred_gray)
                    cv2.imwrite(str(sample_dir / "error_map.png"), error_map)

            logger.info(f"Exported sample {index} (rank {rank}) to {destination_folder.name}.")
        except Exception as exc:
            logger.warning(f"Failed to export sample {index} to {destination_folder.name}: {exc}")
