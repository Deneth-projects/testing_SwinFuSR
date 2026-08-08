"""
image_loader.py
================
Responsible ONLY for opening images, checking for corruption, checking
dimensions, converting to grayscale, resizing, and returning image arrays.

No metric calculation, no plotting, no exporting.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from matcher import SamplePaths, build_sample_paths


def load_grayscale_image(
    path: Path, target_size: Optional[tuple[int, int]] = None
) -> Optional[np.ndarray]:
    """Load an image from disk, convert it to grayscale, and optionally resize.

    Parameters
    ----------
    path : Path
        Path to the image file.
    target_size : tuple[int, int], optional
        (width, height) to resize the image to. If None, the image is
        returned at its native resolution. Resizing only happens when
        config.RESIZE_IF_NEEDED is True.

    Returns
    -------
    np.ndarray or None
        Grayscale image as a uint8 numpy array (H, W), or None if the
        image could not be loaded.
    """
    try:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None

        gray = _to_grayscale(raw)

        # Normalize to 8-bit range consistently regardless of source bit depth.
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if target_size is not None and config.RESIZE_IF_NEEDED:
            current_size = (gray.shape[1], gray.shape[0])
            if current_size != target_size:
                gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_CUBIC)

        return gray

    except Exception:
        return None


def _to_grayscale(raw: np.ndarray) -> np.ndarray:
    """Convert an image array of arbitrary channel layout to single-channel
    grayscale.

    Parameters
    ----------
    raw : np.ndarray
        Image as loaded by cv2.imread (2D grayscale, 3-channel BGR, or
        4-channel BGRA).

    Returns
    -------
    np.ndarray
        Single-channel grayscale image.
    """
    if raw.ndim == 2:
        return raw
    if raw.ndim == 3 and raw.shape[2] == 3:
        return cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    if raw.ndim == 3 and raw.shape[2] == 4:
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
    return raw[:, :, 0]


def is_image_readable(path: Path) -> bool:
    """Check whether a file exists and can be successfully decoded as an image.

    Parameters
    ----------
    path : Path
        Path to the candidate image file.

    Returns
    -------
    bool
        True if the file exists and cv2 can decode it, False otherwise.
    """
    if not path.exists():
        return False
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return img is not None
    except Exception:
        return False


def get_image_size(path: Path) -> Optional[tuple[int, int]]:
    """Return the (width, height) of an image file, or None if unreadable.

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    tuple[int, int] or None
        (width, height) in pixels, or None if the image could not be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return (img.shape[1], img.shape[0])


def verify_dataset(indices: list[int], logger: logging.Logger) -> list[SamplePaths]:
    """Verify that every required image exists and can be opened.

    For each candidate sample index, this function checks:
      1. All four required files exist on disk.
      2. All four files can actually be decoded as images (i.e. are not
         corrupted).
      3. Ground-Truth and Prediction sizes are compatible (either matching,
         or resizing is enabled in config.py).

    Any sample failing verification is skipped (never raises), and a
    message of the form "File {filename} cannot be opened. Skipping..."
    is printed and logged, so that a single damaged image never halts the
    whole evaluation.

    Parameters
    ----------
    indices : list[int]
        Candidate sample indices discovered from the LR folder.
    logger : logging.Logger
        Logger used to record verification results.

    Returns
    -------
    list[SamplePaths]
        The subset of samples that passed verification and are safe to
        evaluate.
    """
    valid_samples: list[SamplePaths] = []
    logger.info(f"Starting dataset verification for {len(indices)} candidate sample(s).")

    for index in indices:
        sample = build_sample_paths(index)
        if not _all_files_exist(sample, logger):
            continue
        if not _all_files_readable(sample, logger):
            continue
        if not _sizes_compatible(sample, logger):
            continue

        valid_samples.append(sample)
        logger.info(f"Sample {index}: verification passed.")

    logger.info(
        f"Dataset verification complete. {len(valid_samples)} / {len(indices)} sample(s) valid."
    )
    return valid_samples


def _all_files_exist(sample: SamplePaths, logger: logging.Logger) -> bool:
    """Check that all four required files for a sample exist on disk."""
    required_files = [sample.lr_path, sample.rgb_path, sample.gt_path, sample.pred_path]
    missing = [str(f) for f in required_files if not f.exists()]
    for m in missing:
        print(f"File {m} cannot be opened. Skipping...")
        logger.warning(f"Missing file for sample {sample.index}: {m}. Skipping sample.")
    return len(missing) == 0


def _all_files_readable(sample: SamplePaths, logger: logging.Logger) -> bool:
    """Check that all four required files for a sample can be decoded."""
    required_files = [sample.lr_path, sample.rgb_path, sample.gt_path, sample.pred_path]
    for f in required_files:
        if not is_image_readable(f):
            print(f"File {f.name} cannot be opened. Skipping...")
            logger.warning(f"Unreadable file for sample {sample.index}: {f}. Skipping sample.")
            return False
    return True


def _sizes_compatible(sample: SamplePaths, logger: logging.Logger) -> bool:
    """Check that Ground-Truth and Prediction sizes are compatible, given
    the RESIZE_IF_NEEDED configuration flag."""
    gt_size = get_image_size(sample.gt_path)
    pred_size = get_image_size(sample.pred_path)

    if gt_size == pred_size:
        return True

    if config.RESIZE_IF_NEEDED:
        logger.info(
            f"Sample {sample.index}: GT size {gt_size} != Prediction size {pred_size}. "
            f"Images will be resized to {config.INPUT_IMAGE_SIZE} before metric computation."
        )
        return True

    print(f"File {sample.pred_path.name} size mismatch with ground truth. Skipping...")
    logger.warning(
        f"Sample {sample.index}: size mismatch (GT={gt_size}, Pred={pred_size}) and "
        f"RESIZE_IF_NEEDED=False. Skipping sample."
    )
    return False
