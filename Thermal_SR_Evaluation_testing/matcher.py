"""
matcher.py
==========
Responsible ONLY for matching filenames across the four input folders
(LR, RGB, HR Ground Truth, HR Predicted).

Input images do NOT share identical filenames. This module discovers the
numeric sample indices present in the LR folder and reconstructs the
corresponding RGB / Ground-Truth / Predicted filenames automatically from
the prefixes/suffixes configured in config.py.

If a dataset uses a different naming convention, only this file needs to
be modified — no other module depends on filename structure directly.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass
class SamplePaths:
    """Container holding the resolved file paths for one dataset sample.

    Attributes
    ----------
    index : int
        Numeric index extracted from the LR filename (e.g. 1 for 'xl1').
    lr_path, rgb_path, gt_path, pred_path : Path
        Full paths to the four images that together form one evaluation
        sample.
    """

    index: int
    lr_path: Path
    rgb_path: Path
    gt_path: Path
    pred_path: Path


def discover_sample_indices(lr_folder: str, lr_prefix: str, lr_extension: str) -> list[int]:
    """Discover all numeric sample indices present in the LR folder.

    The LR folder is treated as the reference set: every filename matching
    '<lr_prefix><index><lr_extension>' contributes one index. The
    corresponding RGB / Ground-Truth / Predicted filenames are built
    automatically from that same index by `build_sample_paths()`.

    Parameters
    ----------
    lr_folder : str
        Path to the LR image folder.
    lr_prefix : str
        Filename prefix used by LR images (e.g. 'xl').
    lr_extension : str
        File extension used by LR images (e.g. '.jpeg').

    Returns
    -------
    list[int]
        Sorted list of discovered numeric sample indices.
    """
    lr_dir = Path(lr_folder)
    if not lr_dir.exists():
        return []

    pattern = re.compile(rf"^{re.escape(lr_prefix)}(\d+){re.escape(lr_extension)}$", re.IGNORECASE)
    indices = []
    for file in lr_dir.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                indices.append(int(match.group(1)))

    return sorted(set(indices))


def build_sample_paths(index: int) -> SamplePaths:
    """Construct the four expected file paths for a given sample index.

    Filenames are built using the prefixes/suffixes/extensions defined in
    config.py, e.g. index=1 -> LR='xl1.jpeg', GT='xl1_T.jpeg',
    RGB='xgb1.jpeg', Prediction='xp1.png'.

    Parameters
    ----------
    index : int
        Numeric sample index (e.g. 1, 2, 3, ...).

    Returns
    -------
    SamplePaths
        Dataclass holding the resolved LR / RGB / GT / Predicted paths.
    """
    lr_path = Path(config.LR_FOLDER) / f"{config.LR_PREFIX}{index}{config.LR_EXTENSION}"
    rgb_path = Path(config.RGB_FOLDER) / f"{config.RGB_PREFIX}{index}{config.RGB_EXTENSION}"
    gt_path = (
        Path(config.HR_GT_FOLDER)
        / f"{config.HR_GT_PREFIX}{index}{config.HR_GT_SUFFIX}{config.HR_GT_EXTENSION}"
    )
    pred_path = Path(config.PRED_FOLDER) / f"{config.PRED_PREFIX}{index}{config.PRED_EXTENSION}"

    return SamplePaths(index=index, lr_path=lr_path, rgb_path=rgb_path, gt_path=gt_path, pred_path=pred_path)
