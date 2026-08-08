"""
Pairs LR / RGB / HR ground-truth / predicted files belonging to the same
sample index, e.g. LR/xl1, HR_ground_truth/xl1_T, RGB/xgb1, PHR/xp1.

Only ground truth + predicted are required (that's all that's quantitatively
compared); LR/RGB are attached when present.
"""
import glob
import os
import re
from dataclasses import dataclass
from typing import Optional

import config


@dataclass
class Sample:
    pair_id: str
    index: int
    lr_path: Optional[str]
    rgb_path: Optional[str]
    hr_gt_path: str
    pred_path: str


def find_file(folder, prefix, index, suffix):
    stem = f"{prefix}{index}{suffix}"
    for ext in config.IMAGE_EXTENSIONS:
        candidate = os.path.join(folder, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    matches = sorted(glob.glob(os.path.join(folder, stem + ".*")))
    return matches[0] if matches else None


def extract_indices(folder, prefix, suffix):
    if not os.path.isdir(folder):
        return set()
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}\.[A-Za-z0-9]+$")
    return {int(m.group(1)) for f in os.listdir(folder) if (m := pattern.match(f))}


def discover_samples():
    gt_indices = extract_indices(config.HR_GT_FOLDER, config.HR_GT_PREFIX, config.HR_GT_SUFFIX)
    pred_indices = extract_indices(config.PRED_FOLDER, config.PRED_PREFIX, config.PRED_SUFFIX)
    common_indices = sorted(gt_indices & pred_indices)

    samples = []
    for idx in common_indices:
        samples.append(Sample(
            pair_id=f"pair{idx}",
            index=idx,
            lr_path=find_file(config.LR_FOLDER, config.LR_PREFIX, idx, config.LR_SUFFIX),
            rgb_path=find_file(config.RGB_FOLDER, config.RGB_PREFIX, idx, config.RGB_SUFFIX),
            hr_gt_path=find_file(config.HR_GT_FOLDER, config.HR_GT_PREFIX, idx, config.HR_GT_SUFFIX),
            pred_path=find_file(config.PRED_FOLDER, config.PRED_PREFIX, idx, config.PRED_SUFFIX),
        ))
    return samples
