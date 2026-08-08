from skimage.metrics import structural_similarity

import config


def compute_ssim(ground_truth, prediction):
    return float(structural_similarity(ground_truth, prediction, data_range=config.DATA_RANGE))
