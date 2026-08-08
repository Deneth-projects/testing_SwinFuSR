from skimage.metrics import peak_signal_noise_ratio

import config


def compute_psnr(ground_truth, prediction):
    return float(peak_signal_noise_ratio(ground_truth, prediction, data_range=config.DATA_RANGE))
