"""
Multi-Scale SSIM, implemented directly with torch (Gaussian-window SSIM/CS
at each scale, standard Wang et al. 2003 formulation) so the project doesn't
depend on the separate pytorch_msssim package.

Number of scales is reduced automatically for small images, since each
scale halves the resolution and the SSIM window must still fit.
"""
import numpy as np
import torch
import torch.nn.functional as F

import config

_WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
_WIN_SIZE = 11
_WIN_SIGMA = 1.5


def _gaussian_kernel(win_size, sigma):
    coords = torch.arange(win_size, dtype=torch.float64) - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    kernel = g.mm(g.t()).unsqueeze(0).unsqueeze(0)
    return kernel


def _ssim_and_cs(img1, img2, kernel, data_range, k1=0.01, k2=0.03):
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2

    mu1 = F.conv2d(img1, kernel)
    mu2 = F.conv2d(img2, kernel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean(), cs_map.mean()


def _usable_scales(height, width, win_size=_WIN_SIZE, max_scales=len(_WEIGHTS)):
    levels, h, w = 1, height, width
    while True:
        h, w = h // 2, w // 2
        if min(h, w) < win_size:
            break
        levels += 1
    return min(levels, max_scales)


def compute_ms_ssim(ground_truth, prediction):
    height, width = ground_truth.shape
    n_scales = max(1, _usable_scales(height, width))
    weights = _WEIGHTS[:n_scales]
    weights = weights / weights.sum()

    kernel = _gaussian_kernel(_WIN_SIZE, _WIN_SIGMA)
    img1 = torch.from_numpy(ground_truth).double().unsqueeze(0).unsqueeze(0)
    img2 = torch.from_numpy(prediction).double().unsqueeze(0).unsqueeze(0)

    cs_values = []
    ssim_value = None
    with torch.no_grad():
        for level in range(n_scales):
            ssim_value, cs_value = _ssim_and_cs(img1, img2, kernel, config.DATA_RANGE)
            cs_values.append(torch.relu(cs_value))
            if level < n_scales - 1:
                img1 = F.avg_pool2d(img1, kernel_size=2)
                img2 = F.avg_pool2d(img2, kernel_size=2)

        final_term = torch.relu(ssim_value) ** float(weights[-1])
        if len(cs_values) > 1:
            weighted_cs = torch.stack(
                [cs ** float(w) for cs, w in zip(cs_values[:-1], weights[:-1])]
            )
            result = torch.prod(weighted_cs) * final_term
        else:
            result = final_term

    return float(result.item())
