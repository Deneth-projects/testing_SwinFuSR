"""Colour-coded absolute error maps - visualization only. Reuses the same
abs_error array used for MAE/RMSE/Max-Error; only rescaled for display,
never fed back into any metric."""
import os

import cv2
import numpy as np

import config

_COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "TURBO": cv2.COLORMAP_TURBO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "HOT": cv2.COLORMAP_HOT,
}


def save_error_map(abs_error, pair_id, output_folder=None):
    output_folder = output_folder or os.path.join(config.OUTPUT_FOLDER, "Error_Maps")
    os.makedirs(output_folder, exist_ok=True)

    err = np.array(abs_error, dtype=np.float64)
    err_max = float(err.max())
    display = (
        np.zeros_like(err, dtype=np.uint8)
        if err_max <= 1e-12
        else np.clip((err / err_max) * 255.0, 0, 255).astype(np.uint8)
    )

    colormap = _COLORMAPS.get(config.ERROR_MAP_COLORMAP.upper(), cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(display, colormap)

    out_path = os.path.join(output_folder, f"{pair_id}_error_map.png")
    cv2.imwrite(out_path, colored)
    return out_path
