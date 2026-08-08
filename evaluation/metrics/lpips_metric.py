"""
LPIPS on the thermal image only (RGB guidance is never used). The grayscale
channel is replicated to 3 channels only to satisfy LPIPS's input format,
then rescaled from [0,1] to LPIPS's own [-1,1] convention - that conversion
stays local to this module.

The pretrained backbone downloads on first use and is cached afterwards.
If unavailable (no internet), compute_lpips returns None instead of raising,
so the rest of the evaluation still runs.
"""
import threading

import torch

import config

_lock = threading.Lock()
_model = None
_load_failed = False
_load_error = None


def _get_model():
    global _model, _load_failed, _load_error

    if _model is not None or _load_failed:
        return _model

    with _lock:
        if _model is not None or _load_failed:
            return _model
        try:
            import lpips
            model = lpips.LPIPS(net=config.LPIPS_NET, verbose=False)
            if config.LPIPS_USE_GPU and torch.cuda.is_available():
                model = model.cuda()
            _model = model.eval()
        except Exception as exc:
            _load_failed = True
            _load_error = exc
    return _model


def lpips_available():
    _get_model()
    return not _load_failed


def lpips_load_error():
    return _load_error


def _to_lpips_tensor(gray_image_01):
    tensor = torch.from_numpy(gray_image_01).float().unsqueeze(0).unsqueeze(0)
    tensor = tensor.repeat(1, 3, 1, 1)
    return tensor * 2.0 - 1.0


def compute_lpips(ground_truth, prediction):
    model = _get_model()
    if model is None:
        return None

    gt_tensor = _to_lpips_tensor(ground_truth)
    pred_tensor = _to_lpips_tensor(prediction)
    if config.LPIPS_USE_GPU and torch.cuda.is_available():
        gt_tensor, pred_tensor = gt_tensor.cuda(), pred_tensor.cuda()

    with torch.no_grad():
        distance = model(gt_tensor, pred_tensor)
    return float(distance.item())
