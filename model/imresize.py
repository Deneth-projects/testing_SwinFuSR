"""
MATLAB-style bicubic image resize (numpy), extracted verbatim (logic-wise) from
the original SwinFuSR / KAIR repo's utils/utils_image.py.

This is used to reproduce the exact pre-processing used to train/evaluate
SwinFuSR: the low-resolution thermal image is upsampled by the scale factor
(x8) using this MATLAB-compatible bicubic kernel *before* being fed into the
network (the network itself does not upsample -- it only refines an
already-upsampled image, see SwinFuSR paper Fig. 1 "Bicubic upsampling").
"""
import math
import numpy as np
import torch


def cubic(x):
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (((absx > 1) * (absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel_width, antialiasing):
    if (scale < 1) and antialiasing:
        kernel_width = kernel_width / scale

    x = torch.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = torch.floor(u - kernel_width / 2)
    P = math.ceil(kernel_width) + 2

    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(1, P).expand(out_length, P)
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices

    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)

    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize_np(img, scale, antialiasing=True):
    """
    input:  img -- Numpy array, HxW or HxWxC, any numeric range (e.g. 0-255 uint8
                    values passed in as float, or 0-1 float)
    output: HxW or HxWxC numpy array (float), same numeric range as input, resized
            by `scale` using MATLAB-compatible bicubic interpolation.
    """
    img = torch.from_numpy(img.astype(np.float32))
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_H, out_W = math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4

    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel_width, antialiasing)

    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch.index_select(0, inv_idx))

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch.index_select(0, inv_idx))

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width_h = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(in_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width_h, :, j].transpose(0, 1).mv(weights_H[i])

    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch.index_select(1, inv_idx))

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch.index_select(1, inv_idx))

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width_w = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(in_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width_w, j].mv(weights_W[i])

    if need_squeeze:
        out_2.squeeze_(2)

    return out_2.numpy()
