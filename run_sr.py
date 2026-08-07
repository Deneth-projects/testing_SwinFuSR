#!/usr/bin/env python3
"""
SwinFuSR - CPU inference pipeline
==================================
RGB-guided thermal image super-resolution (x8), rebuilt from the original
SwinFuSR paper (arXiv:2404.14533) and GitHub repo (VisionICLab/SwinFuSR) to
run entirely on CPU, with no GPU / distributed-training / wandb / dataloader
dependencies.

USAGE
-----
    python run_sr.py

It automatically:
  1. Scans the `LR/`  folder for low-resolution thermal images  (xl1, xl2, ...)
  2. Scans the `RGB/` folder for the matching RGB guide images  (xgb1, xgb2, ...)
  3. Pairs them up by their trailing number (xl3 <-> xgb3, etc.)
  4. Runs each pair through the pretrained SwinFuSR model (weights/robust0x.pth) x means the verison we use
  5. Saves the predicted high-resolution thermal image to `PHR/` as xp<N>.png

See README.md for full details on accepted image formats/ranges.
"""
import os
import re
import sys
import glob
import time
import argparse

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.network_swinfusionSR import SwinFusionSR
from model.imresize import imresize_np


# --------------------------------------------------------------------------------
# Fixed architecture configuration -- this MUST match the pretrained checkpoint
# (weights/robust0x.pth). These values were verified against the checkpoint's
# tensor shapes and against options/test_swinFuSR.json in the original repo.
# --------------------------------------------------------------------------------
MODEL_CONFIG = dict(
    img_size=64,
    patch_size=1,
    in_chans=1,          # both the thermal branch and the RGB-luminance branch use 1 channel
    embed_dim=60,
    Ex_depths=[4],
    Fusion_depths=[2, 2],
    Re_depths=[4],
    Ex_num_heads=[6],
    Fusion_num_heads=[6, 6],
    Re_num_heads=[6],
    window_size=8,
    mlp_ratio=2,
    upscale=1,           # the network itself does NOT upscale; see SCALE_FACTOR below
    img_range=1.0,
    upsampler='',
    resi_connection='1conv',
    weights = 'robust05'
)

SCALE_FACTOR = 8         # actual super-resolution factor (x8), applied via bicubic
                          # pre-upsampling of the LR thermal image before the network,
                          # exactly as done in the original data/dataset_SR_guided.py

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', MODEL_CONFIG['weights'])
DEFAULT_LR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LR')
DEFAULT_RGB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RGB')
DEFAULT_PHR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PHR')

VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


# --------------------------------------------------------------------------------
# Image I/O helpers (ported from utils/utils_image.py, no external repo needed)
# --------------------------------------------------------------------------------
def imread_uint(path, n_channels=1):
    """Read image from disk. n_channels=1 -> HxWx1 grayscale uint8.
       n_channels=3 -> HxWx3 RGB uint8 (converted from BGR)."""
    if n_channels == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def uint2tensor3(img):
    """HxWxC uint8/float [0,255] -> CxHxW torch float tensor [0,1]."""
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)


def tensor2uint(img):
    """CxHxW (or 1xCxHxW) torch tensor [0,1] -> HxWxC uint8 [0,255]."""
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def imsave_gray(img_uint8, path):
    """Save a HxWx1 (or HxW) uint8 grayscale image."""
    img_uint8 = np.squeeze(img_uint8)
    cv2.imwrite(path, img_uint8)


# --------------------------------------------------------------------------------
# Pairing logic: match files in LR/ and RGB/ by trailing number in filename
# e.g. xl3.png <-> xgb3.png  =>  index 3  =>  output xp3.png
# --------------------------------------------------------------------------------
def index_of(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r'(\d+)\s*$', stem)
    if not m:
        return None
    return int(m.group(1))


def build_pairs(lr_dir, rgb_dir):
    lr_files = [f for f in glob.glob(os.path.join(lr_dir, '*')) if f.lower().endswith(VALID_EXTS)]
    rgb_files = [f for f in glob.glob(os.path.join(rgb_dir, '*')) if f.lower().endswith(VALID_EXTS)]

    lr_map = {}
    for f in lr_files:
        idx = index_of(f)
        if idx is not None:
            lr_map[idx] = f

    rgb_map = {}
    for f in rgb_files:
        idx = index_of(f)
        if idx is not None:
            rgb_map[idx] = f

    common = sorted(set(lr_map.keys()) & set(rgb_map.keys()))
    missing_rgb = sorted(set(lr_map.keys()) - set(rgb_map.keys()))
    missing_lr = sorted(set(rgb_map.keys()) - set(lr_map.keys()))

    if missing_rgb:
        print(f"[WARNING] LR images with no matching RGB guide (skipped): "
              f"{[os.path.basename(lr_map[i]) for i in missing_rgb]}")
    if missing_lr:
        print(f"[WARNING] RGB images with no matching LR thermal (skipped): "
              f"{[os.path.basename(rgb_map[i]) for i in missing_lr]}")

    return [(i, lr_map[i], rgb_map[i]) for i in common]


# --------------------------------------------------------------------------------
# Core preprocessing, matching the original data/dataset_SR_guided.py logic
# --------------------------------------------------------------------------------
def preprocess_pair(lr_path, rgb_path, scale=SCALE_FACTOR):
    # 1) Load LR thermal image as single-channel grayscale, uint8, HxWx1
    img_lr_small = imread_uint(lr_path, n_channels=1)

    # 2) Bicubic-upsample the LR thermal image by the scale factor (MATLAB-style
    #    bicubic, matching the original repo's util.imresize_np). This produces
    #    the "artificial HR IR image" shown in Fig.1 of the paper, which is the
    #    actual input to the network's IR branch (the network itself performs
    #    detail refinement, not the upsampling).
    img_lr_up = np.clip(imresize_np(img_lr_small, scale, True), 0, 255).astype(np.uint8)
    target_h, target_w = img_lr_up.shape[0], img_lr_up.shape[1]

    # 3) Load the RGB guide image
    img_guide = imread_uint(rgb_path, n_channels=3)

    # If the guide image doesn't already match the upsampled thermal resolution,
    # resize it (bicubic) so the two branches align spatially -- this is required
    # by the model design (paired, registered LR-thermal / HR-RGB inputs).
    if img_guide.shape[0] != target_h or img_guide.shape[1] != target_w:
        img_guide = cv2.resize(img_guide, dsize=(target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 4) Convert RGB guide to luminance-only (Y channel of YCrCb), matching the
    #    original dataset code -- the network's guide branch expects 1 channel.
    img_guide_ycc = cv2.cvtColor(img_guide, cv2.COLOR_RGB2YCrCb)
    img_guide_luma = np.expand_dims(img_guide_ycc[:, :, 0], axis=2)

    # 5) Convert both to normalized [0,1] tensors, add batch dimension
    lr_tensor = uint2tensor3(img_lr_up).unsqueeze(0)
    guide_tensor = uint2tensor3(img_guide_luma).unsqueeze(0)

    return lr_tensor, guide_tensor, img_lr_small.shape[:2], (target_h, target_w)


def load_model(weights_path, device):
    model = SwinFusionSR(**MODEL_CONFIG)
    state_dict = torch.load(weights_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'params' in state_dict:
        state_dict = state_dict['params']
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="SwinFuSR CPU x8 guided thermal super-resolution")
    parser.add_argument('--lr_dir', type=str, default=DEFAULT_LR_DIR, help='Folder with LR thermal images (xl1, xl2, ...)')
    parser.add_argument('--rgb_dir', type=str, default=DEFAULT_RGB_DIR, help='Folder with RGB guide images (xgb1, xgb2, ...)')
    parser.add_argument('--out_dir', type=str, default=DEFAULT_PHR_DIR, help='Output folder for predicted HR thermal images (xp1, xp2, ...)')
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS, help='Path to pretrained weights (%s)' %MODEL_CONFIG['weights'])
    parser.add_argument('--scale', type=int, default=SCALE_FACTOR, help='Super-resolution factor (default: 8, matches the pretrained model)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cpu')

    print(f"[INFO] Loading SwinFuSR model on CPU from: {args.weights}")
    t0 = time.time()
    model = load_model(args.weights, device)
    print(f"[INFO] Model loaded in {time.time() - t0:.2f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters)")

    pairs = build_pairs(args.lr_dir, args.rgb_dir)
    if not pairs:
        print(f"[ERROR] No matching LR/RGB pairs found.\n"
              f"        LR folder:  {args.lr_dir}\n"
              f"        RGB folder: {args.rgb_dir}\n"
              f"        Filenames must end in the same number, e.g. xl1.png <-> xgb1.png")
        return

    print(f"[INFO] Found {len(pairs)} paired image(s) to process.")

    total_time = 0.0
    for idx, lr_path, rgb_path in pairs:
        print(f"\n[PROCESSING] index {idx}: LR='{os.path.basename(lr_path)}'  RGB='{os.path.basename(rgb_path)}'")
        lr_tensor, guide_tensor, lr_orig_size, hr_size = preprocess_pair(lr_path, rgb_path, scale=args.scale)
        print(f"             LR original size: {lr_orig_size[1]}x{lr_orig_size[0]} (WxH)  "
              f"-> target HR size: {hr_size[1]}x{hr_size[0]} (WxH)")

        lr_tensor = lr_tensor.to(device)
        guide_tensor = guide_tensor.to(device)

        t0 = time.time()
        with torch.no_grad():
            output = model(lr_tensor, guide_tensor)
        elapsed = time.time() - t0
        total_time += elapsed

        out_img = tensor2uint(output)  # HxWx1 uint8, normalized output re-scaled to [0,255]
        out_path = os.path.join(args.out_dir, f"xp{idx}.png")
        imsave_gray(out_img, out_path)
        print(f"             Saved -> {out_path}   (inference time: {elapsed:.2f}s)")

    print(f"\n[DONE] Processed {len(pairs)} image pair(s). "
          f"Average inference time: {total_time / len(pairs):.2f}s/image (CPU).")


if __name__ == '__main__':
    main()
