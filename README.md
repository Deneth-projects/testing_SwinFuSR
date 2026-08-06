# SwinFuSR — CPU Inference Pipeline (x8 RGB-Guided Thermal Super-Resolution)

This package is a **from-scratch, CPU-only reimplementation** of the inference
path of **SwinFuSR** (Arnold, Jouvet & Seoud, *"SwinFuSR: an image
fusion-inspired model for RGB-guided thermal image super-resolution"*,
arXiv:2404.14533), built directly from:

- The paper (architecture, pre/post-processing, PSNR/SSIM behaviour)
- The official GitHub repo (`VisionICLab/SwinFuSR`) — network definition,
  dataset pre-processing logic (`data/dataset_SR_guided.py`), and inference
  logic (`test_SwinFuSR.py`, `models/model_plain.py`)
- The provided pretrained checkpoint `robust03.pth` (its tensor shapes were
  inspected and matched exactly against the network configuration below)

The original repo's `test_SwinFuSR.py` only runs on GPU, requires
`torch.distributed`, `wandb`, and a specific dataset folder layout tied to
the authors' cluster. **None of that is needed here.** This package strips
all of that away and gives you a simple, self-contained CPU script with two
input folders and one output folder.

---

## 1. What this model does

Given:
- a **low-resolution thermal (infrared) image**, and
- a **paired, higher-resolution RGB image of the same scene** (used only as
  a structural "guide" — its texture/edges help reconstruct thermal detail
  that the thermal sensor alone cannot resolve),

the model predicts an **8x super-resolved thermal image**: 8 times the
width and 8 times the height of the input LR thermal image, with sharper
edges and more realistic detail than plain bicubic upsampling.

This model **cannot** change the scale factor at runtime — the pretrained
weights (`robust03.pth`) were trained specifically for **x8** super-resolution
(the PBVS 2024 Thermal Image Super-Resolution Challenge, Track 2), so the
scale factor is fixed at **8x** in this pipeline.

---

## 2. Folder structure

```
SwinFuSR_CPU/
├── LR/                     ← put your LOW-RESOLUTION thermal images here (xl1, xl2, xl3, ...)
├── RGB/                    ← put your matching RGB guide images here    (xgb1, xgb2, xgb3, ...)
├── PHR/                    ← predicted HIGH-RESOLUTION thermal images appear here (xp1, xp2, xp3, ...)
├── model/
│   ├── network_swinfusionSR.py   ← SwinFuSR network architecture (CPU-only, no timm dependency)
│   └── imresize.py               ← MATLAB-compatible bicubic resize (matches original repo exactly)
├── weights/
│   └── robust03.pth        ← pretrained weights (as provided by you)
├── run_sr.py                ← main script — run this
├── requirements.txt
└── README.md                ← this file
```

`LR/`, `RGB/`, and `PHR/` already contain **3 dummy example pairs** (derived
from the two sample images you uploaded) so you can immediately test that
everything works before dropping in your own data — see Section 6.

---

## 3. Installation

Requires **Python 3.9+**. No GPU / CUDA needed.

```bash
cd SwinFuSR_CPU
pip install -r requirements.txt
```

This installs the CPU build of PyTorch, NumPy, and OpenCV. (If you already
have a GPU-enabled `torch` installed, this pipeline still runs fine on CPU
— it never calls `.cuda()`.)

---

## 4. Running it

```bash
python run_sr.py
```

That's it. The script will:
1. Load `weights/robust03.pth` onto CPU.
2. Scan `LR/` and `RGB/`, and automatically pair files by the **number at
   the end of the filename** (e.g. `xl3.png` pairs with `xgb3.png`,
   regardless of extension — `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif` all
   work, and you can mix formats between the two folders).
3. Run each pair through the model.
4. Save `PHR/xp<N>.png` for every processed pair `<N>`.

Optional arguments:

```bash
python run_sr.py --lr_dir LR --rgb_dir RGB --out_dir PHR --weights weights/robust03.pth --scale 8
```

(`--scale` should be left at 8 unless you retrain the model for a different
factor — this checkpoint only supports x8.)

**Speed on CPU:** roughly 10–30 seconds per image pair depending on
resolution (tested on a small 40×40 → 320×320 example). This is expected —
the paper itself notes SwinFuSR is transformer-heavy and "resource-hungry"
even on a GPU (1.3s for an 80×56 → 640×448 image on an RTX 3080). CPU
inference is inherently much slower; there is no way around this without a
GPU, since the model contains no fast fallback path.

---

## 5. Input / output image specification

| | Folder | Naming | Channels | Pixel format | Meaning |
|---|---|---|---|---|---|
| **Input 1** | `LR/` | `xl<N>.png/.jpg/...` | 1 (grayscale) | 8-bit, values **0–255** | Low-resolution thermal image |
| **Input 2** | `RGB/` | `xgb<N>.png/.jpg/...` | 3 (RGB) | 8-bit, values **0–255** per channel | Higher-resolution visible-spectrum guide image of the *same* scene |
| **Output** | `PHR/` | `xp<N>.png` | 1 (grayscale) | 8-bit, values **0–255** | Predicted high-resolution thermal image |

### Accepted pixel range
- Standard 8-bit images (the normal range for PNG/JPG/BMP files) — i.e.
  pixel values from **0 to 255** per channel. This is what you get from any
  normal camera export or thermal-camera software export as an 8-bit
  grayscale image.
- **Internally**, the model does *not* run in the 0–255 range: both the
  thermal input and the RGB guide are divided by 255 to the `[0, 1]` range
  before being passed through the network (exactly as in the original
  repo's `uint2tensor3`), the network operates in `[0, 1]`, and the output
  is multiplied back by 255 and rounded (`tensor2uint`) before being saved.
  So the **file you upload should be a normal 0–255 image** — you do not
  need to pre-normalize it to `[0,1]` yourself; the script does that for
  you and writes back a normal 0–255 PNG.
- 16-bit thermal exports are **not** currently supported by this pipeline
  (the original model was trained on 8-bit data); convert to 8-bit first if
  your thermal camera exports 16-bit radiometric data.

### Resolution / scale factor
- **Fixed scale factor: x8.** The output resolution is always
  `(8 × LR width) × (8 × LR height)`.
- The RGB guide image should ideally already be at that target resolution
  (i.e. exactly 8x the LR thermal image's width and height), since in the
  original PBVS dataset the RGB and thermal images are physically
  registered pairs.
- **If your RGB guide's resolution doesn't exactly match `8×LR`,** the
  script automatically bicubic-resizes it to match before feeding it to the
  model, so it won't crash — but for best results, supply an RGB image that
  is genuinely a higher-resolution photo of the same scene as your thermal
  image, ideally already close to the `8×LR` target size.
- There is no strict minimum/maximum size, but very large images will be
  slow on CPU (transformer window-attention over the full image), and very
  small LR crops (smaller than roughly 8×8 pixels) may not carry enough
  information for the model.

### What the model does *not* do
- It does not colorize the thermal image (output stays single-channel
  grayscale, matching how thermal sensors work).
- If the RGB guide is missing/blank for a given pair, the model can still
  run (SwinFuSR was specifically trained to be somewhat robust to a missing
  guide — see Section 5.2/5.3 of the paper), but you'll get noticeably
  worse results than with a real guide image; this pipeline expects a real
  RGB guide for every LR image, since that's the "guided SR" the model was
  optimized for.

---

## 6. Dummy example data (already included)

Three synchronized dummy pairs are pre-loaded so you can test the pipeline
immediately:

| Pair | LR file | RGB file | Output (after running) |
|---|---|---|---|
| 1 | `LR/xl1.png` (40×40) | `RGB/xgb1.png` (320×320) | `PHR/xp1.png` (320×320) |
| 2 | `LR/xl2.png` (30×40) | `RGB/xgb2.png` (240×320) | `PHR/xp2.png` (240×320) |
| 3 | `LR/xl3.png` (50×35) | `RGB/xgb3.png` (400×280) | `PHR/xp3.png` (400×280) |

These were built from the two sample images you provided (one grayscale
thermal-style blob image, one RGB photo), resized/cropped/rotated to create
three distinct demo pairs at different (non-square) resolutions, so you can
confirm the folder-pairing logic and the model both work correctly before
you replace them with your own data. **They are not real registered
thermal/RGB pairs of the same scene** — they exist purely to let you verify
the pipeline runs end-to-end. Delete them and drop in your own images
whenever you're ready (see Section 7).

---

## 7. Using your own images

1. Clear out (or leave, and just add more) the dummy files in `LR/` and
   `RGB/`.
2. Add your low-resolution thermal image(s) to `LR/`, named `xl1.png`,
   `xl2.png`, `xl3.png`, ... (any image extension in the accepted list
   works; the number at the end is what matters for pairing).
3. Add the matching RGB guide image(s) to `RGB/`, named `xgb1.png`,
   `xgb2.png`, `xgb3.png`, ... — **the number must match its LR
   counterpart** (`xl2` pairs with `xgb2`, not with `xgb1` or `xgb3`).
4. Run `python run_sr.py`.
5. Collect your results from `PHR/` as `xp1.png`, `xp2.png`, `xp3.png`, ...

You can add as many numbered pairs as you like — the script automatically
discovers and processes every matched pair in one run, and warns you (but
doesn't crash) about any unmatched files.

---

## 8. Technical details (for the curious / for reproducibility)

### Architecture (must match `robust03.pth` exactly — verified by inspecting
the checkpoint's tensor shapes)

| Parameter | Value |
|---|---|
| Input channels (both branches) | 1 (thermal branch: grayscale; guide branch: RGB → Y-luminance only) |
| Embedding dimension | 60 |
| Extraction module depth (Ex_depths) | [4] |
| Fusion module depth (Fusion_depths) | [2, 2] |
| Reconstruction module depth (Re_depths) | [4] |
| Attention heads | 6 (all modules) |
| Window size | 8 |
| MLP ratio | 2 |
| Network-internal upscale | 1 (the network only *refines*; the x8 upsampling is done via bicubic *before* the network — see below) |
| Total parameters | ≈ 1M (this checkpoint uses the smaller "robust" config from the paper's ablation, N=1, L=2, P=1) |

### Pre-processing pipeline (mirrors `data/dataset_SR_guided.py` exactly)
1. Read LR thermal image as grayscale.
2. Upsample it by 8x using **MATLAB-compatible bicubic interpolation**
   (`model/imresize.py`, a faithful port of the original repo's
   `utils_image.imresize_np`, not plain OpenCV bicubic, since prior work
   uses MATLAB's slightly different antialiased kernel).
3. Read RGB guide image, convert to YCrCb color space, and keep only the
   Y (luminance) channel — this is the actual "guide" signal fed to the
   network, matching the original dataset loader.
4. If needed, resize the guide to match the upsampled thermal image's
   exact resolution.
5. Normalize both images from `[0,255]` to `[0,1]` and convert to tensors.

### Model forward pass
The network takes the **already-bicubic-upsampled** thermal image and the
guide image, extracts features from each via separate Swin-Transformer
branches, fuses them with Attention-guided Cross-domain Fusion (ACF)
blocks, reconstructs the residual detail, and finally **adds the bicubic
image back as a skip connection** (this is why the network's internal
`upscale` is 1 — the real 8x upsampling already happened in
pre-processing, and the network's job is purely to add the missing
high-frequency detail).

### Post-processing
Model output tensor (`[0,1]` range) is clamped, multiplied by 255, rounded,
and saved as an 8-bit grayscale PNG.

### Differences from the original repo (intentional, for CPU + folder-based use)
- Removed all GPU/CUDA, `torch.distributed`, and `wandb` dependencies.
- Removed the `timm` dependency: `DropPath`, `to_2tuple`, and
  `trunc_normal_` are reimplemented locally in
  `model/network_swinfusionSR.py` (only used for training-time
  initialization/regularization — since we only load pretrained weights,
  this has zero effect on inference results).
- Replaced the original PyTorch `DataLoader` / JSON-option / distributed
  test loop with a simple folder-scanning loop (`run_sr.py`) driven by
  matching filename numbers, per your request.
- No ground-truth / PSNR / SSIM evaluation code is included here, since you
  don't have ground-truth HR thermal images for your own data — this is
  purely a prediction pipeline. (If you do have ground truth and want
  PSNR/SSIM numbers computed automatically, that can be added.)

---

## 9. Troubleshooting

- **`ImportError: libGL.so.1: cannot open shared object file`** (common in
  Codespaces / Docker / headless Linux servers) → this happens because the
  standard `opencv-python` package pulls in GUI libraries (`libGL`) that
  headless containers don't have installed. This project's
  `requirements.txt` now installs **`opencv-python-headless`** instead,
  which has no GUI dependency and works identically for everything this
  pipeline needs (reading, resizing, and writing images). Fix:
  ```bash
  pip uninstall opencv-python opencv-python-contrib-python -y   # remove the GUI build if present
  pip install -r requirements.txt                                # installs opencv-python-headless
  ```
  (If you'd rather keep the full `opencv-python` build for some other
  reason, you can instead install the missing system library:
  `sudo apt-get update && sudo apt-get install -y libgl1`.)
- **"No matching LR/RGB pairs found"** → check that your filenames in `LR/`
  and `RGB/` end in the same number (e.g. `thermal_xl7.png` and
  `guide_xgb7.jpg` — both end in `7`, so they will still be paired
  correctly; only the trailing digits matter).
- **Very slow on CPU** → expected; this is a transformer model. Try smaller
  input images if you're just testing the pipeline. For production-scale
  batches, consider running on a machine with a GPU (the network code will
  work on GPU too, simply move `model.to('cuda')` and the input tensors to
  `'cuda'` — CPU vs GPU device placement is the only change needed).
- **`RuntimeError` about tensor shape mismatch when loading weights** →
  make sure `weights/robust03.pth` hasn't been swapped for a different
  checkpoint; `MODEL_CONFIG` in `run_sr.py` is hard-coded to match this
  specific checkpoint's architecture.
