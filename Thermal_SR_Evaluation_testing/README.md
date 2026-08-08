# Thermal-Guided Super-Resolution — Evaluation Toolkit

A modular, standalone Python toolkit for evaluating already-generated
thermal super-resolution predictions against high-resolution ground truth
images. It is designed to produce the quantitative baseline used in a
Master's thesis / research report, and to remain easy to extend as the
project grows (fine-tuning, transfer learning, multiple models).

This toolkit performs **evaluation only** — it assumes model inference has
already been completed and the predicted images already exist on disk. It
does not depend on any model repository (e.g. SwinFuSR).

---

## 1. Project Purpose

Given four folders of images per dataset —

- `LR/` — low-resolution thermal input
- `RGB/` — RGB guidance image
- `HR_ground_truth/` — high-resolution thermal ground truth
- `HR_predicted/` (here: `PHR/`) — the model's predicted high-resolution thermal image

the toolkit automatically pairs samples by index, verifies the dataset,
computes a full set of full-reference image-quality metrics, ranks
samples, exports the best/worst cases with error maps and centre-slice
profiles, generates summary statistics and histograms, and writes complete
logs — all without requiring any code changes beyond `config.py`.

---

## 2. Folder Structure

```
Thermal_SR_Evaluation/
│
├── evaluation.py          # Entry point — main execution flow only
├── config.py               # ALL user-editable configuration
├── image_loader.py         # Opening / verifying / resizing images
├── matcher.py               # Filename pairing across the 4 input folders
├── logger.py                 # Central logging utility
├── helpers.py                 # Small reusable utilities (folders, timing, export)
├── requirements.txt
├── README.md
│
├── metrics/                    # One file per metric — nothing else
│   ├── psnr.py
│   ├── ssim.py
│   ├── ms_ssim.py
│   ├── lpips_metric.py
│   ├── mae.py
│   ├── rmse.py
│   ├── max_error.py
│   └── normalized_error.py
│
├── analysis/                    # Evaluation logic — no plotting/exporting
│   ├── whole_image.py            # Main per-image evaluation loop
│   ├── slice_analysis.py         # Centre-slice analysis (Worst samples only)
│   ├── ranking.py                # Best/Worst selection (RMSE primary, SSIM secondary)
│   └── statistics.py             # Mean / Median / Std / Min / Max
│
├── visualization/                 # Image & figure generation
│   ├── error_map.py                # Absolute error maps (non-normalized)
│   ├── histograms.py                # PSNR / SSIM / RMSE / MAE histograms
│   └── graphs.py                     # Slice intensity-profile plots, future figures
│
├── exporters/                     # Writing results to disk
│   ├── csv_export.py
│   └── excel_export.py
│
└── outputs/                       # Suggested location for your input dataset /
                                    # a place to keep local run outputs (gitignored)
```

Each module has exactly one responsibility. If you need to change how
files are paired, edit `matcher.py`. If you need a new metric, add one
file to `metrics/`. Nothing else needs to change.

---

## 3. Dependencies

- Python 3.10+
- numpy, opencv-python, pandas, matplotlib
- torch, torchvision, torchmetrics, lpips
- scikit-image, openpyxl, tqdm

Install everything with:

```bash
pip install -r requirements.txt
```

> **Note on LPIPS:** the first time LPIPS runs, `torchmetrics` downloads a
> small pretrained AlexNet backbone from the internet. If no internet
> connection is available, LPIPS is disabled automatically — a warning is
> logged once, and LPIPS is reported as `NaN` for every sample. No other
> part of the evaluation is affected.

---

## 4. Installation

```bash
git clone <this-project>
cd Thermal_SR_Evaluation
pip install -r requirements.txt
```

Place your dataset so that the four input folders referenced in
`config.py` exist and contain your LR / RGB / Ground-Truth / Predicted
images (see Section 6 for the naming convention).

---

## 5. Configuration

Open **`config.py`** — this is the only file you should normally need to
edit.

| Setting | Description |
|---|---|
| `LR_FOLDER`, `RGB_FOLDER`, `HR_GT_FOLDER`, `PRED_FOLDER` | Paths to the four input folders |
| `OUTPUT_FOLDER` | Where the `Evaluation/` results tree is created |
| `LR_PREFIX`, `HR_GT_PREFIX`, `HR_GT_SUFFIX`, `RGB_PREFIX`, `PRED_PREFIX` | Filename prefixes/suffix used to pair samples |
| `LR_EXTENSION`, `RGB_EXTENSION`, `HR_GT_EXTENSION`, `PRED_EXTENSION` | Per-folder file extensions |
| `INPUT_IMAGE_SIZE`, `RESIZE_IF_NEEDED` | Target (width, height) GT/Prediction are resized to before scoring |
| `TOP_BEST_IMAGES`, `TOP_WORST_IMAGES` | How many samples to export as Best/Worst |
| `CENTER_SLICE_WIDTH` | Thickness (pixels) of the centre-slice profile |
| `SAVE_ERROR_MAPS`, `SAVE_GRAPHS`, `SAVE_EXCEL`, `SAVE_CSV`, `SAVE_SLICES` | Output toggles |

---

## 6. Filename Convention

Input images across the four folders do **not** need identical filenames.
Given a numeric sample index `N`, the expected filenames are:

```
LR                : {LR_PREFIX}{N}{LR_EXTENSION}            e.g. xl1.jpeg
HR Ground Truth   : {HR_GT_PREFIX}{N}{HR_GT_SUFFIX}{HR_GT_EXTENSION}   e.g. xl1_T.jpeg
RGB               : {RGB_PREFIX}{N}{RGB_EXTENSION}           e.g. xgb1.jpeg
HR Predicted      : {PRED_PREFIX}{N}{PRED_EXTENSION}         e.g. xp1.png
```

Sample indices are discovered automatically from whatever numeric indices
exist in `LR_FOLDER`. To support a different naming convention entirely,
edit **`matcher.py`** only — no other module needs to change.

---

## 7. Running the Project

```bash
python evaluation.py
```

The script will:

1. Discover and verify the dataset (skipping any damaged sample with a
   clear warning — a single bad image never stops the run).
2. Compute PSNR, SSIM, MS-SSIM, LPIPS, MAE, RMSE, Maximum Error and
   Normalized Error (%) for every valid sample, with a live progress bar.
3. Save per-image results and summary statistics to CSV/Excel.
4. Rank samples (lowest RMSE, then highest SSIM) and export the
   configured number of Best/Worst samples with error maps.
5. Run centre-slice analysis on the Worst samples only.
6. Generate PSNR/SSIM/RMSE/MAE histograms.
7. Write a complete timestamped log to `Evaluation/Logs/evaluation.log`.

---

## 8. Expected Outputs

```
Evaluation/
├── CSV/
│   ├── image_metrics.csv
│   ├── summary_statistics.csv
│   └── slice_metrics.csv
├── Excel/
│   ├── image_metrics.xlsx
│   ├── summary_statistics.xlsx
│   └── slice_metrics.xlsx
├── Best_Samples/
│   └── rank_1_sample_<N>/        (RGB, LR, ground truth, prediction, error map)
├── Worst_Samples/
│   └── rank_1_sample_<N>/
├── Error_Maps/
│   └── error_map_sample_<N>.png  (every evaluated sample)
├── Slices/
│   └── sample_<N>/                (Worst samples only: slice images + profile plot)
├── Graphs/
│   ├── PSNR_histogram.png
│   ├── SSIM_histogram.png
│   ├── RMSE_histogram.png
│   └── MAE_histogram.png
└── Logs/
    └── evaluation.log
```

---

## 9. How to Add a New Metric

1. Create a new file in `metrics/`, e.g. `metrics/niqe.py`, containing a
   single `calculate_niqe(...)` function with the same style (docstring,
   type hints, `NaN`-on-failure) as the existing metric modules.
2. Add it to `metrics/__init__.py`'s imports and `__all__` list.
3. Call it from `analysis/whole_image.py::_compute_metrics_for_pair()` (or
   `analysis/slice_analysis.py` for a slice-level metric) and add it to
   the relevant column-order list.

No other file needs to change.

---

## 10. How to Modify the Filename Convention

All filename construction logic lives in **`matcher.py`**
(`build_sample_paths()`), driven entirely by the prefixes/suffixes in
`config.py`. To support an entirely different naming scheme (e.g. a
different separator or a fixed-width zero-padded index), only
`discover_sample_indices()` and `build_sample_paths()` in `matcher.py`
need to change — every other module consumes the resulting `SamplePaths`
object and never parses filenames itself.

---

## 11. Design Principles

- **One responsibility per file.** Metric calculation, image loading,
  ranking, plotting and exporting are all fully separated.
- **No hidden global state.** `config.py` is the single source of
  configuration; every function receives what it needs as arguments.
- **Never stop on one bad sample.** Verification and per-image evaluation
  both catch and log errors, skip the sample, and continue.
- **Extensible by addition, not modification.** New metrics, new export
  formats, and new figure types are added as new files, not by editing
  existing ones.
