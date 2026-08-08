# Thermal Super-Resolution — Evaluation Project

Evaluates already-generated thermal super-resolution predictions against
ground truth (the model itself is not part of this project).

## Setup

```bash
pip install -r requirements.txt
```

`torch` + `lpips` need one-time internet access on first run to download the
pretrained AlexNet backbone (cached afterwards under
`~/.cache/torch/hub/checkpoints/`). If unavailable, LPIPS is reported as
`N/A` and every other metric still runs normally.

## Configure

Edit **`config.py`** only — folder paths, file-naming prefixes/suffixes,
image size, and output toggles all live there.

## Run

```bash
python evaluation.py
```

## Outputs (written to `Evaluation/`)

```
Evaluation/
  CSV/image_metrics.csv            one row per pair
  CSV/summary_statistics.csv       dataset-level Mean/Std/Min/Max
  Excel/evaluation_report.xlsx     3 sheets: Image Metrics, Summary Statistics, Evaluation Info
  Excel/summary_statistics.xlsx    focused summary-only workbook
  Error_Maps/<pair>_error_map.png  colourized |pred - gt| (visualization only)
  Graphs/                          PSNR / SSIM-MSSIM / normalized-error charts
  Logs/evaluation_<timestamp>.log
```

## Scope

Whole-image PSNR, SSIM, MS-SSIM, LPIPS, MAE, RMSE, Maximum Absolute Error
(and their normalized % forms), dataset statistics, CSV/Excel export, and
error-map visualization. Nothing else (no best/worst selection, ROI,
slices, hotspot, edge, frequency analysis, or model inference).
