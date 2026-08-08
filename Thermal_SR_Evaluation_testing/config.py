"""
config.py
=========
Central, user-editable configuration for the Thermal-Guided Super-Resolution
evaluation project.

This is the ONLY file a user should normally need to edit. It contains
folder paths, filename conventions, processing options and output toggles.
No calculations, image processing, or business logic live here.
"""

import torch

# =============================================================================
# INPUT FOLDERS
# =============================================================================
# Paths to the four input folders. They do NOT need to contain identically
# named files — filenames are reconstructed automatically (see matcher.py)
# from the prefixes / suffixes configured below plus a numeric image index.
LR_FOLDER = "../LR"
RGB_FOLDER = "../RGB"
HR_GT_FOLDER = "../HR_ground_truth"
PRED_FOLDER = "../PHR"

# =============================================================================
# OUTPUT FOLDER
# =============================================================================
OUTPUT_FOLDER = "Evaluation"

# =============================================================================
# FILENAME PREFIXES / SUFFIXES
# =============================================================================
# Example dataset naming convention:
#   LR image             : xl1.jpeg
#   HR Ground Truth image: xl1_T.jpeg
#   RGB guidance image   : xgb1.jpeg
#   Predicted HR image   : xp1.png
LR_PREFIX = "xl"
HR_GT_PREFIX = "xl"
HR_GT_SUFFIX = "_T"
RGB_PREFIX = "xgb"
PRED_PREFIX = "xp"

# =============================================================================
# FILE EXTENSIONS
# =============================================================================
# A single global extension is provided for convenience (IMAGE_EXTENSION),
# but real-world datasets frequently mix formats between folders (e.g. LR /
# RGB / GT stored as .jpeg while predictions are stored as .png). Per-folder
# overrides are therefore also provided; leave them equal to IMAGE_EXTENSION
# if your dataset uses a single consistent format.
IMAGE_EXTENSION = ".jpeg"
LR_EXTENSION = ".jpeg"
RGB_EXTENSION = ".jpeg"
HR_GT_EXTENSION = ".jpeg"
PRED_EXTENSION = ".png"

# =============================================================================
# IMAGE PROCESSING
# =============================================================================
# Target (width, height) that Ground-Truth and Predicted images are resized
# to before metric computation, guaranteeing a fair, consistent comparison
# even if a small size mismatch exists between the two. Set
# RESIZE_IF_NEEDED = False to disable resizing entirely (in that case the
# script requires GT and Prediction to already match exactly).
INPUT_IMAGE_SIZE = (640, 480)  # (width, height)
RESIZE_IF_NEEDED = True

# =============================================================================
# RANKING / SAMPLE SELECTION
# =============================================================================
TOP_BEST_IMAGES = 1
TOP_WORST_IMAGES = 1

# =============================================================================
# SLICE ANALYSIS
# =============================================================================
# Thickness (in pixels) of the centre horizontal / vertical profile slice
# extracted for each Worst sample.
CENTER_SLICE_WIDTH = 5

# =============================================================================
# OUTPUT TOGGLES
# =============================================================================
SAVE_ERROR_MAPS = True
SAVE_GRAPHS = True
SAVE_EXCEL = True
SAVE_CSV = True
SAVE_SLICES = True

# =============================================================================
# COMPUTE DEVICE
# =============================================================================
# Shared torch device used by all torch-based metrics (MS-SSIM, LPIPS).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
