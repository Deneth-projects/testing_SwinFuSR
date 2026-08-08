import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Dataset folders ---
#LR_FOLDER = os.path.join(BASE_DIR, "evaluation", "LR")
LR_FOLDER = "../LR"

#RGB_FOLDER = os.path.join(BASE_DIR, "evaluation", "RGB")
RGB_FOLDER = "../RGB"

#HR_GT_FOLDER = os.path.join(BASE_DIR, "evaluation", "HR_ground_truth")
HR_GT_FOLDER = "../HR_ground_truth"

#PRED_FOLDER = os.path.join(BASE_DIR, "evaluation", "PHR")
PRED_FOLDER = "../PHR"

OUTPUT_FOLDER = os.path.join(BASE_DIR, "Evaluation")

# --- Expected HR resolution (width, height), used for reporting only ---
INPUT_IMAGE_SIZE = (640, 480)

# --- File extensions (searched in this order; folders may mix extensions) ---
IMAGE_EXTENSION = ".jpeg"
IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff")

# --- Naming convention: <prefix><index><suffix> per folder ---
LR_PREFIX, LR_SUFFIX = "xl", ""
HR_GT_PREFIX, HR_GT_SUFFIX = "xl", "_T"
RGB_PREFIX, RGB_SUFFIX = "xgb", ""
PRED_PREFIX, PRED_SUFFIX = "xp", ""

# --- Output toggles ---
SAVE_ERROR_MAPS = True
SAVE_CSV = True
SAVE_EXCEL = True
SAVE_GRAPHS = True

# --- Metric settings ---
DATA_RANGE = 1.0        # all metrics operate on normalized [0, 1] data
DECIMAL_PLACES = 4      # rounding applied only when reporting/exporting

LPIPS_NET = "alex"      # 'alex' | 'vgg' | 'squeeze'
LPIPS_USE_GPU = False

ERROR_MAP_COLORMAP = "JET"  # JET | INFERNO | TURBO | VIRIDIS | HOT
