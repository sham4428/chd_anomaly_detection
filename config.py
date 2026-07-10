import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("CHD_DATA_ROOT", BASE_DIR / "data"))

# Dataset paths. Environment variables can override the local defaults.
PTBXL_PATH = str(Path(os.getenv("PTBXL_PATH", DATA_ROOT / "ptbxl")))
CIRCOR_PATH = str(Path(os.getenv("CIRCOR_PATH", DATA_ROOT / "circor")))
CHD_CXR_PATH = str(Path(os.getenv("CHD_CXR_PATH", DATA_ROOT / "chd_xray")))

# Preprocessing output directory.
OUTPUT_DIR = str(Path(os.getenv("CHD_OUTPUT_DIR", BASE_DIR / "data")))

# Training parameters.
BATCH_SIZE = int(os.getenv("CHD_BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("CHD_EPOCHS", "50"))
LEARNING_RATE = float(os.getenv("CHD_LEARNING_RATE", "1e-4"))
DEVICE = os.getenv("CHD_DEVICE", "cuda")
