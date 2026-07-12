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

# RAG knowledge base configuration.
KNOWLEDGE_PDF_DIR = str(
    Path(os.getenv("CHD_KNOWLEDGE_PDF_DIR", BASE_DIR / "knowledge_base" / "pdfs"))
)
KNOWLEDGE_PERSIST_DIR = str(
    Path(os.getenv("CHD_KNOWLEDGE_PERSIST_DIR", BASE_DIR / "chroma_db"))
)
KNOWLEDGE_COLLECTION_NAME = os.getenv(
    "CHD_KNOWLEDGE_COLLECTION_NAME", "chd_clinical_guidance"
)
KNOWLEDGE_EMBEDDING_MODEL = os.getenv(
    "CHD_RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)
KNOWLEDGE_TOP_K = int(os.getenv("CHD_RAG_TOP_K", "3"))
