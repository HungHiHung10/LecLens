from pathlib import Path
import os
import logging
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CACHE_DIR = DATA_DIR / "cache"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
KEYFRAMES_DIR = DATA_DIR / "keyframes"
UPLOADS_DIR = DATA_DIR / "uploads"
VIDEOS_DIR = BASE_DIR / "videos"

# Ensure all runtime directories exist
for folder in [CACHE_DIR, ANNOTATIONS_DIR, SAMPLES_DIR, KEYFRAMES_DIR, UPLOADS_DIR, VIDEOS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# CPU Thread Optimization for Intel Core i5-1135G7 (4 Cores / 8 Threads)
torch.set_num_threads(4)

# Available Embedding Models (Optimized for CPU & pre-cached on machine)
AVAILABLE_MODELS = {
    "vietnamese-sbert": {
        "name": "keepitreal/vietnamese-sbert",
        "label": "Vietnamese SBERT (Tiếng Việt tối ưu)",
        "dim": 768,
        "lang": "vi"
    },
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "label": "MiniLM-L6-v2 (Tiếng Anh & Siêu nhẹ)",
        "dim": 384,
        "lang": "en"
    }
}

DEFAULT_MODEL_KEY = "vietnamese-sbert"

# Default Chunking & Grounding parameters
DEFAULT_WINDOW_SECONDS = 30.0   # 30-second window
DEFAULT_STRIDE_SECONDS = 15.0   # 15-second stride (50% overlap)
DEFAULT_HYBRID_ALPHA = 0.7      # 0.7 Semantic + 0.3 BM25 Keyword
DEFAULT_TOP_K = 3               # Top 3 results

# Vision & Keyframe parameters
DEFAULT_KEYFRAME_INTERVAL_SEC = 25.0
DEFAULT_SCENE_DIFF_THRESHOLD = 0.10
DEFAULT_MAX_KEYFRAMES = 50

# Centralized Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_logger(name: str) -> logging.Logger:
    """Returns a standardized logger for LecLens components."""
    return logging.getLogger(f"leclens.{name}")
