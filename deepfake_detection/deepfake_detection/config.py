from pathlib import Path 
ROOT = Path(__file__).resolve().parents[2] 
DATA_RAW = ROOT / "data" / "raw" 
DATA_PROCESSED = ROOT / "data" / "processed" 
DATA_CLEANED = ROOT / "data" / "cleaned" 
MODELS_DIR = ROOT / "models" 
IMG_SIZE = 224 
SEED = 42 
