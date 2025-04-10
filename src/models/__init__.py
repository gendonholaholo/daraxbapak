from pathlib import Path
from .config import settings

# Export model paths
MODEL_PATH = settings.MODEL_DIR / "interviewer_transformer.pth"
VOCAB_PATH = settings.MODEL_DIR / "vocab.json"

__all__ = ['MODEL_PATH', 'VOCAB_PATH'] 