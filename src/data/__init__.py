from pathlib import Path
from .config import settings

# Export data paths
INTERVIEW_DATA_PATH = settings.DATA_DIR / "interview_data.jsonl"

__all__ = ['INTERVIEW_DATA_PATH'] 