"""Data loading, normalisation and the PyTorch Dataset.

Filled in once the Kaggle data format is known (see notebooks/01_data_exploration.ipynb).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_DIR = REPO_ROOT / "data"
KAGGLE_INPUT_DIR = Path("/kaggle/input")


def get_data_dir() -> Path:
    """Return the competition data directory, locally or on Kaggle.

    On Kaggle the attached competition data lives under /kaggle/input/<competition-slug>/.
    Locally, unzip the competition data into ./data/.
    """
    if KAGGLE_INPUT_DIR.exists():
        candidates = [p for p in KAGGLE_INPUT_DIR.iterdir() if p.is_dir()]
        if len(candidates) == 1:
            return candidates[0]
        override = os.environ.get("COS711_DATA_DIR")
        if override:
            return Path(override)
        raise RuntimeError(
            f"Multiple inputs attached {candidates}; set COS711_DATA_DIR to the competition folder."
        )
    return LOCAL_DATA_DIR
