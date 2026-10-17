"""Data loading for the MAGFiLO Kaggle 2026 data (COCO-format annotations).

Layout of the competition data:
    MAGFiLO_1.0_Kaggle_2026/
        train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
        train/train_images/<YYYYMMDDhhmmss><site>h.jpeg
        test/test_images/<YYYYMMDDhhmmss><site>h.jpeg

One observation (image file) can have several annotation sets: the JSON "images"
entry id is "<set_id>-<observation>", e.g. "050102-20111116063134Lh".
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_DIR = REPO_ROOT / "data"
KAGGLE_INPUT_DIR = Path("/kaggle/input")
ANNOTATION_FILE = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

# GONG network site codes (letter before the trailing "h" in the file name)
SITES = {
    "B": "Big Bear",
    "C": "Cerro Tololo",
    "L": "Learmonth",
    "M": "Mauna Loa",
    "T": "El Teide",
    "U": "Udaipur",
}

CATEGORIES = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}


def get_data_dir() -> Path:
    """Return the dataset root (the folder containing train/ and test/), locally or on Kaggle."""
    override = os.environ.get("COS711_DATA_DIR")
    if override:
        return Path(override)
    base = KAGGLE_INPUT_DIR if KAGGLE_INPUT_DIR.exists() else LOCAL_DATA_DIR
    hits = sorted(base.rglob(ANNOTATION_FILE))
    if not hits:
        raise FileNotFoundError(f"{ANNOTATION_FILE} not found under {base}")
    return hits[0].parent.parent


def parse_observation(file_name: str) -> dict:
    """'20111116063134Lh.jpeg' -> {'obs': '20111116063134Lh', 'time': Timestamp, 'site': 'L'}."""
    stem = Path(file_name).stem
    return {
        "obs": stem,
        "time": pd.to_datetime(stem[:14], format="%Y%m%d%H%M%S"),
        "site": stem[14],
    }


def load_annotations(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training annotations.

    Returns
    -------
    images : one row per annotation set (image_id), with obs / time / site / set_id columns.
    anns   : one row per filament instance, with polygon and spine as (N, 2) float arrays.
    """
    data_dir = data_dir or get_data_dir()
    with open(data_dir / "train" / ANNOTATION_FILE) as f:
        coco = json.load(f)

    images = pd.DataFrame(coco["images"]).rename(columns={"id": "image_id"})
    images["set_id"] = images["image_id"].str.split("-").str[0]
    meta = pd.DataFrame([parse_observation(fn) for fn in images["file_name"]])
    images = pd.concat([images, meta], axis=1)
    images["path"] = [str(data_dir / "train" / "train_images" / fn) for fn in images["file_name"]]

    anns = pd.DataFrame(coco["annotations"])
    anns["polygon"] = [np.asarray(s[0], dtype=np.float32).reshape(-1, 2) for s in anns["segmentation"]]
    anns["spine"] = [np.asarray(s, dtype=np.float32).reshape(-1, 2) for s in anns["spine"]]
    anns["category"] = anns["category_id"].map(CATEGORIES)
    anns = anns.drop(columns=["segmentation"])
    anns = anns.merge(images[["image_id", "obs", "site", "set_id"]], on="image_id", how="left")
    return images, anns


def list_test_images(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or get_data_dir()
    folder = data_dir / "test" / "test_images"
    rows = [{**parse_observation(fn), "file_name": fn, "path": str(folder / fn)}
            for fn in sorted(os.listdir(folder))]
    return pd.DataFrame(rows)


def load_image(path: str | Path) -> np.ndarray:
    """Load an H-alpha image as a 2D uint8 array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read {path}")
    return img


def instance_mask(polygons: list[np.ndarray], shape: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """Rasterise polygons into an instance label map (0 = background, i+1 = i-th polygon).

    `scale` rasterises directly at another resolution (e.g. 0.5 for 1024 px), which keeps thin
    filaments intact better than resizing a full-resolution mask. Where polygons overlap,
    the later one wins.
    """
    shift = 4  # sub-pixel precision for fillPoly
    mask = np.zeros(shape, dtype=np.int32)
    for i, poly in enumerate(polygons, start=1):
        pts = np.round(poly * scale * (1 << shift)).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=i, shift=shift)
    return mask


def instance_mask_for(image_id: str, anns: pd.DataFrame, shape=(2048, 2048), scale: float = 1.0) -> np.ndarray:
    return instance_mask(list(anns.loc[anns["image_id"] == image_id, "polygon"]), shape, scale)
