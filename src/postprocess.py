import cv2
import numpy as np

from .preprocess import disk_mask


def components(fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of a boolean map: (label map, area per label; label 0 = background)."""
    _, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    return lab, stats[:, cv2.CC_STAT_AREA]


def drop_small(lab: np.ndarray, areas: np.ndarray, min_area: int) -> np.ndarray:
    """Remove components smaller than min_area px and renumber the rest 1..n."""
    keep = np.zeros(len(areas), dtype=np.int32)
    ids = np.flatnonzero(areas[1:] >= min_area) + 1
    keep[ids] = np.arange(1, len(ids) + 1)
    return keep[lab]


def upsample(prob: np.ndarray, size: int = 2048) -> np.ndarray:
    prob = prob.astype(np.float32)
    return prob if prob.shape[0] == size else cv2.resize(prob, (size, size), interpolation=cv2.INTER_LINEAR)


def to_instances(prob: np.ndarray, threshold: float = 0.5, min_area: int = 0,
                 disk: tuple[float, float, float] | None = None, size: int = 2048,
                 limb_cut: float = 0.985) -> np.ndarray:
    """Instance map at `size` px from a probability map at any resolution. disk: full-resolution geometry."""
    p = upsample(prob, size)
    fg = p > threshold
    if disk is not None:
        fg &= disk_mask(fg.shape, *disk, limb_cut)
    return drop_small(*components(fg), min_area)
