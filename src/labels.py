"""Fusing several annotation sets of one observation: union (any), vote (≥ ceil(n/2)), intersection (all)
or soft labels. Fused instances are connected components of the fused foreground."""
import math

import cv2
import numpy as np

FUSION_METHODS = ("each", "union", "vote")


def vote_threshold(n_sets: int, method: str) -> int:
    if method == "union":
        return 1
    if method == "vote":
        return math.ceil(n_sets / 2)
    if method == "intersection":
        return n_sets
    raise ValueError(f"No vote threshold for method {method!r}")


def fuse_instance_masks(masks: list[np.ndarray], method: str, min_area: int = 20) -> np.ndarray:
    """Fuse instance label maps from several annotators into one instance label map."""
    if len(masks) == 1:
        return masks[0]
    votes = np.sum([m > 0 for m in masks], axis=0)
    fg = (votes >= vote_threshold(len(masks), method)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    keep = np.zeros(n, dtype=np.int32)
    ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    keep[ids] = np.arange(1, len(ids) + 1)
    return keep[lab]


def soft_label(masks: list[np.ndarray]) -> np.ndarray:
    """Per-pixel fraction of annotators that marked the pixel as filament (float32 in [0, 1])."""
    return np.mean([m > 0 for m in masks], axis=0).astype(np.float32)
