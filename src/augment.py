"""Training-time augmentations for preprocessed images and their label maps.
Ranges are calibrated against real train + test images in notebooks/03_augmentation.ipynb."""
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import disk_mask


@dataclass
class AugConfig:
    rotate: bool = True
    flip: bool = True
    contrast: tuple[float, float] = (0.65, 1.35)
    haze_prob: float = 0.3
    haze_std: tuple[float, float] = (0.005, 0.028)
    haze_grid: int = 4                      # control points per side of the smooth haze field
    blur_prob: float = 0.25
    blur_sigma: tuple[float, float] = (0.4, 0.65)
    sharpen_prob: float = 0.15
    sharpen_amount: tuple[float, float] = (0.3, 0.8)
    limb_cut: float = 0.985                 # must match preprocess()


NO_AUG = AugConfig(rotate=False, flip=False, contrast=(1.0, 1.0), haze_prob=0.0, blur_prob=0.0, sharpen_prob=0.0)


class Augmenter:
    def __init__(self, cfg: AugConfig = AugConfig(), seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray, labels: list[np.ndarray] = (),
                 disk: tuple[float, float, float] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
        """Augment one image and its label maps (int maps: nearest, float maps: linear).
        disk: (cx, cy, radius) in image pixels; defaults to the typical GONG disk (centre, 0.4425 × width)."""
        c, rng = self.cfg, self.rng
        h, w = image.shape
        cx, cy, r = disk if disk is not None else (w / 2 - 0.5, h / 2 - 0.5, 0.4425 * w)
        image = image.astype(np.float32, copy=True)
        labels = [l.copy() for l in labels]

        # --- geometric: rotation about the disk centre, then flips (applied identically to labels)
        if c.rotate or c.flip:
            M = cv2.getRotationMatrix2D((cx, cy), rng.uniform(0, 360) if c.rotate else 0.0, 1.0)
            if c.flip and rng.random() < 0.5:   # horizontal flip about the disk centre
                M = np.array([[-1, 0, 2 * cx], [0, 1, 0]], np.float64) @ np.vstack([M, [0, 0, 1]])
            if c.flip and rng.random() < 0.5:   # vertical flip about the disk centre
                M = np.array([[1, 0, 0], [0, -1, 2 * cy]], np.float64) @ np.vstack([M, [0, 0, 1]])
            # Lanczos: bilinear blurs the image (sharpness ×0.80, notebook 03)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4, borderValue=1.0)
            labels = [self._warp_label(l, M) for l in labels]

        # --- photometric, on the disk only
        on_disk = disk_mask(image.shape, cx, cy, r, c.limb_cut)
        if c.haze_prob and rng.random() < c.haze_prob:
            image *= 1.0 + self._smooth_field(h, w, rng.uniform(*c.haze_std))
        a = rng.uniform(*c.contrast)
        image = 1.0 + (image - 1.0) * a
        if c.blur_prob and rng.random() < c.blur_prob:
            image = cv2.GaussianBlur(image, (0, 0), rng.uniform(*c.blur_sigma))
        elif c.sharpen_prob and rng.random() < c.sharpen_prob:
            image = image + rng.uniform(*c.sharpen_amount) * (image - cv2.GaussianBlur(image, (0, 0), 1.0))

        image[~on_disk] = 1.0
        return np.clip(image, 0, 2).astype(np.float32), labels

    def _smooth_field(self, h: int, w: int, std: float) -> np.ndarray:
        g = self.cfg.haze_grid
        coarse = self.rng.standard_normal((g, g)).astype(np.float32)
        field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        return field / (field.std() + 1e-6) * std

    @staticmethod
    def _warp_label(label: np.ndarray, M: np.ndarray) -> np.ndarray:
        h, w = label.shape
        if np.issubdtype(label.dtype, np.integer):
            return cv2.warpAffine(label.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderValue=0).astype(label.dtype)
        return cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
