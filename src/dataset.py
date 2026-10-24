import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import NO_AUG, AugConfig, Augmenter
from .data import instance_mask
from .labels import fuse_instance_masks, soft_label
from .preprocess import disk_mask, load_preprocessed

LABELS = ("each", "union", "vote", "soft")
FULL_RES = 2048


class FilamentDataset(Dataset):
    def __init__(self, obs, images, anns, geo_full, label: str = "each", mode: str = "full", size: int = 1024,
                 aug: AugConfig = NO_AUG, norm: tuple[float, float] = (1.0, 1.0), cache_dir=None,
                 limb_cut: float = 0.985, seed: int | None = None):
        """obs: observation ids. images / anns: from load_annotations (anns already cleaned).
        geo_full: DataFrame indexed by obs with full-resolution cx, cy, radius.
        norm: (centre, scale) applied to the input. With mode="raw" use geometric augmentation only:
        the photometric ones assume the quiet Sun ≈ 1 (limb / full)."""
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        self.obs = list(obs)
        sets = images[images.obs.isin(self.obs)].groupby("obs")
        self.paths = sets.path.first().to_dict()
        self.set_ids = sets.image_id.apply(list).to_dict()
        self.polygons = anns.groupby("image_id").polygon.apply(list).to_dict()
        self.geo = {o: tuple(geo_full.loc[o, ["cx", "cy", "radius"]]) for o in self.obs}
        self.label, self.mode, self.size, self.norm = label, mode, size, norm
        self.cache_dir, self.limb_cut = cache_dir, limb_cut
        self.k = size / FULL_RES
        self.fill = 0.0 if mode == "raw" else 1.0   # off-disk value, identical for train and validation
        self.aug = aug
        self.reseed(seed)

    def reseed(self, seed: int | None) -> None:
        """Fresh random streams (called per DataLoader worker, see src.train.seed_worker)."""
        self.rng = np.random.default_rng(seed)
        self.augmenter = Augmenter(self.aug, seed=None if seed is None else seed + 1)

    def __len__(self) -> int:
        return len(self.obs)

    def masks(self, o: str, size: int | None = None) -> list[np.ndarray]:
        """Instance map of every annotation set of `o` at `size` px (default: working resolution)."""
        size = size or self.size
        return [instance_mask(self.polygons.get(iid, []), (size, size), size / FULL_RES)
                for iid in self.set_ids[o]]

    def target(self, masks: list[np.ndarray]) -> np.ndarray:
        if self.label == "each":
            return (masks[self.rng.integers(len(masks))] > 0).astype(np.uint8)
        if self.label == "soft":
            return soft_label(masks)
        return (fuse_instance_masks(masks, self.label) > 0).astype(np.uint8)

    def __getitem__(self, i: int):
        o = self.obs[i]
        disk = tuple(v * self.k for v in self.geo[o])
        x = load_preprocessed(self.paths[o], self.geo[o], self.size, self.mode, self.cache_dir)
        y = self.target(self.masks(o))
        if self.aug != NO_AUG:
            x, (y,) = self.augmenter(x, [y], disk)
        x = model_input(x, disk, self.norm, self.fill, self.limb_cut)
        return torch.from_numpy(x[None]), torch.from_numpy(y[None].astype(np.float32)), i


def model_input(x: np.ndarray, disk: tuple[float, float, float], norm: tuple[float, float], fill: float,
                limb_cut: float = 0.985) -> np.ndarray:
    """Preprocessed image at working resolution -> network input: constant off-disk, then normalised.
    disk: geometry at working resolution. Shared by training and end-to-end inference."""
    x = x.astype(np.float32, copy=True)
    x[~disk_mask(x.shape, *disk, limb_cut)] = fill
    return (x - norm[0]) / norm[1]
