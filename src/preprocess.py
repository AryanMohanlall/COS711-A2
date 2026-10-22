"""Image preprocessing: solar disk detection, limb-darkening correction and background flattening.
GONG disks are already centred and rescaled (radius 906 ± 1.6 px, notebook 01)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .data import REPO_ROOT, load_image

CACHE_DIR = REPO_ROOT / "cache"


def find_disk(img: np.ndarray, size: int = 512, n_rays: int = 360) -> tuple[float, float, float]:
    """Locate the solar disk as (cx, cy, radius): fit a circle to the strongest intensity drop on rays
    from the centre. Robust to the bright off-limb halo that breaks Otsu thresholding."""
    scale = img.shape[0] / size
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), 1.0)

    c0 = size / 2
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    radii = np.arange(int(0.35 * size), int(0.5 * size))
    xs = (c0 + np.cos(angles)[:, None] * radii[None]).astype(np.float32)
    ys = (c0 + np.sin(angles)[:, None] * radii[None]).astype(np.float32)
    prof = cv2.remap(small, xs, ys, cv2.INTER_LINEAR)
    j = (prof[:, :-3] - prof[:, 3:]).argmax(axis=1) + 1
    px, py = xs[np.arange(n_rays), j].astype(np.float64), ys[np.arange(n_rays), j].astype(np.float64)

    for _ in range(2):
        A = np.c_[2 * px, 2 * py, np.ones_like(px)]
        cx, cy, c = np.linalg.lstsq(A, px ** 2 + py ** 2, rcond=None)[0]
        r = np.sqrt(c + cx ** 2 + cy ** 2)
        res = np.abs(np.hypot(px - cx, py - cy) - r)
        keep = res < max(2.0, 3 * np.median(res))
        px, py = px[keep], py[keep]
    return cx * scale, cy * scale, r * scale


def disk_mask(shape: tuple[int, int], cx: float, cy: float, radius: float, scale: float = 1.0) -> np.ndarray:
    """Boolean mask of pixels within `scale * radius` of the disk centre."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= (scale * radius) ** 2


def radial_profile(img: np.ndarray, cx: float, cy: float, radius: float,
                   n_bins: int = 100, work: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Median intensity vs r / R (limb-darkening curve) at low resolution.
    The median ignores small dark filaments and bright plage within each ring."""
    s = work / img.shape[0]
    small = cv2.resize(img, (work, work), interpolation=cv2.INTER_AREA).astype(np.float32)
    yy, xx = np.mgrid[:work, :work]
    rr = np.hypot(xx - cx * s, yy - cy * s) / (radius * s)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(rr, edges) - 1
    centres = 0.5 * (edges[1:] + edges[:-1])
    prof = np.array([np.median(small[idx == i]) if (idx == i).any() else np.nan for i in range(n_bins)])
    good = ~np.isnan(prof)
    return centres[good], prof[good]


def correct_limb_darkening(img: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    """Divide by the image's own radial median profile. Quiet Sun → ≈ 1 everywhere on the disk."""
    r_c, prof = radial_profile(img, cx, cy, radius)
    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    denom = np.interp(rr, r_c, prof).astype(np.float32)
    return img.astype(np.float32) / np.maximum(denom, 1.0)


def flatten_background(ratio: np.ndarray, cx: float, cy: float, radius: float,
                       work: int = 256, kernel_frac: float = 0.25) -> np.ndarray:
    """Divide out large-scale gradients / haze, estimated with a ~500 px median filter (filaments ~12 px
    wide are not absorbed). Off-disk pixels are set to 1 first so the limb does not bias it."""
    H = ratio.shape[0]
    s = work / H
    small = cv2.resize(ratio, (work, work), interpolation=cv2.INTER_AREA)
    inside = disk_mask(small.shape, cx * s, cy * s, radius * s, 0.98)
    small = np.where(inside, small, 1.0).astype(np.float32)
    # medianBlur needs uint8 for large kernels: encode ratio in [0, 2] as 0..255
    enc = np.clip(small * 127.5, 0, 255).astype(np.uint8)
    k = int(kernel_frac * work) | 1
    bg = cv2.medianBlur(enc, k).astype(np.float32) / 127.5
    bg = cv2.resize(bg, (H, H), interpolation=cv2.INTER_LINEAR)
    return ratio / np.maximum(bg, 0.2)


def preprocess(img: np.ndarray, mode: str = "full", disk: tuple[float, float, float] | None = None,
               limb_cut: float = 0.985) -> np.ndarray:
    """mode: "raw" (/255), "limb" (limb-darkening corrected, quiet Sun ≈ 1) or "full" (limb + flattening).
    limb/full set pixels beyond limb_cut·R to 1, removing the ring artefact from the limb rim."""
    if mode == "raw":
        return img.astype(np.float32) / 255.0
    cx, cy, r = disk if disk is not None else find_disk(img)
    out = correct_limb_darkening(img, cx, cy, r)
    if mode == "full":
        out = flatten_background(out, cx, cy, r)
    elif mode != "limb":
        raise ValueError(f"Unknown mode {mode!r}")
    out[~disk_mask(out.shape, cx, cy, r, limb_cut)] = 1.0
    return np.clip(out, 0, 2).astype(np.float32)


def load_preprocessed(path: str | Path, disk: tuple[float, float, float] | None = None, size: int = 1024,
                      mode: str = "full", cache_dir: Path | None = None) -> np.ndarray:
    """Preprocessed image resized to `size` px, cached as float16 .npy on first use.
    disk: full-resolution (cx, cy, radius), e.g. from artifacts/image_stats.csv; detected if omitted."""
    folder = Path(cache_dir or CACHE_DIR) / f"{mode}_{size}"
    f = folder / f"{Path(path).stem}.npy"
    if f.exists():
        return np.load(f).astype(np.float32)
    x = preprocess(load_image(path), mode, disk)
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA).astype(np.float16)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{np.random.randint(1 << 30)}.tmp")  # write then rename: safe with parallel loaders
    with open(tmp, "wb") as fh:
        np.save(fh, x)
    tmp.replace(f)
    return x.astype(np.float32)


def build_cache(paths, disks=None, workers: int = 8, **kwargs) -> None:
    """Fill the cache for every image in `paths` (disks: matching full-resolution geometries or None)."""
    disks = list(disks) if disks is not None else [None] * len(paths)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda a: load_preprocessed(*a, **kwargs), zip(paths, disks)))
