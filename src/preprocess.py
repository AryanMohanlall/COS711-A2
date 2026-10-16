"""Image preprocessing: solar disk geometry (and later normalisation / limb-darkening correction).

Findings from notebooks/01_data_exploration.ipynb: GONG images are already centred and
rescaled, disk radius 906 ± 1.6 px, centre within ~2 px of the image centre.
"""
import cv2
import numpy as np


def find_disk(img: np.ndarray, size: int = 512, n_rays: int = 360) -> tuple[float, float, float]:
    """Locate the solar disk as (cx, cy, radius) in full-resolution pixels.

    Casts rays from the image centre, takes the strongest outward intensity drop on each ray
    (the limb), and fits a circle to those points with one round of outlier rejection.
    Robust to the bright off-limb halo that breaks global thresholding (Otsu).
    """
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
