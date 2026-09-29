"""Turn a relative model output into metric heights using a coarse DEM.

Inputs are on the IMAGE grid (float32): R = height-like relative model output
(height_sign * prediction), D = DEM resampled onto the image grid (metres, its own datum).

Methods (Experiment 0):
  (a) dem_only          Z = D
  (b) robust_affine     Z = a*R + b, (a, b) fitted with a robust loss between lowpass(R) and D
                        on ANCHOR pixels only (fit happens at the DEM's scale, ~30 m)
  (c) dem_plus_residual Z = D + a*(R - mean_cell(R)): DEM plus the model's fine structure,
                        forced to zero mean inside every DEM cell, so the coarse heights stay the DEM's

Anchors come from a checkerboard of blocks; the complementary blocks are for testing only.
Output units = the DEM's (metres), vertical datum = the DEM's. Nothing here changes a grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from depthwizard.io import RasterMetadata


@dataclass
class Calibrated:
    z: np.ndarray
    method: str
    params: dict[str, Any] = field(default_factory=dict)


def checkerboard(height: int, width: int, block: int) -> np.ndarray:
    """True = anchor block, False = test block."""
    r = (np.arange(height) // block)[:, None]
    c = (np.arange(width) // block)[None, :]
    return ((r + c) % 2 == 0)


def lowpass(x: np.ndarray, size_px: int) -> np.ndarray:
    """NaN-aware box mean over size_px x size_px."""
    k = max(1, int(size_px)) | 1
    valid = np.isfinite(x).astype(np.float32)
    num = cv2.blur(np.nan_to_num(x, nan=0.0).astype(np.float32), (k, k), borderType=cv2.BORDER_REFLECT)
    den = cv2.blur(valid, (k, k), borderType=cv2.BORDER_REFLECT)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.5, num / den, np.nan).astype(np.float32)


def slope_deg(dem: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    gy, gx = np.gradient(dem.astype(np.float64), res_y, res_x)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def dem_cell_ids(dem_path, meta: RasterMetadata) -> np.ndarray:
    """For every image pixel, the index of the DEM cell it falls in (nearest; -1 outside)."""
    with rasterio.open(dem_path) as src:
        ids = np.arange(src.width * src.height, dtype=np.float64).reshape(src.height, src.width)
        out = np.full((meta.height, meta.width), -1.0)
        reproject(ids, out, src_transform=src.transform, src_crs=src.crs, src_nodata=None,
                  dst_transform=meta.affine, dst_crs=meta.crs, dst_nodata=-1.0, resampling=Resampling.nearest)
    return out.astype(np.int64)


def cell_mean(x: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Mean of x within each cell id, broadcast back to pixels (NaN-aware)."""
    ok = np.isfinite(x) & (cell >= 0)
    n = int(cell.max()) + 1
    s = np.bincount(cell[ok], weights=x[ok].astype(np.float64), minlength=n)
    c = np.bincount(cell[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
    out = np.full(x.shape, np.nan, dtype=np.float64)
    out[cell >= 0] = m[cell[cell >= 0]]
    return out


def anchor_pixels(dem: np.ndarray, r: np.ndarray, anchor_blocks: np.ndarray, res: tuple[float, float],
                  max_slope_deg: float) -> np.ndarray:
    """Anchor mask: anchor block, finite R and D, DEM slope <= max_slope_deg.
    Water is NOT excluded (no water mask available from RGB-only input) - recorded by callers."""
    return (anchor_blocks & np.isfinite(dem) & np.isfinite(r)
            & (slope_deg(dem, *res) <= max_slope_deg))


def dem_only(dem: np.ndarray) -> Calibrated:
    return Calibrated(dem.astype(np.float32), "a_dem_only", {})


def robust_affine(r: np.ndarray, dem: np.ndarray, anchors: np.ndarray, lowpass_px: int,
                  robust: str = "huber", max_px: int = 200_000, seed: int = 0) -> Calibrated:
    from sklearn.linear_model import HuberRegressor, RANSACRegressor
    r_low = lowpass(r, lowpass_px)
    m = anchors & np.isfinite(r_low)
    idx = np.flatnonzero(m)
    if idx.size < 100:
        raise ValueError(f"only {idx.size} anchor pixels; cannot fit")
    rng = np.random.default_rng(seed)
    if idx.size > max_px:
        idx = rng.choice(idx, max_px, replace=False)
    x = r_low.ravel()[idx].astype(np.float64)
    y = dem.ravel()[idx].astype(np.float64)
    # standardise x for a well-conditioned robust fit, then map back
    mu, sd = x.mean(), x.std()
    if sd == 0:
        raise ValueError("model output has no variation on the anchors")
    xs = ((x - mu) / sd)[:, None]
    if robust == "huber":
        est = HuberRegressor(epsilon=1.35, max_iter=1000).fit(xs, y)
        a_s, b_s, extra = float(est.coef_[0]), float(est.intercept_), {"huber_scale": float(est.scale_)}
    elif robust == "ransac":
        est = RANSACRegressor(random_state=seed).fit(xs, y)
        a_s, b_s = float(est.estimator_.coef_[0]), float(est.estimator_.intercept_)
        extra = {"inlier_fraction": float(est.inlier_mask_.mean())}
    else:
        raise ValueError("robust must be 'huber' or 'ransac'")
    a = a_s / sd
    b = b_s - a * mu
    z = (a * r.astype(np.float64) + b).astype(np.float32)
    return Calibrated(z, "b_robust_affine", {"a": a, "b": b, "robust": robust, "n_anchors_used": int(idx.size),
                                             "lowpass_px": lowpass_px, **extra})


def dem_plus_residual(r: np.ndarray, dem: np.ndarray, cell: np.ndarray, scale: float) -> Calibrated:
    resid = r.astype(np.float64) - cell_mean(r, cell)
    z = (dem.astype(np.float64) + scale * resid).astype(np.float32)
    return Calibrated(z, "c_dem_plus_residual", {"scale": float(scale),
                                                  "scale_source": "slope a of (b), fitted on anchors"})


def dem_plus_smooth_residual(r: np.ndarray, dem: np.ndarray, lowpass_px: int, scale: float) -> Calibrated:
    """Variant of (c): remove a CONTINUOUS ~DEM-cell low-pass of R instead of per-cell means.
    Keeps coarse heights ~= DEM without the 30 m block steps that per-cell means leave at cell edges."""
    resid = r.astype(np.float64) - lowpass(r, lowpass_px).astype(np.float64)
    z = (dem.astype(np.float64) + scale * resid).astype(np.float32)
    return Calibrated(z, "c2_dem_plus_smooth_residual", {"scale": float(scale), "lowpass_px": int(lowpass_px),
                                                          "scale_source": "slope a of (b), fitted on anchors"})
