"""Object heights: a DERIVED local ground surface, nDSM = DSM - ground, and per-building heights.

Ground: in each moving window, the robust low percentile of DSM heights over candidate ground pixels
(locally smooth, not water). Canopy and roofs are rough or raised, bare ground is smooth and low.
Where a window has too few candidate pixels (dense forest, dense urban with no visible ground) the
estimate is flagged LOW CONFIDENCE instead of being trusted.

Everything here is labelled "derived": none of it is a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class GroundResult:
    ground: np.ndarray             # metres, same datum as the DSM
    ndsm: np.ndarray               # metres above the derived ground, >= 0
    low_confidence: np.ndarray     # bool
    method: str
    notes: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def local_roughness(dsm: np.ndarray, res: tuple[float, float], window_m: float = 3.0) -> np.ndarray:
    """Std of the DSM in a small window (metres): roofs edges and canopy are rough, open ground is not."""
    k = max(3, int(round(window_m / res[0])) | 1)
    z = np.nan_to_num(dsm.astype(np.float64), nan=np.nanmean(dsm))
    m = cv2.blur(z, (k, k))
    m2 = cv2.blur(z * z, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0)).astype(np.float32)


def estimate_ground(dsm: np.ndarray, res: tuple[float, float], water: np.ndarray | None = None,
                    window_m: float = 60.0, percentile: float = 10.0, max_roughness_m: float = 0.5,
                    min_candidate_fraction: float = 0.05) -> GroundResult:
    h, w = dsm.shape
    rough = local_roughness(dsm, res)
    cand = np.isfinite(dsm) & (rough <= max_roughness_m)
    if water is not None:
        cand &= ~water
    win = max(8, int(round(window_m / res[0])))
    step = max(4, win // 2)
    rows = np.arange(0, h, step)
    cols = np.arange(0, w, step)
    coarse = np.full((len(rows), len(cols)), np.nan, np.float32)
    frac = np.zeros((len(rows), len(cols)), np.float32)
    for i, r in enumerate(rows):
        r0, r1 = max(0, r - win // 2), min(h, r + win // 2 + 1)
        for j, c in enumerate(cols):
            c0, c1 = max(0, c - win // 2), min(w, c + win // 2 + 1)
            m = cand[r0:r1, c0:c1]
            frac[i, j] = m.mean()
            if m.sum() >= 20:
                # detrend: plane through the candidate pixels, then the low percentile of the residuals
                # (a plain percentile on a slope would pick the downhill side)
                rr, cc = np.nonzero(m)
                zz = dsm[r0:r1, c0:c1][m].astype(np.float64)
                A = np.c_[rr - (r - r0), cc - (c - c0), np.ones(rr.size)]
                keep = np.ones(zz.size, bool)
                for _ in range(3):      # refit on the lower half: flat roofs are smooth too, drop them
                    coef, *_ = np.linalg.lstsq(A[keep], zz[keep], rcond=None)
                    res_ = zz - A @ coef
                    keep = res_ <= np.median(res_[keep])
                    if keep.sum() < 10:
                        break
                coarse[i, j] = coef[2] + np.percentile(zz - A @ coef, percentile)
    # fill windows without candidates from their neighbours (flagged below), then interpolate to full res
    filled = coarse.copy()
    if np.isnan(filled).all():
        filled[:] = np.nanpercentile(dsm, percentile)
    else:
        idx = np.argwhere(np.isfinite(filled))
        for i, j in np.argwhere(np.isnan(filled)):
            d = np.abs(idx - (i, j)).sum(1)
            filled[i, j] = filled[tuple(idx[d.argmin()])]
    # interpolate at the TRUE window-centre coordinates (cv2.resize would assume evenly spread samples and
    # shift them by up to half a window, which becomes a height error on slopes)
    from scipy.interpolate import RegularGridInterpolator
    if len(rows) > 1 and len(cols) > 1:
        f = RegularGridInterpolator((rows, cols), filled, method="linear", bounds_error=False, fill_value=None)
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        ground = f(np.stack([rr.ravel(), cc.ravel()], -1)).reshape(h, w).astype(np.float32)
    else:
        ground = np.full((h, w), float(filled.ravel()[0]), np.float32)
    ground = np.minimum(ground, np.where(np.isfinite(dsm), dsm, ground)).astype(np.float32)   # never above the surface
    lowc_coarse = (frac < min_candidate_fraction).astype(np.float32)
    low_conf = cv2.resize(lowc_coarse, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
    if water is not None:
        low_conf |= water
    low_conf |= ~np.isfinite(dsm)
    ndsm = np.where(np.isfinite(dsm), np.maximum(dsm - ground, 0), np.nan).astype(np.float32)
    return GroundResult(
        ground=np.where(np.isfinite(dsm), ground, np.nan).astype(np.float32), ndsm=ndsm, low_confidence=low_conf,
        method=f"p{percentile:g} of smooth (roughness <= {max_roughness_m} m), non-water DSM pixels in "
               f"{window_m:g} m windows",
        notes=["derived, not measured", f"low confidence where < {min_candidate_fraction:.0%} of a window is "
               "candidate ground (e.g. continuous canopy, dense roofs) or water"],
        summary={"candidate_ground_fraction": float(cand.mean()), "low_confidence_fraction": float(low_conf.mean()),
                 "ndsm_p95_m": float(np.nanpercentile(ndsm, 95))})


def building_heights(dsm: np.ndarray, ground: np.ndarray, footprints: np.ndarray, ring_px: int = 4,
                     erode_px: int = 2) -> dict[int, dict]:
    """Per-building DERIVED height: median roof interior minus median DSM in a ground ring around it.

    footprints: int raster, 0 = background, k > 0 = building id (rasterised on the DSM grid).
    """
    out = {}
    kern_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    kern_r = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    any_bldg = footprints > 0
    for k in np.unique(footprints[footprints > 0]):
        m = (footprints == k).astype(np.uint8)
        interior = cv2.erode(m, kern_e) > 0
        ring = (cv2.dilate(m, kern_r) > 0) & ~any_bldg
        roof = dsm[interior & np.isfinite(dsm)]
        grd = dsm[ring & np.isfinite(dsm)]
        if roof.size < 5 or grd.size < 5:
            continue
        out[int(k)] = {"height_m": float(np.median(roof) - np.median(grd)), "roof_px": int(roof.size),
                       "ring_px": int(grd.size), "label": "derived"}
    return out


# ------------------------------------------------------------------ confidence of derived heights (Phase 11 T5)
LOW, MEDIUM = 1, 2                  # 0 = no data. There is deliberately no HIGH level: derived heights have not been
LEVEL_NAMES = {0: "none", LOW: "low", MEDIUM: "medium"}     # validated to that standard (r <= 0.49 vs LiDAR, Phase 8)
REASONS = {1: "no visible ground nearby (ground estimate unreliable)",
           2: "tree cover (canopy hides the ground; Phase 8: ground +5 m, heights -4 m)",
           4: "water",
           8: "height below 2x the DSM uncertainty (not distinguishable from noise)"}
TREE_CODES = (10, 95)               # ESA WorldCover: tree cover, mangroves


def height_confidence(ground: GroundResult, sigma: np.ndarray | None = None, landcover: np.ndarray | None = None,
                      water: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel confidence of the DERIVED height above ground: (level uint8, reason bitmask uint8).

    LOW if any reason applies, else MEDIUM. Reasons (bits, see REASONS): ground low-confidence window, tree cover
    (ESA WorldCover codes 10/95), water, nDSM < 2 sigma. landcover=None -> tree reason cannot be checked (the caller
    must say so). THIS SHOULD BE TESTED per release: experiments/11_confidence checks error per level vs LiDAR.
    """
    nd = ground.ndsm
    reasons = np.zeros(nd.shape, np.uint8)
    reasons[ground.low_confidence] |= 1
    if landcover is not None:
        reasons[np.isin(landcover, TREE_CODES)] |= 2
    if water is not None:
        reasons[water] |= 4
    if sigma is not None:
        with np.errstate(invalid="ignore"):
            reasons[np.isfinite(sigma) & (nd < 2 * sigma)] |= 8
    level = np.where(reasons > 0, LOW, MEDIUM).astype(np.uint8)
    level[~np.isfinite(nd)] = 0
    return level, reasons


def describe_reasons(bits: int) -> list[str]:
    return [txt for b, txt in REASONS.items() if bits & b]
