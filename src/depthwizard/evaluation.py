"""DSM evaluation: MAE / RMSE / r / bias (+ NMAD, p95), nodata-safe, native or 30 m aggregated, stratified.

Rules this module enforces:
- Only pixels finite in BOTH rasters count; how many were excluded is always reported.
- "native" mode never resamples: the grids must already be identical.
- "aggregate" mode AREA-AVERAGES the fine prediction onto the coarse reference grid
  (e.g. 0.6 m -> Copernicus 30 m). Cells with too little valid fine coverage are dropped.
- A 30 m DEM is a coarse-scale reference only. It is never building-height truth, so
  aggregate-mode scores say how well coarse terrain is matched, not buildings.
- Vertical datums are compared from file tags; a mismatch or missing tag is flagged, not ignored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform as warp_transform

SLOPE_CLASSES = {0: "flat (<5 deg)", 1: "moderate (5-15 deg)", 2: "steep (>=15 deg)"}


def metrics(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Error statistics of pred - ref over pixels finite in both (and inside `mask`)."""
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {ref.shape}")
    valid = np.isfinite(pred) & np.isfinite(ref)
    considered = valid.size if mask is None else int(mask.sum())
    if mask is not None:
        valid &= mask.astype(bool)
    n = int(valid.sum())
    out: dict[str, Any] = {"n": n, "excluded_nodata": int(considered - n)}
    if n == 0:
        return out
    p, r = pred[valid], ref[valid]
    d = p - r
    out.update(
        bias=float(d.mean()),
        mae=float(np.abs(d).mean()),
        rmse=float(np.sqrt((d ** 2).mean())),
        median_error=float(np.median(d)),
        nmad=float(1.4826 * np.median(np.abs(d - np.median(d)))),
        p95_abs_error=float(np.percentile(np.abs(d), 95)),
        le90=float(np.percentile(np.abs(d), 90)),        # linear error at 90 % confidence
        pearson_r=(float(np.corrcoef(p, r)[0, 1]) if n > 1 and p.std() > 0 and r.std() > 0 else None),
        # coefficient of determination of pred as a predictor of ref (can be < 0; not r^2)
        r2=(float(1.0 - (d ** 2).sum() / ((r - r.mean()) ** 2).sum()) if n > 1 and r.std() > 0 else None),
    )
    return out


def affine_fitted_metrics(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """For relative DSMs: fit ref ~ a*pred + b on the SAME pixels, then score.

    OPTIMISTIC by construction (the fit sees the answers). Useful to compare relative
    models' shape quality; never a deployable metric-accuracy number.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(ref)
    if mask is not None:
        valid &= mask.astype(bool)
    if valid.sum() < 2 or pred[valid].std() == 0:
        return {"n": int(valid.sum()), "note": "not enough valid variation to fit"}
    A = np.stack([pred[valid], np.ones(valid.sum())], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, ref[valid], rcond=None)
    out = metrics(a * pred + b, ref, valid)
    out.update(fit_a=float(a), fit_b=float(b),
               warning="affine fitted on the evaluation data itself - OPTIMISTIC, not deployable accuracy")
    return out


def slope_degrees(dem: np.ndarray, res_x_m: float, res_y_m: float) -> np.ndarray:
    """Slope in degrees from a metric-grid DEM (NaN-propagating central differences)."""
    gy, gx = np.gradient(dem.astype(np.float64), res_y_m, res_x_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def slope_classes(dem: np.ndarray, res_x_m: float, res_y_m: float) -> np.ndarray:
    """0 flat, 1 moderate, 2 steep; -1 where slope is undefined."""
    s = slope_degrees(dem, res_x_m, res_y_m)
    cls = np.full(s.shape, -1, dtype=np.int8)
    cls[s < 5] = 0
    cls[(s >= 5) & (s < 15)] = 1
    cls[s >= 15] = 2
    return cls


def stratified_metrics(pred: np.ndarray, ref: np.ndarray, labels: np.ndarray,
                       names: dict[int, str]) -> dict[str, Any]:
    """metrics() per label value; labels not in `names` (e.g. -1 = unknown) are skipped."""
    if labels.shape != pred.shape:
        raise ValueError("label raster must be on the evaluation grid")
    return {names[k]: metrics(pred, ref, labels == k) for k in sorted(names)}


# ------------------------------------------------------------------ raster level
def _grid(src: rasterio.io.DatasetReader) -> tuple:
    return (src.crs, src.transform, src.width, src.height)


def _read_nan(src: rasterio.io.DatasetReader) -> np.ndarray:
    a = src.read(1).astype(np.float32)
    if src.nodata is not None and not np.isnan(src.nodata):
        a[a == src.nodata] = np.nan
    return a


def metric_res(transform: rasterio.Affine, crs: rasterio.crs.CRS, height: int) -> tuple[float, float]:
    """Pixel size in metres. Geographic grids use the scene's centre latitude
    (fine for stratifying one scene; not a geodesic computation)."""
    if crs.is_projected:
        return abs(transform.a), abs(transform.e)
    lat = transform.f + transform.e * height / 2
    m_per_deg_lat = 111_132.954 - 559.822 * np.cos(np.radians(2 * lat))
    m_per_deg_lon = 111_412.84 * np.cos(np.radians(lat))
    return abs(transform.a) * m_per_deg_lon, abs(transform.e) * m_per_deg_lat


def _cells_inside(pred_src: rasterio.io.DatasetReader, ref_src: rasterio.io.DatasetReader) -> np.ndarray:
    """True for reference cells whose four corners all lie inside the prediction's footprint.
    GDAL 'average' only averages the fine pixels that exist, so a half-covered edge cell
    would otherwise look fully covered while the reference describes the whole cell."""
    h, w = ref_src.height, ref_src.width
    rows, cols = np.mgrid[0:h + 1, 0:w + 1]
    xs, ys = ref_src.transform * (cols.ravel(), rows.ravel())
    px, py = warp_transform(ref_src.crs, pred_src.crs, xs, ys)
    l, b, r, t = pred_src.bounds
    ok = ((np.asarray(px) >= l) & (np.asarray(px) <= r) & (np.asarray(py) >= b) & (np.asarray(py) <= t))
    ok = ok.reshape(h + 1, w + 1)
    return ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]


def aggregate_to_grid(pred_src: rasterio.io.DatasetReader, ref_src: rasterio.io.DatasetReader,
                      min_coverage: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Area-average the fine prediction onto the reference grid.

    Returns (aggregated, coverage) where coverage is the valid fraction of fine pixels
    per coarse cell (0 for cells not fully inside the prediction footprint); cells below
    `min_coverage` are set to NaN.
    """
    pred = _read_nan(pred_src)
    valid = np.isfinite(pred).astype(np.float32)
    shape = (ref_src.height, ref_src.width)
    common = dict(src_transform=pred_src.transform, src_crs=pred_src.crs,
                  dst_transform=ref_src.transform, dst_crs=ref_src.crs, resampling=Resampling.average)
    agg = np.full(shape, np.nan, dtype=np.float32)
    reproject(np.nan_to_num(pred, nan=0.0) * valid, agg, src_nodata=None, dst_nodata=np.nan, **common)
    cov = np.full(shape, np.nan, dtype=np.float32)
    reproject(valid, cov, src_nodata=None, dst_nodata=np.nan, **common)
    cov = np.where(_cells_inside(pred_src, ref_src), cov, 0.0).astype(np.float32)
    # agg is mean(pred*valid); divide by the valid fraction to get the mean over valid pixels
    with np.errstate(invalid="ignore", divide="ignore"):
        agg = np.where(cov >= min_coverage, agg / cov, np.nan).astype(np.float32)
    return agg, cov


def datum_check(pred_tag: str | None, ref_tag: str | None) -> str:
    """MATCH / MISMATCH / UNKNOWN from two VERTICAL_DATUM tags, compared as datums, not strings
    ("EGM2008 (converted from NAVD88)" == "EGM2008"). Unrecognised text is UNKNOWN, never guessed."""
    if not (pred_tag and ref_tag):
        return "UNKNOWN - NEEDS VERIFICATION (tag missing)"
    from depthwizard.calibration.datum import parse_datum
    try:
        a, b = parse_datum(pred_tag), parse_datum(ref_tag)
    except ValueError:
        return "MATCH" if pred_tag == ref_tag else "UNKNOWN - NEEDS VERIFICATION (unrecognised datum tag)"
    return "MATCH" if a == b else f"MISMATCH - scores include a datum offset ({a} vs {b})"


def evaluate_rasters(pred_path: str | Path, ref_path: str | Path, mode: str = "native",
                     relative: bool = False, min_coverage: float = 0.9,
                     labels: np.ndarray | None = None, label_names: dict[int, str] | None = None
                     ) -> dict[str, Any]:
    """Score a DSM GeoTIFF against a reference GeoTIFF.

    mode="native": grids must match exactly (raises otherwise; nothing is resampled).
    mode="aggregate": prediction area-averaged onto the reference grid (for 30 m DEMs).
    relative=True: also report affine-fitted (optimistic) metrics for an rDSM.
    Slope strata are always reported (computed from the reference on the evaluation grid);
    `labels` adds e.g. land-cover strata and must be on the evaluation grid.
    """
    with rasterio.open(pred_path) as ps, rasterio.open(ref_path) as rs:
        report: dict[str, Any] = {"pred": str(pred_path), "ref": str(ref_path), "mode": mode}
        pt, rt = ps.tags(), rs.tags()
        pdatum, rdatum = pt.get("VERTICAL_DATUM"), rt.get("VERTICAL_DATUM")
        report["vertical_datum"] = {"pred": pdatum, "ref": rdatum, "check": datum_check(pdatum, rdatum)}
        report["pred_kind"] = pt.get("DSM_KIND")
        if ps.crs is None or rs.crs is None:
            raise ValueError("both rasters need a CRS; refusing to compare un-georeferenced grids")
        if mode == "native":
            if _grid(ps) != _grid(rs):
                raise ValueError("native mode needs identical grids (CRS, transform, size); "
                                 "use mode='aggregate' or regrid explicitly first")
            pred, ref = _read_nan(ps), _read_nan(rs)
        elif mode == "aggregate":
            if abs(ps.res[0]) >= abs(rs.res[0]) and ps.crs == rs.crs:
                raise ValueError("aggregate mode expects a finer prediction than the reference")
            pred, cov = aggregate_to_grid(ps, rs, min_coverage)
            if not np.any(cov > 0):
                raise ValueError("no reference cell lies inside the prediction footprint "
                                 "(disjoint extents, wrong CRS, or prediction smaller than one cell)")
            ref = _read_nan(rs)
            report["aggregation"] = {"method": "area average", "min_coverage": min_coverage,
                                     "cells_dropped_low_coverage": int(np.sum(np.nan_to_num(cov) < min_coverage))}
        else:
            raise ValueError("mode must be 'native' or 'aggregate'")
        transform, crs = rs.transform, rs.crs

    report["grid"] = {"crs": crs.to_string(), "shape": list(ref.shape), "res": [abs(transform.a), abs(transform.e)]}
    report["overall"] = metrics(pred, ref)
    if relative:
        report["affine_fitted"] = affine_fitted_metrics(pred, ref)

    rx, ry = metric_res(transform, crs, ref.shape[0])
    report["by_slope"] = stratified_metrics(pred, ref, slope_classes(ref, rx, ry), SLOPE_CLASSES)
    report["slope_source"] = f"reference DSM, pixel {rx:.2f} x {ry:.2f} m"
    if labels is not None:
        report["by_label"] = stratified_metrics(pred, ref, labels, label_names or {})
    return report
