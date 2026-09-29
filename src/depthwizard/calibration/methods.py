"""Calibration methods M0-M2 behind one interface, each producing a calibration report.

  M0 dem_only       Z = DEM (EGM2008) on the image grid
  M1 global_affine  Z = a*R + b; (a, b) robustly fitted (Huber, cross-checked with RANSAC) between the
                    low-pass of R and the DEM on anchor pixels (no water, nodata, steep slope, edges)
  M2 dem_residual   Z = DEM + s*HP(R); HP zero-mean in each DEM cell ("cell") or a continuous 30 m
                    high-pass ("smooth", measured best in Phase 8); s = a from the M1 fit

Anchors are split spatially (fit_mask vs eval_mask) and never overlap. Every method runs a quality gate;
on failure the result is NOT metric (is_metric=False, z=None) and the report says why. Heights are
EGM2008 metres: the DEM must already be EGM2008 (see datum.to_egm2008).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from depthwizard.calibration.core import cell_mean, slope_deg
from depthwizard.calibration.signal import Signal


@dataclass
class GateConfig:
    min_dem_coverage: float = 0.95
    min_anchors: int = 1000
    max_heldout_nmad_m: float = 3.0      # M1: held-out residual vs DEM at DEM scale
    max_scale_rel_std: float = 0.5       # M2: bootstrap std(s) / s
    bootstrap: int = 30


@dataclass
class CalibrationInputs:
    signal: Signal
    dem: np.ndarray                       # EGM2008 metres on the image grid (NaN = no data)
    res: tuple[float, float]              # pixel size in metres
    fit_mask: np.ndarray                  # pixels that may be used to FIT (spatial split)
    eval_mask: np.ndarray                 # held-out pixels (never used to fit)
    cell_ids: np.ndarray | None = None    # DEM cell index per pixel (required for M2 "cell")
    water: np.ndarray | None = None       # True = water (Copernicus WBM > 0)
    dem_error: np.ndarray | None = None   # per-pixel DEM height error (Copernicus HEM), metres
    max_slope_deg: float = 15.0
    edge_px: int = 32
    seed: int = 0
    dem_datum: str = "EGM2008"


@dataclass
class CalibrationResult:
    method: str
    is_metric: bool
    z: np.ndarray | None                  # EGM2008 metres, or None when the gate failed
    uncertainty: np.ndarray | None        # 1-sigma metres (estimate), NaN = low confidence / unknown
    low_confidence: np.ndarray | None     # True where the estimate should not be trusted (e.g. water)
    report: dict[str, Any] = field(default_factory=dict)


def _nmad(e: np.ndarray) -> float:
    e = e[np.isfinite(e)]
    return float(1.4826 * np.median(np.abs(e - np.median(e)))) if e.size else float("nan")


def _stats(e: np.ndarray) -> dict[str, float]:
    e = e[np.isfinite(e)]
    if not e.size:
        return {"n": 0}
    return {"n": int(e.size), "bias": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())), "nmad": _nmad(e),
            "le90": float(np.percentile(np.abs(e), 90))}


def anchor_masks(inp: CalibrationInputs) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Fit and held-out anchor pixels: finite R/low/DEM, not water, slope <= max, away from edges."""
    h, w = inp.dem.shape
    ok = np.isfinite(inp.dem) & np.isfinite(inp.signal.low) & np.isfinite(inp.signal.r)
    excluded = {"nodata": int((~ok).sum())}
    if inp.water is not None:
        excluded["water"] = int((ok & inp.water).sum())
        ok &= ~inp.water
    steep = slope_deg(inp.dem, *inp.res) > inp.max_slope_deg
    excluded["steep"] = int((ok & steep).sum())
    ok &= ~steep
    edge = np.zeros((h, w), bool)
    e = inp.edge_px
    edge[:e], edge[-e:], edge[:, :e], edge[:, -e:] = True, True, True, True
    excluded["edge"] = int((ok & edge).sum())
    ok &= ~edge
    fit, held = ok & inp.fit_mask, ok & inp.eval_mask
    if np.any(fit & held):
        raise ValueError("fit and held-out anchors overlap")
    return fit, held, excluded


def _dem_error(inp: CalibrationInputs) -> np.ndarray:
    return inp.dem_error.astype(np.float32) if inp.dem_error is not None else \
        np.full(inp.dem.shape, np.nan, np.float32)


def _low_conf(inp: CalibrationInputs) -> np.ndarray:
    lc = ~np.isfinite(inp.dem)
    if inp.water is not None:
        lc |= inp.water
    return lc


def _coverage_gate(inp: CalibrationInputs, gate: GateConfig) -> list[str]:
    cov = float(np.isfinite(inp.dem).mean())
    return [] if cov >= gate.min_dem_coverage else [f"DEM covers {cov:.1%} of the image (< {gate.min_dem_coverage:.0%})"]


def _base_report(name: str, inp: CalibrationInputs, gate: GateConfig) -> dict[str, Any]:
    return {"method": name, "output_datum": "EGM2008", "dem_datum": inp.dem_datum, "units": "metres",
            "signal": inp.signal.info, "gate_config": asdict(gate),
            "dem_coverage": float(np.isfinite(inp.dem).mean()),
            "water_fraction": float(inp.water.mean()) if inp.water is not None else None}


def _finish(name: str, z, unc, inp, report, failures) -> CalibrationResult:
    report["gate"] = {"passed": not failures, "reasons": failures}
    lc = _low_conf(inp)
    if failures:
        report["fallback"] = "relative DSM only (metric output refused by the quality gate)"
        return CalibrationResult(name, False, None, None, lc, report)
    unc = np.where(lc, np.nan, unc).astype(np.float32)
    report["uncertainty_m"] = {"median": float(np.nanmedian(unc)) if np.isfinite(unc).any() else None,
                               "p90": float(np.nanpercentile(unc, 90)) if np.isfinite(unc).any() else None,
                               "low_confidence_fraction": float(lc.mean()),
                               "model": report.get("uncertainty_model")}
    return CalibrationResult(name, True, z.astype(np.float32), unc, lc, report)


class CalibrationMethod:
    name = "base"

    def calibrate(self, inp: CalibrationInputs, gate: GateConfig | None = None) -> CalibrationResult:
        raise NotImplementedError


class DemOnly(CalibrationMethod):
    name = "M0_dem_only"

    def calibrate(self, inp, gate=None):
        gate = gate or GateConfig()
        rep = _base_report(self.name, inp, gate)
        rep["uncertainty_model"] = "Copernicus HEM (per-pixel DEM error)"
        return _finish(self.name, inp.dem, _dem_error(inp), inp, rep, _coverage_gate(inp, gate))


def _fit_affine(x: np.ndarray, y: np.ndarray, how: str, seed: int) -> tuple[float, float, dict]:
    from sklearn.linear_model import HuberRegressor, RANSACRegressor
    mu, sd = float(x.mean()), float(x.std())
    if sd == 0:
        raise ValueError("no variation in the relative signal on the anchors")
    xs = ((x - mu) / sd)[:, None]
    if how == "huber":
        est = HuberRegressor(epsilon=1.35, max_iter=1000).fit(xs, y)
        a_s, b_s, extra = float(est.coef_[0]), float(est.intercept_), {"huber_scale": float(est.scale_)}
    else:
        est = RANSACRegressor(random_state=seed).fit(xs, y)
        a_s, b_s = float(est.estimator_.coef_[0]), float(est.estimator_.intercept_)
        extra = {"inlier_fraction": float(est.inlier_mask_.mean())}
    a = a_s / sd
    return a, b_s - a * mu, extra


class GlobalAffine(CalibrationMethod):
    name = "M1_global_affine"

    def __init__(self, max_fit_px: int = 200_000):
        self.max_fit_px = max_fit_px

    def fit(self, inp: CalibrationInputs, gate: GateConfig) -> dict[str, Any]:
        fit, held, excluded = anchor_masks(inp)
        rng = np.random.default_rng(inp.seed)
        idx = np.flatnonzero(fit)
        out: dict[str, Any] = {"anchors": {"source": "DEM pixels (low-pass of R vs DEM)", "fit": int(idx.size),
                                           "held_out": int(held.sum()), "excluded": excluded,
                                           "split": "spatial (fit_mask vs eval_mask)"}}
        if idx.size < max(gate.min_anchors, 100):
            out["error"] = f"only {idx.size} fit anchors"
            return out
        use = rng.choice(idx, min(idx.size, self.max_fit_px), replace=False)
        x, y = inp.signal.low.ravel()[use].astype(np.float64), inp.dem.ravel()[use].astype(np.float64)
        a, b, hx = _fit_affine(x, y, "huber", inp.seed)
        ar, br, rx = _fit_affine(x, y, "ransac", inp.seed)
        # m-out-of-n bootstrap (nb points), std rescaled by sqrt(nb/n). Pixels are spatially correlated, so
        # this still UNDER-states the scale uncertainty; sigma coverage is checked against LiDAR in Exp 0.
        boot, nb = [], min(use.size, 5_000)
        for _ in range(gate.bootstrap):
            j = rng.choice(use.size, nb)
            boot.append(_fit_affine(x[j], y[j], "huber", inp.seed)[0])
        boot_scale = float(np.sqrt(nb / use.size))
        pred_fit = a * inp.signal.low + b
        out.update(a=a, b=b, huber=hx, ransac={"a": ar, "b": br, **rx},
                   huber_vs_ransac_rel_diff_a=float(abs(ar - a) / max(abs(a), 1e-12)),
                   a_bootstrap_std=float(np.std(boot)) * boot_scale,
                   bootstrap={"resamples": gate.bootstrap, "m": nb, "n": int(use.size), "rescale": boot_scale},
                   residual_fit_m=_stats((pred_fit - inp.dem)[fit]),
                   residual_heldout_m=_stats((pred_fit - inp.dem)[held]))
        return out

    def calibrate(self, inp, gate=None):
        gate = gate or GateConfig()
        rep = _base_report(self.name, inp, gate)
        f = self.fit(inp, gate)
        rep.update(f)
        fails = _coverage_gate(inp, gate)
        if "error" in f:
            return _finish(self.name, None, None, inp, rep, fails + [f["error"]])
        if f["a"] <= 0:
            fails.append(f"fitted scale a = {f['a']:.4g} <= 0 (polarity or model failure)")
        nm = f["residual_heldout_m"].get("nmad", np.inf)
        if not nm <= gate.max_heldout_nmad_m:
            fails.append(f"held-out residual vs DEM NMAD {nm:.2f} m > {gate.max_heldout_nmad_m} m: the model's "
                         "low-frequency signal does not follow the terrain")
        z = f["a"] * inp.signal.r + f["b"]
        rep["uncertainty_model"] = "sqrt(heldout_nmad^2 + (a_std*|R-mean(R)|)^2)"
        unc = np.sqrt(nm ** 2 + (f["a_bootstrap_std"] * np.abs(inp.signal.r - np.nanmean(inp.signal.r))) ** 2)
        return _finish(self.name, z, unc, inp, rep, fails)


class DemResidual(CalibrationMethod):
    def __init__(self, mode: str = "smooth"):
        if mode not in ("cell", "smooth"):
            raise ValueError("mode must be 'cell' or 'smooth'")
        self.mode = mode
        self.name = f"M2_dem_residual_{mode}"

    def high_pass(self, inp: CalibrationInputs) -> np.ndarray:
        if self.mode == "smooth":
            return inp.signal.high
        if inp.cell_ids is None:
            raise ValueError("M2 'cell' needs cell_ids (DEM cell per pixel)")
        return (inp.signal.r - cell_mean(inp.signal.r, inp.cell_ids)).astype(np.float32)

    def calibrate(self, inp, gate=None):
        gate = gate or GateConfig()
        rep = _base_report(self.name, inp, gate)
        f = GlobalAffine().fit(inp, gate)
        rep["scale_fit"] = f
        fails = _coverage_gate(inp, gate)
        if "error" in f:
            return _finish(self.name, None, None, inp, rep, fails + [f["error"]])
        s, s_std = f["a"], f["a_bootstrap_std"]
        rep.update(s=s, s_bootstrap_std=s_std, s_source="slope a of the robust (Huber) fit, DEM anchors")
        if s <= 0:
            fails.append(f"scale s = {s:.4g} <= 0 (polarity or model failure)")
        elif s_std / s > gate.max_scale_rel_std:
            fails.append(f"scale unstable: bootstrap std/s = {s_std / s:.2f} > {gate.max_scale_rel_std}")
        hp = self.high_pass(inp)
        z = inp.dem + s * hp
        dem_err = _dem_error(inp)
        rep["uncertainty_model"] = "sqrt(HEM^2 + (s_std*|HP|)^2); HEM missing -> detail term only"
        unc = np.sqrt(np.nan_to_num(dem_err, nan=0.0) ** 2 + (s_std * np.abs(hp)) ** 2)
        unc = np.where(np.isfinite(dem_err) | (inp.dem_error is None), unc, np.nan)
        rep["added_detail_m"] = _stats((s * hp)[np.isfinite(hp)])
        return _finish(self.name, z, unc, inp, rep, fails)


METHODS = {"M0": DemOnly, "M1": GlobalAffine, "M2_cell": lambda: DemResidual("cell"),
           "M2_smooth": lambda: DemResidual("smooth")}
