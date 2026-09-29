#!/usr/bin/env python3
"""Phase 8: stratified validation + GSD sweep against independent LiDAR (configs/exp8.yaml).

Per site x GSD x {DEM-only, (b) affine, (c) per-cell residual, (c2) smooth residual} x model:
  - main tables (GSD 0.6 m): vs LiDAR DSM at its native 2 m, stratified by land cover (ESA WorldCover)
    and terrain slope (LiDAR DTM smoothed to ~10 m); also vs Copernicus 30 m (consistency only)
  - GSD sweep: every GSD scored on a common 6 m grid (LiDAR DSM block-averaged 3x3)
  - ground-estimate check: crude 30 m opening vs LiDAR DTM; derived height vs LiDAR nDSM
  - failure cases: worst 128 m windows (c2, DA3) rendered RGB | prediction | LiDAR | error
Only TEST blocks are scored (checkerboard; anchors used for fitting are excluded).
LiDAR DSM/DTM are converted NAVD88 -> EGM2008 (GEOID18 + EGM2008 grids via PROJ, datum.raster_file_to_egm2008)
before any comparison, so bias vs LiDAR no longer contains a datum offset (Phase 11 T1; the pre-T1 run
20260928T202021Z did not convert). std_error (= RMSE with bias removed) is reported alongside.

  PYTHONPATH=src python experiments/08_stratified/run_stratified.py
"""
from __future__ import annotations

import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.warp import reproject  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

from depthwizard.calibration import (anchor_pixels, checkerboard, dem_cell_ids, dem_only, dem_plus_residual,  # noqa: E402
                                     dem_plus_smooth_residual, lowpass, robust_affine, slope_deg)
from depthwizard.calibration.datum import raster_file_to_egm2008  # noqa: E402
from depthwizard.evaluation import aggregate_to_grid, metrics  # noqa: E402
from depthwizard.export import export_dsm  # noqa: E402
from depthwizard.geo import COPERNICUS_GLO30, dem_to_image_grid  # noqa: E402
from depthwizard.io import downsample_geotiff, read_raster  # noqa: E402
from depthwizard.pipeline import get_predictor  # noqa: E402
from depthwizard.reconstruction import ground_estimate  # noqa: E402
from depthwizard.tiling import run_tiled  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KEEP = ("n", "bias", "mae", "rmse", "pearson_r", "r2", "nmad")


def slim(m: dict) -> dict:
    out = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k in KEEP}
    if m.get("n") and "rmse" in m:
        out["std_error"] = round(math.sqrt(max(m["rmse"] ** 2 - m["bias"] ** 2, 0.0)), 3)
    return out


def read1(path) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float32)
        if s.nodata is not None and not np.isnan(s.nodata):
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs


def onto(src_path, like_path, resampling) -> np.ndarray:
    with rasterio.open(src_path) as s, rasterio.open(like_path) as d:
        out = np.full((d.height, d.width), np.nan, np.float32)
        reproject(rasterio.band(s, 1), out, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan,
                  dst_transform=d.transform, dst_crs=d.crs, dst_nodata=np.nan, resampling=resampling)
    return out


def strata_masks(ref_path, dtm_path, wc_path, cfg) -> dict[str, np.ndarray]:
    wc = onto(wc_path, ref_path, Resampling.nearest)
    dtm, tr, _ = read1(dtm_path)
    res = abs(tr.a)
    slope = slope_deg(lowpass(dtm, max(1, round(10 / res))), res, res)
    masks = {}
    for name, codes in cfg["strata"]["land_cover"].items():
        masks[f"land:{name}"] = np.isin(wc, codes)
    for name, (lo, hi) in cfg["strata"]["terrain_slope_deg"].items():
        masks[f"terrain:{name}"] = (slope >= lo) & (slope < hi)
    return masks


def score(pred_path, ref_path, masks=None) -> dict:
    with rasterio.open(pred_path) as ps, rasterio.open(ref_path) as rs:
        agg, _ = aggregate_to_grid(ps, rs, 0.9)
    ref, _, _ = read1(ref_path)
    out = {"overall": slim(metrics(agg, ref))}
    if masks:
        out["strata"] = {k: slim(metrics(agg, ref, m)) for k, m in masks.items() if m.any()}
    return out, agg


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "exp8.yaml").read_text())
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = ROOT / "runs" / "exp8" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "configs" / "exp8.yaml", out / "config.yaml")
    cc = cfg["calibration"]
    results, ground_checks, failures, timings = [], [], [], []
    datum_records = {}

    for site_name in cfg["sites"]:
        site = json.loads((ROOT / "data" / "exp8" / site_name / "site.json").read_text())
        sdir = out / site_name
        sdir.mkdir(exist_ok=True)
        lid2, dl = raster_file_to_egm2008(site["lidar_dsm"]["path"], sdir / "lidar_dsm_2m_egm2008.tif")
        dtm2, dt = raster_file_to_egm2008(site["lidar_dtm"]["path"], sdir / "lidar_dtm_2m_egm2008.tif")
        datum_records[site_name] = {"lidar_dsm": dl, "lidar_dtm": dt}
        lid6 = downsample_geotiff(lid2, cfg["sweep_eval_factor"], sdir / "lidar_dsm_6m.tif")
        dtm6 = downsample_geotiff(dtm2, cfg["sweep_eval_factor"], sdir / "lidar_dtm_6m.tif")
        masks2 = strata_masks(lid2, dtm2, site["worldcover"]["path"], cfg)
        masks6 = strata_masks(lid6, dtm6, site["worldcover"]["path"], cfg)
        cop = site["copernicus"]["path"]

        for f in cfg["gsd_factors"]:
            img = site["naip"]["gsd_paths"][str(f)]
            data, meta = read_raster(img)
            rgb = np.moveaxis(data[:3], 0, -1)
            res = (abs(meta.affine.a), abs(meta.affine.e))
            dem, _ = dem_to_image_grid(cop, meta)
            cell = dem_cell_ids(cop, meta)
            anchor_blocks = checkerboard(meta.height, meta.width, max(8, round(cfg["split"]["block_m"] / res[0])))
            lowpass_px = max(1, round(cc["lowpass_m"] / res[0]))
            products = {("a_dem_only", "none"): dem_only(dem)}
            for model in cfg["models"]:
                tile = min(1036, max(meta.height, meta.width))
                pred = get_predictor(model, "cuda", int(math.ceil(tile / 14) * 14))
                t0 = time.perf_counter()
                mosaic, tinfo = run_tiled(rgb, pred, tile=tile, overlap=128 if tile >= 1036 else 0)
                timings.append({"site": site_name, "gsd_m": res[0], "model": model, "tiles": tinfo["n_tiles"],
                                "seconds": round(time.perf_counter() - t0, 2)})
                r = (pred.height_sign * mosaic).astype(np.float32)
                anchors = anchor_pixels(dem, r, anchor_blocks, res, cc["max_slope_deg"])
                b = robust_affine(r, dem, anchors, lowpass_px, robust=cc["robust"], max_px=cc["max_anchor_px"], seed=seed)
                products[("b_robust_affine", model)] = b
                products[("c_dem_plus_residual", model)] = dem_plus_residual(r, dem, cell, b.params["a"])
                products[("c2_dem_plus_smooth_residual", model)] = dem_plus_smooth_residual(r, dem, lowpass_px, b.params["a"])

            for (method, model), cal in products.items():
                zt = np.where(anchor_blocks, np.nan, cal.z).astype(np.float32)      # TEST blocks only
                p = export_dsm(sdir / f"g{f}_{method}_{model}_TEST.tif", zt, meta, model=model, calibration=method,
                               is_metric=True, vertical_datum=COPERNICUS_GLO30.vertical_datum)
                row = {"site": site_name, "gsd_m": round(res[0], 2), "method": method, "model": model,
                       "fit_a": cal.params.get("a") or cal.params.get("scale")}
                row["lidar_6m"], _ = score(p, lid6, masks6)
                if f == 1:
                    row["lidar_2m"], agg2 = score(p, lid2, masks2)
                    row["copernicus_30m"], _ = score(p, cop)
                    if method == "c2_dem_plus_smooth_residual" and model == "da3_mono_large":
                        ref2, tr2, crs2 = read1(lid2)
                        failures.append((site_name, agg2 - ref2, img, tr2))
                    if method == "a_dem_only":
                        ref2, _, _ = read1(lid2)
                        failures.append((site_name + "|dem_only", agg2 - ref2, img, None))
                p.unlink()                                                            # keep runs small
                results.append(row)
                print(f"{site_name:15s} gsd {res[0]:.1f} {method:28s} {model:15s} "
                      f"LiDAR6 RMSE {row['lidar_6m']['overall'].get('rmse')}"
                      + (f" | LiDAR2 RMSE {row['lidar_2m']['overall'].get('rmse')}" if f == 1 else ""), flush=True)

            if f == 1:  # ground estimate + derived height, for the smooth-residual DA3 product
                z = products[("c2_dem_plus_smooth_residual", "da3_mono_large")].z
                g = ground_estimate(z, res[0])
                for name, arr in (("ground", g), ("dsm", z)):
                    export_dsm(sdir / f"chk_{name}.tif", arr, meta, model="da3", calibration="c2", is_metric=True,
                               vertical_datum=COPERNICUS_GLO30.vertical_datum)
                with rasterio.open(sdir / "chk_ground.tif") as ps, rasterio.open(dtm2) as rs:
                    g2, _ = aggregate_to_grid(ps, rs, 0.9)
                with rasterio.open(sdir / "chk_dsm.tif") as ps, rasterio.open(lid2) as rs:
                    z2, _ = aggregate_to_grid(ps, rs, 0.9)
                dtm_a, _, _ = read1(dtm2)
                dsm_a, _, _ = read1(lid2)
                ndsm_ref = dsm_a - dtm_a
                objects = ndsm_ref > 2.0
                ground_checks.append({
                    "site": site_name,
                    "ground_estimate_vs_lidar_dtm": slim(metrics(g2, dtm_a)),
                    "derived_height_vs_lidar_ndsm_on_objects_gt2m": slim(metrics(z2 - g2, ndsm_ref, objects)),
                    "lidar_object_fraction": round(float(objects.mean()), 3)})
                for n in ("chk_ground.tif", "chk_dsm.tif"):
                    (sdir / n).unlink()

    # ---------------------------------------------------------------- failure-case figures
    fig_index = []
    k, win = cfg["failure_cases"]["n_per_site"], cfg["failure_cases"]["window_px"]
    for site_name, err, img, tr2 in [x for x in failures if "|" not in x[0]]:
        ref2, tr2, _ = read1(out / site_name / "lidar_dsm_2m_egm2008.tif")
        pred2 = err + ref2
        base_err = next(e for s, e, _, _ in failures if s == site_name + "|dem_only")
        H, W = err.shape
        cands = []
        for r0 in range(0, H - win + 1, win // 2):
            for c0 in range(0, W - win + 1, win // 2):
                e = err[r0:r0 + win, c0:c0 + win]
                if np.isfinite(e).mean() > 0.5:
                    cands.append((float(np.sqrt(np.nanmean(e ** 2))), r0, c0))
        cands.sort(reverse=True)
        chosen = []
        for rm, r0, c0 in cands:
            if all(abs(r0 - a) >= win or abs(c0 - b) >= win for _, a, b in chosen):
                chosen.append((rm, r0, c0))
            if len(chosen) == k:
                break
        with rasterio.open(img) as src:
            for i, (rm, r0, c0) in enumerate(chosen):
                x0, y0 = tr2 * (c0, r0)
                x1, y1 = tr2 * (c0 + win, r0 + win)
                w = from_bounds(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), src.transform)
                rgb = np.moveaxis(src.read([1, 2, 3], window=w, boundless=True), 0, -1)
                sl = (slice(r0, r0 + win), slice(c0, c0 + win))
                lo, hi = np.nanpercentile(ref2[sl], (2, 98))
                lim = max(2.0, float(np.nanpercentile(np.abs(err[sl]), 98)))
                fig, ax = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
                ax[0].imshow(rgb); ax[0].set_title("NAIP 0.6 m")
                ax[1].imshow(pred2[sl], cmap="terrain", vmin=lo, vmax=hi); ax[1].set_title("DSM (DA3, c2) at 2 m")
                ax[2].imshow(ref2[sl], cmap="terrain", vmin=lo, vmax=hi); ax[2].set_title("LiDAR DSM 2 m")
                im = ax[3].imshow(err[sl], cmap="RdBu_r", vmin=-lim, vmax=lim); ax[3].set_title("DSM minus LiDAR (m)")
                fig.colorbar(im, ax=ax[3], shrink=0.8)
                for a in ax:
                    a.set_xticks([]); a.set_yticks([])
                base_rm = float(np.sqrt(np.nanmean(base_err[sl] ** 2)))
                fig.suptitle(f"{site_name}: worst window #{i + 1} (128 m), RMSE {rm:.1f} m (DEM-only here: {base_rm:.1f} m)")
                name = f"failure_{site_name}_{i + 1}.png"
                fig.savefig(figs / name, dpi=100); plt.close(fig)
                fig_index.append({"site": site_name, "file": name, "rmse_c2_da3": round(rm, 2),
                                  "rmse_dem_only": round(base_rm, 2), "row0": r0, "col0": c0})
        # whole-site error maps: DEM-only vs c2 DA3
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
        for a, (e, t) in zip(ax, ((base_err, "DEM-only minus LiDAR"), (err, "DEM + smooth DA3 detail (c2) minus LiDAR"))):
            im = a.imshow(e, cmap="RdBu_r", vmin=-10, vmax=10); a.set_title(t); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8, label="m (test blocks only; blank = anchor blocks)")
        fig.suptitle(f"{site_name}: error maps at 2 m")
        fig.savefig(figs / f"errormap_{site_name}.png", dpi=90); plt.close(fig)
        fig_index.append({"site": site_name, "file": f"errormap_{site_name}.png"})

    (out / "metrics.json").write_text(json.dumps({"seed": seed, "results": results, "ground_checks": ground_checks,
                                                  "failure_figures": fig_index, "timings": timings,
                                                  "lidar_datum_conversion": datum_records}, indent=2, default=str))
    print("run dir:", out)


if __name__ == "__main__":
    main()
