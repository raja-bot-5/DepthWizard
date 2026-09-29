#!/usr/bin/env python3
"""Experiment 0 v2: calibration methods M0 / M1 / M2(cell) / M2(smooth) x {da2_small, da3_mono_large}
on the three Phase 8 sites, everything in EGM2008.

- LiDAR DSM/DTM (NAVD88) converted to EGM2008 with GEOID18 before any comparison (datum.to_egm2008).
- Copernicus DEM is EGM2008 natively (verified from tile XML); water from Copernicus WBM; DEM error from HEM.
- Spatial split: checkerboard (fit blocks / held-out blocks). Only held-out blocks are scored.
- A method whose quality gate fails produces NO metric DSM: it is reported as "rDSM fallback, not evaluated".
- Uncertainty honesty check: fraction of |error| <= sigma and <= 2 sigma vs LiDAR (ideal 1-sigma: ~68 % / ~95 %).

  PYTHONPATH=src python experiments/05_calibration_methods/run_methods.py
"""
from __future__ import annotations

import json
import math
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "08_stratified"))
from run_stratified import onto, read1, strata_masks  # noqa: E402

from depthwizard.calibration import checkerboard, dem_cell_ids  # noqa: E402
from depthwizard.calibration.datum import to_egm2008  # noqa: E402
from depthwizard.calibration.methods import CalibrationInputs, GateConfig, METHODS  # noqa: E402
from depthwizard.calibration.signal import condition  # noqa: E402
from depthwizard.evaluation import aggregate_to_grid, metrics  # noqa: E402
from depthwizard.export import export_dsm  # noqa: E402
from depthwizard.geo import (COPERNICUS_GLO30, COPERNICUS_HEM, COPERNICUS_WBM, dem_to_image_grid,  # noqa: E402
                             fetch_dem_window, footprint_wgs84)
from depthwizard.io import read_raster  # noqa: E402
from depthwizard.pipeline import get_predictor  # noqa: E402
from depthwizard.tiling import run_tiled  # noqa: E402

KEEP = ("n", "bias", "mae", "rmse", "nmad", "le90", "pearson_r", "r2")


def slim(m):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k in KEEP}


def lidar_egm2008(path: str, out: Path, kind: str) -> Path:
    with rasterio.open(path) as s:
        z = s.read(1).astype(np.float32)
        if s.nodata is not None and not np.isnan(s.nodata):
            z[z == s.nodata] = np.nan
        conv, info = to_egm2008(z, s.transform, s.crs, s.tags().get("VERTICAL_DATUM", "NAVD88 height"))
        prof = s.profile.copy()
    prof.update(dtype="float32", nodata=np.nan)
    with rasterio.open(out, "w", **prof) as d:
        d.write(conv, 1)
        d.update_tags(VERTICAL_DATUM="EGM2008 (converted from NAVD88 via GEOID18)", UNITS="metres",
                      DATUM_CONVERSION=json.dumps(info))
    return out, info


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "exp8.yaml").read_text())
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = ROOT / "runs" / "exp0_v2" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / "reports").mkdir(parents=True)
    shutil.copy(ROOT / "configs" / "exp8.yaml", out / "config.yaml")
    gate = GateConfig()
    rows, datum_records = [], {}

    for site_name in cfg["sites"]:
        site = json.loads((ROOT / "data" / "exp8" / site_name / "site.json").read_text())
        sdir = out / site_name
        sdir.mkdir()
        lid, lid_info = lidar_egm2008(site["lidar_dsm"]["path"], sdir / "lidar_dsm_egm2008.tif", "dsm")
        datum_records[site_name] = lid_info
        masks = strata_masks(lid, site["lidar_dtm"]["path"], site["worldcover"]["path"], cfg)
        data, meta = read_raster(site["naip"]["path"])
        rgb = np.moveaxis(data[:3], 0, -1)
        res = (abs(meta.affine.a), abs(meta.affine.e))
        fp = footprint_wgs84(meta, margin_deg=0.001)
        cop = fetch_dem_window(fp, ROOT / "data" / "dem_cache", COPERNICUS_GLO30)
        wbm = fetch_dem_window(fp, ROOT / "data" / "dem_cache", COPERNICUS_WBM)
        hem = fetch_dem_window(fp, ROOT / "data" / "dem_cache", COPERNICUS_HEM)
        dem, _ = dem_to_image_grid(cop.path, meta)                 # EGM2008 native
        with rasterio.open(site["naip"]["path"]) as ref_img:
            ref_like = site["naip"]["path"]
        water = np.nan_to_num(onto(wbm.path, ref_like, Resampling.nearest)) > 0
        dem_err = onto(hem.path, ref_like, Resampling.bilinear)
        cell = dem_cell_ids(cop.path, meta)
        blocks = checkerboard(meta.height, meta.width, round(cfg["split"]["block_m"] / res[0]))
        print(f"\n=== {site_name}: water {water.mean():.1%}, LiDAR NAVD88->EGM2008 offset "
              f"{lid_info['offset_min_m']:+.2f}..{lid_info['offset_max_m']:+.2f} m", flush=True)

        for model in cfg["models"]:
            pred = get_predictor(model, "cuda", 1036)
            mosaic, _ = run_tiled(rgb, pred, tile=1036, overlap=128)
            sig = condition(mosaic, pred.height_sign, res[0], polarity=pred.polarity)
            inp = CalibrationInputs(signal=sig, dem=dem, res=res, fit_mask=blocks, eval_mask=~blocks, cell_ids=cell,
                                    water=water, dem_error=dem_err, seed=seed)
            for key in ("M0", "M1", "M2_cell", "M2_smooth"):
                if key == "M0" and model != cfg["models"][0]:
                    continue                                             # model-independent
                res_ = METHODS[key]().calibrate(inp, gate)
                tag = f"{site_name}__{res_.method}__{'none' if key == 'M0' else model}"
                report = {**res_.report, "site": site_name, "model": None if key == "M0" else model,
                          "lidar_reference_datum": lid_info}
                row = {"site": site_name, "method": res_.method, "model": None if key == "M0" else model,
                       "gate_passed": res_.report["gate"]["passed"], "gate_reasons": res_.report["gate"]["reasons"]}
                if res_.is_metric:
                    zt = np.where(blocks, np.nan, res_.z).astype(np.float32)          # held-out blocks only
                    p = export_dsm(sdir / f"{tag}.tif", zt, meta, model=model, calibration=res_.method, is_metric=True,
                                   vertical_datum="EGM2008")
                    u2 = export_dsm(sdir / f"{tag}_var.tif", np.where(blocks, np.nan, res_.uncertainty ** 2),
                                    meta, model=model, calibration="variance", is_metric=True, vertical_datum="EGM2008")
                    with rasterio.open(p) as ps, rasterio.open(lid) as rs:
                        agg, _ = aggregate_to_grid(ps, rs, 0.9)
                    with rasterio.open(u2) as ps, rasterio.open(lid) as rs:
                        var, _ = aggregate_to_grid(ps, rs, 0.9)
                    ref, _, _ = read1(lid)
                    err = agg - ref
                    sig_ = np.sqrt(var)
                    ok = np.isfinite(err) & np.isfinite(sig_) & (sig_ > 0)
                    row["lidar_2m_egm2008"] = slim(metrics(agg, ref))
                    row["strata"] = {k: slim(metrics(agg, ref, m)) for k, m in masks.items() if m.sum() >= 500}
                    row["sigma_coverage"] = {"n": int(ok.sum()),
                                             "within_1sigma": float((np.abs(err[ok]) <= sig_[ok]).mean()) if ok.any() else None,
                                             "within_2sigma": float((np.abs(err[ok]) <= 2 * sig_[ok]).mean()) if ok.any() else None,
                                             "median_sigma_m": float(np.median(sig_[ok])) if ok.any() else None}
                    with rasterio.open(p) as ps, rasterio.open(cop.path) as rs:
                        agg30, _ = aggregate_to_grid(ps, rs, 0.9)
                    c30, _, _ = read1(cop.path)
                    row["copernicus_30m"] = slim(metrics(agg30, c30))
                    for f in (p, u2):
                        f.unlink(); f.with_suffix(".json").unlink()
                else:
                    row["lidar_2m_egm2008"] = "not evaluated (quality gate failed -> rDSM fallback)"
                report["evaluation"] = {k: row.get(k) for k in ("lidar_2m_egm2008", "sigma_coverage", "copernicus_30m")}
                (out / "reports" / f"{tag}.json").write_text(json.dumps(report, indent=2, default=str))
                rows.append(row)
                l = row["lidar_2m_egm2008"]
                if isinstance(l, dict) and "rmse" not in l:
                    raise RuntimeError(f"{tag}: no valid comparison cells - check the reference conversion")
                print(f"  {res_.method:22s} {str(row['model']):15s} gate={'PASS' if row['gate_passed'] else 'FAIL'} "
                      + (f"RMSE {l['rmse']} bias {l['bias']} NMAD {l['nmad']} LE90 {l['le90']} | "
                         f"1s/2s cover {row['sigma_coverage']['within_1sigma']:.2f}/{row['sigma_coverage']['within_2sigma']:.2f}"
                         if isinstance(l, dict) else "; ".join(row["gate_reasons"])[:160]), flush=True)
            del pred
            torch.cuda.empty_cache()

    (out / "metrics.json").write_text(json.dumps({"seed": seed, "gate": gate.__dict__, "rows": rows,
                                                  "lidar_datum_conversion": datum_records}, indent=2, default=str))
    print("run dir:", out)


if __name__ == "__main__":
    main()
