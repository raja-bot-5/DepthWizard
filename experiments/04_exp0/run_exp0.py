#!/usr/bin/env python3
"""Experiment 0: {DEM-only, robust affine, DEM + zero-mean residual} x {da2_small, da3_mono_large}
on each site of configs/exp0.yaml. Scored on TEST blocks only against
  - LiDAR DSM (independent; 2 m native grid; prediction area-averaged 0.6 -> 2 m)
  - Copernicus GLO-30 (30 m; prediction area-averaged). NOTE: (a) IS Copernicus -> circular there.

Writes runs/exp0/<timestamp>/{config.yaml, metrics.json, results.md, provenance.json, <site>/*.tif}.

  PYTHONPATH=src python experiments/04_exp0/run_exp0.py
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml

from depthwizard.calibration import (anchor_pixels, checkerboard, dem_cell_ids, dem_only, dem_plus_residual,
                                     robust_affine)
from depthwizard.depth import PREDICTORS
from depthwizard.evaluation import evaluate_rasters
from depthwizard.export import export_dsm
from depthwizard.geo import COPERNICUS_GLO30, dem_to_image_grid
from depthwizard.io import read_raster
from depthwizard.tiling import run_tiled

ROOT = Path(__file__).resolve().parents[2]
KEEP = ("n", "bias", "mae", "rmse", "nmad", "pearson_r", "r2")


def slim(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in KEEP}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "exp0.yaml").read_text())
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = ROOT / "runs" / "exp0" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "configs" / "exp0.yaml", out / "config.yaml")
    prov = {"git": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            or "no commits (git identity unset)", "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "provenance_txt": (ROOT / "setup" / "provenance.txt").read_text(), "sites": {}}
    results: list[dict] = []
    ccfg = cfg["calibration"]

    for site_name in cfg["sites"]:
        site = json.loads((ROOT / "data" / "exp0" / site_name / "site.json").read_text())
        prov["sites"][site_name] = site
        sdir = out / site_name
        sdir.mkdir(exist_ok=True)
        data, meta = read_raster(site["naip"]["path"])
        rgb = np.moveaxis(data[:3], 0, -1)
        res = (abs(meta.affine.a), abs(meta.affine.e))
        dem, rinfo = dem_to_image_grid(site["copernicus"]["path"], meta)
        cell = dem_cell_ids(site["copernicus"]["path"], meta)
        anchor_blocks = checkerboard(meta.height, meta.width, cfg["split"]["block_px"])
        test_blocks = ~anchor_blocks
        lowpass_px = int(round(ccfg["lowpass_m"] / res[0]))

        def score(z: np.ndarray, tag: str, method: str, model: str, params: dict) -> None:
            zt = np.where(test_blocks, z, np.nan).astype(np.float32)   # test pixels only
            p = export_dsm(sdir / f"{tag}_TEST.tif", zt, meta, model=model, calibration=method, is_metric=True,
                           vertical_datum=COPERNICUS_GLO30.vertical_datum,
                           provenance={"site": site_name, "params": params, "split": "test blocks only"})
            row = {"site": site_name, "method": method, "model": model, "params": params}
            for ref_name, ref_path in (("lidar_2m", site["lidar_dsm"]["path"]), ("copernicus_30m", site["copernicus"]["path"])):
                rep = evaluate_rasters(p, ref_path, mode="aggregate", min_coverage=0.9)
                row[ref_name] = {"overall": slim(rep["overall"]), "datum_check": rep["vertical_datum"]["check"],
                                 "by_slope": {k: slim(v) for k, v in rep["by_slope"].items()}}
            results.append(row)
            l, c = row["lidar_2m"]["overall"], row["copernicus_30m"]["overall"]
            print(f"  {site_name:15s} {method:20s} {model:15s} LiDAR RMSE {l.get('rmse')} bias {l.get('bias')} r {l.get('pearson_r')}"
                  f" | Cop RMSE {c.get('rmse')}", flush=True)

        print(f"\n=== {site_name}: DEM coverage {rinfo['coverage_fraction']:.3f}, lowpass {lowpass_px} px", flush=True)
        score(dem_only(dem).z, "a_dem_only", "a_dem_only", "none", {"resampling": "bilinear"})

        for model in cfg["models"]:
            kw = {"device": "cuda", "process_res": cfg["tiling"]["tile"]}
            if model == "da2_small":
                kw["fp16"] = True
            pred = PREDICTORS[model](**kw)
            t0 = time.perf_counter()
            mosaic, tinfo = run_tiled(rgb, pred, tile=cfg["tiling"]["tile"], overlap=cfg["tiling"]["overlap"])
            infer_s = time.perf_counter() - t0
            r = (pred.height_sign * mosaic).astype(np.float32)
            export_dsm(sdir / f"rdsm_{model}.tif", r, meta, model=model, calibration="none", is_metric=False,
                       provenance={"tiles": tinfo["n_tiles"], "height_sign": pred.height_sign})
            anchors = anchor_pixels(dem, r, anchor_blocks, res, ccfg["max_slope_deg"])
            b = robust_affine(r, dem, anchors, lowpass_px, robust=ccfg["robust"], max_px=ccfg["max_anchor_px"], seed=seed)
            common = {"inference_s": round(infer_s, 2), "n_tiles": tinfo["n_tiles"],
                      "anchor_px": int(anchors.sum()), "water_masked": False}
            score(b.z, f"b_{model}", "b_robust_affine", model, {**b.params, **common})
            c = dem_plus_residual(r, dem, cell, scale=b.params["a"])
            score(c.z, f"c_{model}", "c_dem_plus_residual", model, {**c.params, **common})
            del pred
            torch.cuda.empty_cache()

    (out / "metrics.json").write_text(json.dumps({"seed": seed, "results": results}, indent=2))
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))

    lines = ["# Experiment 0 results", "",
             "Test blocks only (checkerboard, anchors disjoint). Heights in metres; product datum = Copernicus "
             "(EGM2008, VERIFY); LiDAR = NAVD88 -> **bias vs LiDAR includes a datum offset**. "
             "(a) vs Copernicus is circular (it IS Copernicus). LiDAR 2013 vs NAIP 2021.", ""]
    for ref in ("lidar_2m", "copernicus_30m"):
        lines += [f"## vs {ref}", "", "| site | method | model | n | bias | RMSE | MAE | NMAD | r | R2 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in results:
            m = row[ref]["overall"]
            lines.append(f"| {row['site']} | {row['method']} | {row['model']} | {m.get('n')} | {m.get('bias')} | "
                         f"{m.get('rmse')} | {m.get('mae')} | {m.get('nmad')} | {m.get('pearson_r')} | {m.get('r2')} |")
        lines.append("")
    (out / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("run dir:", out)


if __name__ == "__main__":
    main()
