#!/usr/bin/env python3
"""A6c: per-building height error vs LiDAR, using OpenStreetMap footprints (ODbL).

Per site (configs/exp8.yaml): DSM from M0 (DEM only) and M2-smooth with each model; derived building height
= median roof interior - median DSM in a ground ring (calibration/ndsm.building_heights).
Reference height = median LiDAR DSM over the eroded footprint - median LiDAR DTM over the footprint.
Only buildings whose centroid lies in a HELD-OUT block are scored; footprints < 50 m2 are skipped.
LiDAR 2013 vs imagery 2021: demolished/new buildings are real outliers -> robust stats (NMAD) reported.

  PYTHONPATH=src python experiments/06_buildings/run_buildings.py
"""
from __future__ import annotations

import json
import sys
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "08_stratified"))
from run_stratified import onto  # noqa: E402

from depthwizard.calibration import checkerboard, dem_cell_ids  # noqa: E402
from depthwizard.calibration.methods import CalibrationInputs, DemOnly, DemResidual  # noqa: E402
from depthwizard.calibration.ndsm import building_heights, estimate_ground  # noqa: E402
from depthwizard.calibration.signal import condition  # noqa: E402
from depthwizard.evaluation import metrics  # noqa: E402
from depthwizard.geo import (COPERNICUS_GLO30, COPERNICUS_HEM, COPERNICUS_WBM, dem_to_image_grid,  # noqa: E402
                             fetch_dem_window, footprint_wgs84)
from depthwizard.geo.osm import ATTRIBUTION, fetch_buildings, rasterize  # noqa: E402
from depthwizard.io import read_raster  # noqa: E402
from depthwizard.pipeline import get_predictor  # noqa: E402
from depthwizard.tiling import run_tiled  # noqa: E402

KEEP = ("n", "bias", "mae", "rmse", "nmad", "le90", "pearson_r")


def lidar_heights(site, buildings) -> dict[int, float]:
    with rasterio.open(site["lidar_dsm"]["path"]) as d, rasterio.open(site["lidar_dtm"]["path"]) as t:
        dsm, dtm = d.read(1).astype(np.float32), t.read(1).astype(np.float32)
        for a, nd in ((dsm, d.nodata), (dtm, t.nodata)):
            if nd is not None:
                a[a == nd] = np.nan
        fp, _ = rasterize(buildings, d.transform, d.crs, dsm.shape, min_area_m2=50)
    import cv2
    out = {}
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for k in np.unique(fp[fp > 0]):
        m = (fp == k).astype(np.uint8)
        inner = cv2.erode(m, kern) > 0
        roof, ground = dsm[inner], dtm[m > 0]
        roof, ground = roof[np.isfinite(roof)], ground[np.isfinite(ground)]
        if roof.size >= 5 and ground.size >= 5:
            out[int(k)] = float(np.median(roof) - np.median(ground))
    return out


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "exp8.yaml").read_text())
    out = ROOT / "runs" / "buildings" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    results, pairs_all = {}, {}
    for site_name in cfg["sites"]:
        site = json.loads((ROOT / "data" / "exp8" / site_name / "site.json").read_text())
        data, meta = read_raster(site["naip"]["path"])
        rgb = np.moveaxis(data[:3], 0, -1)
        res = (abs(meta.affine.a), abs(meta.affine.e))
        fp_wgs = footprint_wgs84(meta)
        osm = fetch_buildings(fp_wgs, ROOT / "data" / "osm")
        fp_img, ids = rasterize(osm["buildings"], meta.affine, meta.crs, (meta.height, meta.width), min_area_m2=50)
        ref = lidar_heights(site, osm["buildings"])
        blocks = checkerboard(meta.height, meta.width, round(cfg["split"]["block_m"] / res[0]))
        # building centroid in a held-out block?
        held = {}
        for k in np.unique(fp_img[fp_img > 0]):
            rr, cc = np.nonzero(fp_img == k)
            held[int(k)] = not blocks[int(rr.mean()), int(cc.mean())]
        cop = fetch_dem_window(footprint_wgs84(meta, margin_deg=0.001), ROOT / "data" / "dem_cache", COPERNICUS_GLO30)
        wbm = fetch_dem_window(footprint_wgs84(meta, margin_deg=0.001), ROOT / "data" / "dem_cache", COPERNICUS_WBM)
        hem = fetch_dem_window(footprint_wgs84(meta, margin_deg=0.001), ROOT / "data" / "dem_cache", COPERNICUS_HEM)
        dem, _ = dem_to_image_grid(cop.path, meta)
        water = np.nan_to_num(onto(wbm.path, site["naip"]["path"], Resampling.nearest)) > 0
        dem_err = onto(hem.path, site["naip"]["path"], Resampling.bilinear)
        print(f"\n=== {site_name}: OSM {osm['counts']}, rasterised {len(ids)} (>= 50 m2), LiDAR heights {len(ref)}, "
              f"held-out {sum(held.values())}", flush=True)
        variants = {"M0_dem_only": None}
        for model in cfg["models"]:
            variants[f"M2_smooth_{model}"] = model
        results[site_name] = {"osm": {k: osm[k] for k in ("counts", "fetched_utc", "xml_sha256")}}
        for name, model in variants.items():
            if model is None:
                z = dem
            else:
                pred = get_predictor(model, "cuda", 1036)
                mosaic, _ = run_tiled(rgb, pred, tile=1036, overlap=128)
                sig = condition(mosaic, pred.height_sign, res[0])
                inp = CalibrationInputs(signal=sig, dem=dem, res=res, fit_mask=blocks, eval_mask=~blocks,
                                        cell_ids=dem_cell_ids(cop.path, meta), water=water, dem_error=dem_err)
                r = DemResidual("smooth").calibrate(inp)
                if not r.is_metric:
                    results[site_name][name] = {"not evaluated": r.report["gate"]["reasons"]}
                    print(f"  {name:28s} gate FAIL: {r.report['gate']['reasons']}", flush=True)
                    continue
                z = r.z
            g = estimate_ground(z, res, water=water)
            est = building_heights(z, g.ground, fp_img)
            ks = [k for k in est if k in ref and held.get(k)]
            e = np.array([est[k]["height_m"] for k in ks]); t = np.array([ref[k] for k in ks])
            m = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics(e, t).items() if k in KEEP}
            by = {}
            for lab, lo, hi in (("<5 m", -99, 5), ("5-15 m", 5, 15), (">=15 m", 15, 999)):
                sel = (t >= lo) & (t < hi)
                if sel.sum() >= 3:
                    by[lab] = {k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in metrics(e[sel], t[sel]).items() if k in KEEP}
            results[site_name][name] = {"overall": m, "by_reference_height": by}
            pairs_all.setdefault(name, []).extend(zip(t.tolist(), e.tolist()))
            print(f"  {name:28s} buildings {m.get('n')}: bias {m.get('bias')} MAE {m.get('mae')} RMSE {m.get('rmse')} "
                  f"NMAD {m.get('nmad')} r {m.get('pearson_r')}", flush=True)
        torch.cuda.empty_cache()
    fig, ax = plt.subplots(1, len(pairs_all), figsize=(4.2 * len(pairs_all), 4), constrained_layout=True)
    for a, (name, pr) in zip(np.atleast_1d(ax), pairs_all.items()):
        pr = np.array(pr)
        a.scatter(pr[:, 0], pr[:, 1], s=8, alpha=0.6)
        lim = [0, max(5.0, float(np.nanmax(pr)) * 1.05)]
        a.plot(lim, lim, "k--", lw=1); a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel("LiDAR building height (m)"); a.set_ylabel("derived height (m)"); a.set_title(name, fontsize=9)
    fig.suptitle(f"Per-building height, held-out buildings, 3 sites. Footprints {ATTRIBUTION}", fontsize=8)
    fig.savefig(out / "buildings_scatter.png", dpi=100)
    (out / "metrics.json").write_text(json.dumps({"attribution": ATTRIBUTION, "results": results}, indent=2))
    print("run dir:", out)


if __name__ == "__main__":
    main()
