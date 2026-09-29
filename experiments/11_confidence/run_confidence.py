#!/usr/bin/env python3
"""Phase 11 T5a: do the derived-height confidence levels separate good from bad heights?

Real pipeline (DA3, Copernicus, default c2, WorldCover on) per Phase 8 site. Derived nDSM (ndsm.tif) is
area-averaged onto the LiDAR 2 m grid; a 2 m cell is LOW if > 50 % of its pixels are LOW. Truth = LiDAR DSM - DTM
(same survey, same datum -> the datum cancels). Scored on held-out test blocks. Reported per level:
all cells, and object cells (LiDAR nDSM > 2 m). Site A: also per building (OSM, majority level in the footprint).

  PYTHONPATH=src python experiments/11_confidence/run_confidence.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "06_buildings"))
from run_buildings import lidar_heights  # noqa: E402

from depthwizard.calibration import checkerboard  # noqa: E402
from depthwizard.calibration.ndsm import REASONS, building_heights  # noqa: E402
from depthwizard.evaluation import aggregate_to_grid, metrics  # noqa: E402
from depthwizard.export import export_dsm  # noqa: E402
from depthwizard.geo import footprint_wgs84  # noqa: E402
from depthwizard.geo.osm import fetch_buildings, rasterize  # noqa: E402
from depthwizard.io import read_raster  # noqa: E402
from depthwizard.pipeline import PipelineConfig, run_pipeline  # noqa: E402

KEEP = ("n", "bias", "mae", "rmse", "nmad", "pearson_r")


def slim(m):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k in KEEP}


def read(p, band=1):
    with rasterio.open(p) as s:
        a = s.read(band).astype(np.float32)
        if s.nodata is not None and not np.isnan(s.nodata) and a.dtype.kind == "f":
            a[a == s.nodata] = np.nan
        return a


def main() -> None:
    cfg8 = yaml.safe_load((ROOT / "configs" / "exp8.yaml").read_text())
    out = ROOT / "runs" / "exp11_confidence" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    rep = {}
    for site_name in cfg8["sites"]:
        site = json.loads((ROOT / "data" / "exp8" / site_name / "site.json").read_text())
        img = site["naip"]["path"]
        _, meta = read_raster(img)
        j = out / site_name
        md = run_pipeline(img, j, PipelineConfig(mesh_max_side=64, dem_cache=str(ROOT / "data" / "dem_cache")))
        for f in j.glob("*.glb"):
            f.unlink()
        summ = next(s for s in md["stages"] if s.get("stage") == "ndsm" and s.get("type") == "stage_done")
        test = ~checkerboard(meta.height, meta.width, max(8, round(154 / abs(meta.affine.a))))
        nd = np.where(test, read(j / "ndsm.tif"), np.nan).astype(np.float32)
        level = read(j / "height_confidence.tif", 1)
        why = read(j / "height_confidence.tif", 2).astype(np.uint8)
        lid_dsm, lid_dtm = site["lidar_dsm"]["path"], site["lidar_dtm"]["path"]
        ref = read(lid_dsm) - read(lid_dtm)
        tp = export_dsm(j / "tmp_nd.tif", nd, meta, model="derived", calibration="nDSM", is_metric=True,
                        vertical_datum="height above derived ground")
        with rasterio.open(tp) as ps, rasterio.open(lid_dsm) as rs:
            agg, _ = aggregate_to_grid(ps, rs, 0.9)
            low = np.zeros((rs.height, rs.width), np.float32)
            reproject((level == 1).astype(np.float32), low, src_transform=meta.affine, src_crs=meta.crs,
                      dst_transform=rs.transform, dst_crs=rs.crs, resampling=Resampling.average)
            bits = {}
            for b in REASONS:
                arr = np.zeros_like(low)
                reproject(((why & b) > 0).astype(np.float32), arr, src_transform=meta.affine, src_crs=meta.crs,
                          dst_transform=rs.transform, dst_crs=rs.crs, resampling=Resampling.average)
                bits[b] = arr > 0.5
        tp.unlink()
        lowc = low > 0.5
        obj = ref > 2.0
        r = {"pipeline_confidence_summary": summ.get("summary", {}).get("height_confidence")}
        for name, sel in (("all_cells", np.ones_like(obj)), ("object_cells_gt2m", obj)):
            r[name] = {"low": slim(metrics(agg, ref, sel & lowc)), "medium": slim(metrics(agg, ref, sel & ~lowc))}
            r[name]["by_reason"] = {REASONS[b]: slim(metrics(agg, ref, sel & m)) for b, m in bits.items() if (sel & m).any()}
        print(site_name, "all:", {k: (v["n"], v.get("rmse")) for k, v in r["all_cells"].items() if k != "by_reason"},
              "objects:", {k: (v["n"], v.get("rmse"), v.get("bias")) for k, v in r["object_cells_gt2m"].items() if k != "by_reason"},
              flush=True)
        # per building (site with buildings)
        osm = fetch_buildings(footprint_wgs84(meta), ROOT / "data" / "osm")
        fp, _ = rasterize(osm["buildings"], meta.affine, meta.crs, (meta.height, meta.width), min_area_m2=50)
        if fp.max() > 0:
            truth = lidar_heights(site, osm["buildings"])
            est = building_heights(read(j / "dsm.tif"), read(j / "ground.tif"), fp)
            groups = {"low": ([], []), "medium": ([], [])}
            for k, v in est.items():
                if k not in truth:
                    continue
                m = fp == k
                rr, cc = np.nonzero(m)
                if not test[int(rr.mean()), int(cc.mean())]:
                    continue
                g = "low" if (level[m] == 1).mean() > 0.5 else "medium"
                groups[g][0].append(v["height_m"])
                groups[g][1].append(truth[k])
            r["buildings_held_out"] = {g: slim(metrics(np.array(e), np.array(t))) if len(e) >= 3 else {"n": len(e)}
                                       for g, (e, t) in groups.items()}
            print("   buildings:", r["buildings_held_out"], flush=True)
        rep[site_name] = r
    (out / "confidence_eval.json").write_text(json.dumps(rep, indent=2, default=str))
    print("run dir:", out)


if __name__ == "__main__":
    main()
