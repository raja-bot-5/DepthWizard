#!/usr/bin/env python3
"""Fetch Phase 8 data per site (configs/exp8.yaml) into data/exp8/<site>/:
NAIP 0.6 m window (+ x2, x4 block-mean copies with rescaled transforms), 3DEP LiDAR DSM and DTM 2 m
(same survey, native grid), ESA WorldCover 10 m, Copernicus GLO-30. Writes site.json.

  PYTHONPATH=src python experiments/08_stratified/fetch_sites8.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
import yaml

from depthwizard.geo import COPERNICUS_GLO30, fetch_dem_window, footprint_wgs84
from depthwizard.geo.planetary import esa_worldcover, naip_window, usgs_3dep
from depthwizard.io import downsample_geotiff, read_raster

ROOT = Path(__file__).resolve().parents[2]


def summary(path: Path) -> dict:
    with rasterio.open(path) as s:
        a = s.read(1)
        return {"path": str(path), "crs": s.crs.to_string()[:80], "res": list(s.res), "shape": list(a.shape),
                "nan_fraction": float(np.isnan(a).mean()), "min": float(np.nanmin(a)), "max": float(np.nanmax(a))}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "exp8.yaml").read_text())
    print(f"disk free: {shutil.disk_usage(ROOT).free / 1e9:.1f} GB (expected ~60 MB for 3 sites)")
    dsm_src, dtm_src = usgs_3dep(cfg["lidar_survey"], "dsm"), usgs_3dep(cfg["lidar_survey"], "dtm")
    wc_src = esa_worldcover(2021)
    for name, s in cfg["sites"].items():
        d = ROOT / "data" / "exp8" / name
        d.mkdir(parents=True, exist_ok=True)
        tif, nprov = naip_window(s["lon"], s["lat"], cfg["window_px"], d, item_id=s["naip_item"])
        gsd = {"1": str(tif)}
        for f in cfg["gsd_factors"]:
            if f > 1:
                gsd[str(f)] = str(downsample_geotiff(tif, f, d / f"naip_x{f}.tif"))
        _, meta = read_raster(tif)
        fp = footprint_wgs84(meta, margin_deg=0.001)
        dsm = fetch_dem_window(fp, d, dsm_src, pad_px=2)
        dtm = fetch_dem_window(fp, d, dtm_src, pad_px=2)
        wc = fetch_dem_window(fp, d, wc_src, pad_px=2)
        cop = fetch_dem_window(fp, d, COPERNICUS_GLO30, pad_px=2)
        with rasterio.open(dsm.path) as a, rasterio.open(dtm.path) as b:
            same_grid = (a.crs == b.crs) and (a.transform == b.transform) and (a.shape == b.shape)
        site = {"name": name, **s, "naip": {"path": str(tif), "item": nprov["item"], "gsd_paths": gsd},
                "lidar_dsm": summary(dsm.path), "lidar_dtm": summary(dtm.path), "dsm_dtm_same_grid": same_grid,
                "worldcover": {"path": str(wc.path), "source": wc_src.name},
                "copernicus": summary(cop.path)}
        (d / "site.json").write_text(json.dumps(site, indent=2))
        print(name, "| DSM", site["lidar_dsm"]["shape"], f"{site['lidar_dsm']['min']:.0f}-{site['lidar_dsm']['max']:.0f} m",
              "| DTM", site["lidar_dtm"]["shape"], "| same grid:", same_grid, "| GSD copies:", list(gsd))
    print(f"disk free after: {shutil.disk_usage(ROOT).free / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
