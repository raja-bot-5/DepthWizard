#!/usr/bin/env python3
"""Fetch Experiment 0 site data from configs/exp0.yaml into data/exp0/<site>/.

Per site: NAIP 0.6 m RGB window, 3DEP LiDAR 2 m DSM window (one named survey, native grid,
snapped, no resampling), Copernicus GLO-30 window. Writes data/exp0/<site>/site.json.

  PYTHONPATH=src python experiments/04_exp0/fetch_sites.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
import yaml

from depthwizard.geo import COPERNICUS_GLO30, fetch_dem_window, footprint_wgs84
from depthwizard.geo.planetary import naip_window, usgs_3dep_dsm
from depthwizard.io import read_raster

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "exp0.yaml").read_text())
    print(f"disk free before: {shutil.disk_usage(ROOT).free / 1e9:.1f} GB (expected download ~25 MB)")
    lidar_src = usgs_3dep_dsm(cfg["lidar_survey"])
    for name, s in cfg["sites"].items():
        d = ROOT / "data" / "exp0" / name
        d.mkdir(parents=True, exist_ok=True)
        tif, nprov = naip_window(s["lon"], s["lat"], cfg["window_px"], d, item_id=s["naip_item"])
        _, meta = read_raster(tif)
        fp = footprint_wgs84(meta, margin_deg=0.001)
        lidar = fetch_dem_window(fp, d, lidar_src, pad_px=2)
        cop = fetch_dem_window(fp, d, COPERNICUS_GLO30, pad_px=2)
        with rasterio.open(lidar.path) as src:
            a = src.read(1)
            lid = {"path": str(lidar.path), "crs": src.crs.to_string(), "res": list(src.res), "shape": list(a.shape),
                   "nan_fraction": float(np.isnan(a).mean()), "min": float(np.nanmin(a)), "max": float(np.nanmax(a)),
                   "tiles": [t["tile"].rsplit("/", 1)[-1] for t in lidar.provenance["tiles_used"]]}
        site = {"name": name, **s, "naip": {"path": str(tif), **{k: nprov[k] for k in ("item", "datetime", "gsd", "window")}},
                "lidar_dsm": lid, "copernicus": {"path": str(cop.path), "shape": cop.provenance["shape"],
                                                 "nan_fraction": cop.provenance["nan_fraction"]}}
        (d / "site.json").write_text(json.dumps(site, indent=2))
        print(json.dumps(site, indent=2))
    print(f"disk free after: {shutil.disk_usage(ROOT).free / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
