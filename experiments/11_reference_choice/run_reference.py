#!/usr/bin/env python3
"""Phase 11 T2: how much does the score depend on which 30 m DEM the organisers score against?

The organisers score GeoTIFF output against "an absolute DSM such as SRTM or Copernicus" (FAQ). For each Phase 8
site (NAIP 0.6 m) the REAL pipeline (run_pipeline, DA3, default calibration c2 = dem_plus_smooth_residual) runs
twice: calibrated to Copernicus GLO-30 and calibrated to SRTM (NASADEM). Products per run:
  (a)  dem.tif  - the calibration DEM bilinear on the image grid (EGM2008)
  (c2) dsm.tif  - DEM + smooth DA3 detail
Each product is scored on held-out TEST blocks (same 154 m checkerboard the pipeline fits on) against:
  Copernicus GLO-30 (EGM2008 native), SRTM = NASADEM (EGM96 -> EGM2008), Tilezen skadi (EGM96 -> EGM2008; over the
  USA it is 3DEP/NED-derived, NOT SRTM), and the independent LiDAR DSM 2 m (NAVD88 -> EGM2008).
Two scales for the 30 m references:
  native_0.6m : reference BILINEAR-UPSAMPLED to the 0.6 m image grid, compared pixel by pixel (labelled as such)
  agg_30m     : product area-averaged onto the reference's own 1" grid (cells fully inside, >= 90 % test pixels)
LiDAR is scored at 2 m (area-averaged), as in Phase 8.

  PYTHONPATH=src python experiments/11_reference_choice/run_reference.py
"""
from __future__ import annotations

import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import yaml

from depthwizard.calibration import checkerboard
from depthwizard.calibration.datum import raster_file_to_egm2008
from depthwizard.evaluation import aggregate_to_grid, metrics
from depthwizard.export import export_dsm
from depthwizard.geo import COPERNICUS_GLO30, TILEZEN_SKADI, dem_to_image_grid, fetch_dem_window, footprint_wgs84
from depthwizard.geo.planetary import NASADEM
from depthwizard.io import read_raster
from depthwizard.pipeline import PipelineConfig, run_pipeline

ROOT = Path(__file__).resolve().parents[2]
KEEP = ("n", "bias", "mae", "rmse", "nmad", "le90", "pearson_r")
REFS = {"copernicus": COPERNICUS_GLO30, "srtm_nasadem": NASADEM, "skadi": TILEZEN_SKADI}
CALIBRATE_TO = {"copernicus": "copernicus", "srtm_nasadem": "nasadem"}      # label -> PipelineConfig.dem_source


def slim(m: dict) -> dict:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k in KEEP}


def read_nan(path) -> np.ndarray:
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float32)
        if s.nodata is not None and not np.isnan(s.nodata):
            a[a == s.nodata] = np.nan
    return a


def main() -> None:
    cfg8 = yaml.safe_load((ROOT / "configs" / "exp8.yaml").read_text())
    out = ROOT / "runs" / "exp11_reference" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    rows, dem_vs_dem, refs_info, runs = [], [], {}, []
    for site_name in cfg8["sites"]:
        site = json.loads((ROOT / "data" / "exp8" / site_name / "site.json").read_text())
        sdir = out / site_name
        sdir.mkdir()
        img = site["naip"]["path"]
        data, meta = read_raster(img)
        res = abs(meta.affine.a)
        test = ~checkerboard(meta.height, meta.width, max(8, round(154 / res)))   # pipeline's fit blocks = True

        # references, all EGM2008, on their native grids (+ bilinear on the image grid)
        fp = footprint_wgs84(meta, margin_deg=0.003)
        ref30, ref06 = {}, {}
        for name, src in REFS.items():
            win = fetch_dem_window(fp, ROOT / "data" / "dem_cache", src)
            p, info = raster_file_to_egm2008(win.path, sdir / f"ref_{name}_egm2008.tif")
            ref30[name], refs_info[f"{site_name}:{name}"] = p, {"source": src.name, **info}
            ref06[name] = dem_to_image_grid(p, meta)[0]
        lid, linfo = raster_file_to_egm2008(site["lidar_dsm"]["path"], sdir / "lidar_dsm_2m_egm2008.tif")
        refs_info[f"{site_name}:lidar"] = linfo

        # how far apart are the candidate references themselves? (whole footprint, 30 m, Copernicus grid)
        cop = read_nan(ref30["copernicus"])
        for name in ("srtm_nasadem", "skadi"):
            with rasterio.open(ref30[name]) as s, rasterio.open(ref30["copernicus"]) as r:
                assert s.transform == r.transform and s.shape == r.shape, "reference grids differ"
            dem_vs_dem.append({"site": site_name, "a": name, "b": "copernicus",
                               **slim(metrics(read_nan(ref30[name]), cop))})

        for cal_label, dem_source in CALIBRATE_TO.items():
            jdir = sdir / f"job_{cal_label}"
            pc = PipelineConfig(dem_source=dem_source, dem_cache=str(ROOT / "data" / "dem_cache"), mesh_max_side=64)
            t0 = time.perf_counter()
            md = run_pipeline(img, jdir, pc)
            runs.append({"site": site_name, "calibrated_to": cal_label, "seconds": round(time.perf_counter() - t0, 1),
                         "method": md["product"]["calibration"], "calibration_dem": md["product"]["calibration_dem"],
                         "gate": md["product"]["gate"]["passed"]})
            with rasterio.open(jdir / "dsm.tif") as s:
                tags = s.tags()
            assert tags.get("CALIBRATION_DEM", "").startswith(REFS[cal_label].name[:12]), tags.get("CALIBRATION_DEM")
            for prod_label, fname in (("a_dem_only", "dem.tif"), ("c2_dem_plus_smooth_residual", "dsm.tif")):
                z = read_nan(jdir / fname)
                zt = np.where(test, z, np.nan).astype(np.float32)
                tp = export_dsm(sdir / "tmp_test.tif", zt, meta, model="da3_mono_large", calibration=prod_label,
                                is_metric=True, vertical_datum="EGM2008")
                row = {"site": site_name, "calibrated_to": cal_label, "product": prod_label, "scores": {}}
                for ref_name in REFS:
                    with rasterio.open(tp) as ps, rasterio.open(ref30[ref_name]) as rs:
                        agg, _ = aggregate_to_grid(ps, rs, 0.9)
                    row["scores"][f"{ref_name}|agg_30m"] = slim(metrics(agg, read_nan(ref30[ref_name])))
                    row["scores"][f"{ref_name}|native_0.6m_bilinear_ref"] = slim(metrics(zt, ref06[ref_name]))
                with rasterio.open(tp) as ps, rasterio.open(lid) as rs:
                    agg, _ = aggregate_to_grid(ps, rs, 0.9)
                row["scores"]["lidar|2m"] = slim(metrics(agg, read_nan(lid)))
                rows.append(row)
                print(site_name, cal_label, prod_label, {k: v.get("rmse") for k, v in row["scores"].items()}, flush=True)
            tp.unlink()
            for f in jdir.glob("*.glb"):
                f.unlink()                                                     # keep the run small
    (out / "metrics.json").write_text(json.dumps({"rows": rows, "dem_vs_dem": dem_vs_dem, "runs": runs,
                                                  "references": refs_info}, indent=2, default=str))
    shutil.copy(ROOT / "configs" / "exp8.yaml", out / "sites_config.yaml")
    print("run dir:", out)


if __name__ == "__main__":
    main()
