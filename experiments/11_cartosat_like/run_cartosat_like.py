#!/usr/bin/env python3
"""Phase 11 T4: is the pipeline robust to Cartosat-like input? Synthetic variants of the site A NAIP scene.

  base          : NAIP 0.6 m, 3-band uint8 RGB (colour interpretation declared)            -> reference run
  bgrn_u16      : 4 bands B,G,R,NIR (NIR = SYNTHETIC copy of green: NAIP window has no NIR), uint16 = round(x*2047/255),
                  NBITS=11, colour interpretation undeclared; run with band_order=bgrn     -> must equal base exactly
  bgrn_u16_auto : same file, band_order=auto                                              -> must WARN (order undeclared)
  pan_u16       : single band (luminance), uint16 11-bit                                   -> runs, scored vs LiDAR
  gsd065        : RGB resampled to 0.65 m (average), transform recomputed for the new grid -> scored vs LiDAR
                  (cannot be identical to base: different pixels)
All metric runs are scored on held-out test blocks vs LiDAR 2 m (EGM2008), as in Phase 8/11.
Model: DA-V2 Small (the app server held DA3 in VRAM during this run). The identity check does not depend on
the model: identical model input must give an identical DSM.

  PYTHONPATH=src python experiments/11_cartosat_like/run_cartosat_like.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.warp import reproject

from depthwizard.calibration import checkerboard
from depthwizard.calibration.datum import raster_file_to_egm2008
from depthwizard.evaluation import aggregate_to_grid, metrics
from depthwizard.export import export_dsm
from depthwizard.io import read_raster
from depthwizard.pipeline import PipelineConfig, run_pipeline

ROOT = Path(__file__).resolve().parents[2]
KEEP = ("n", "bias", "mae", "rmse", "pearson_r")


def write(path, arr, profile, **kw):
    prof = {**profile, "count": arr.shape[0], "dtype": str(arr.dtype), **kw}
    prof.pop("photometric", None)
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr)
        d.colorinterp = [ColorInterp.undefined] * arr.shape[0]
    return path


def main() -> None:
    site = json.loads((ROOT / "data" / "exp8" / "A_urban" / "site.json").read_text())
    src = site["naip"]["path"]
    out = ROOT / "runs" / "exp11_cartosat_like" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / "inputs").mkdir(parents=True)
    with rasterio.open(src) as s:
        rgb = s.read()                                         # (3, H, W) uint8, R G B
        prof = s.profile.copy()
        tr, crs, H, W = s.transform, s.crs, s.height, s.width
    prof.update(compress="deflate", nodata=None)
    u16 = lambda x: np.round(x.astype(np.float64) * 2047 / 255).astype(np.uint16)   # noqa: E731
    bgrn = np.stack([rgb[2], rgb[1], rgb[0], rgb[1]])          # B G R NIR(synthetic = G)
    f_bgrn = write(out / "inputs" / "bgrn_u16.tif", u16(bgrn), prof, nbits=11)
    lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])[None]
    f_pan = write(out / "inputs" / "pan_u16.tif", u16(lum), prof, nbits=11)
    # 0.65 m: new grid with the SAME bounds, transform recomputed (never keep the old transform)
    f = 0.65 / abs(tr.a)
    W2, H2 = int(round(W / f)), int(round(H / f))
    tr2 = rasterio.Affine(tr.a * W / W2, 0, tr.c, 0, tr.e * H / H2, tr.f)
    g = np.zeros((3, H2, W2), np.uint8)
    for b in range(3):
        reproject(rgb[b], g[b], src_transform=tr, src_crs=crs, dst_transform=tr2, dst_crs=crs, resampling=Resampling.average)
    p2 = {**prof, "width": W2, "height": H2, "transform": tr2}
    f_gsd = out / "inputs" / "gsd065.tif"
    with rasterio.open(f_gsd, "w", **{**p2, "count": 3, "dtype": "uint8"}) as d:
        d.write(g)
        d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]

    lid, _ = raster_file_to_egm2008(site["lidar_dsm"]["path"], out / "lidar_egm2008.tif")
    runs = {"base": (src, "auto"), "bgrn_u16": (f_bgrn, "bgrn"), "bgrn_u16_auto": (f_bgrn, "auto"),
            "pan_u16": (f_pan, "auto"), "gsd065": (f_gsd, "auto")}
    rep = {"note": "NIR band is a synthetic copy of green (NAIP window has 3 bands); uint16 = round(uint8*2047/255), NBITS=11"}
    for name, (path, bo) in runs.items():
        j = out / name
        md = run_pipeline(path, j, PipelineConfig(model="da2_small", band_order=bo, mesh_max_side=64, dem_cache=str(ROOT / "data" / "dem_cache")))
        for x in j.glob("*.glb"):
            x.unlink()
        _, meta = read_raster(path)
        r = {"band_order": bo, "model_input": md["input"]["model_input"], "warnings": md["input"]["warnings"],
             "kind": md["product"]["kind"], "gsd_m": md["input"]["gsd_m"]}
        if (j / "dsm.tif").exists():
            test = ~checkerboard(meta.height, meta.width, max(8, round(154 / abs(meta.affine.a))))
            with rasterio.open(j / "dsm.tif") as d:
                z = d.read(1)
            tp = export_dsm(j / "tmp.tif", np.where(test, z, np.nan).astype(np.float32), meta, model="x",
                            calibration="x", is_metric=True, vertical_datum="EGM2008")
            with rasterio.open(tp) as ps, rasterio.open(lid) as rs:
                agg, _ = aggregate_to_grid(ps, rs, 0.9)
                ref = rs.read(1)
            r["vs_lidar_2m"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics(agg, ref).items() if k in KEEP}
            tp.unlink()
        rep[name] = r
        print(name, r.get("vs_lidar_2m", {}).get("rmse"), r["model_input"], r["warnings"], flush=True)
    with rasterio.open(out / "base" / "dsm.tif") as a, rasterio.open(out / "bgrn_u16" / "dsm.tif") as b:
        za, zb = a.read(1), b.read(1)
        rep["bgrn_u16_vs_base"] = {"same_grid": a.transform == b.transform and a.crs == b.crs and a.shape == b.shape,
                                   "max_abs_diff_m": float(np.nanmax(np.abs(za - zb))),
                                   "identical": bool(np.array_equal(za, zb, equal_nan=True))}
    with rasterio.open(out / "base" / "rdsm.tif") as a, rasterio.open(out / "bgrn_u16" / "rdsm.tif") as b:
        rep["bgrn_u16_vs_base"]["rdsm_identical"] = bool(np.array_equal(a.read(1), b.read(1), equal_nan=True))
    print(json.dumps(rep["bgrn_u16_vs_base"]))
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    print("run dir:", out)


if __name__ == "__main__":
    main()
