#!/usr/bin/env python3
"""Phase 2 geo spine, end to end on real DEM data.

GeoTIFF -> WGS84 footprint -> Copernicus GLO-30 window (range reads, cached)
-> DEM bilinear-resampled onto the image grid -> float32 GeoTIFF with datum tags.

If --image is omitted, a SYNTHETIC GeoTIFF (random pixels, real georeferencing,
0.6 m, near Dehradun) is written so the chain can run without real imagery.
Reported heights are the DEM's own values on the image grid, NOT an accuracy result.

Usage
  PYTHONPATH=src python experiments/01_geo_spine/run_geo_spine.py [--image scene.tif]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from depthwizard.geo import COPERNICUS_GLO30, dem_to_image_grid, fetch_dem_window, footprint_wgs84
from depthwizard.io import read_raster, write_dsm

ROOT = Path(__file__).resolve().parents[2]


def synthetic_geotiff(path: Path, size: int = 2000, res: float = 0.6) -> Path:
    # upper-left ~78.03E 30.32N (Dehradun, hilly), UTM 44N
    rng = np.random.default_rng(0)
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8",
                       crs="EPSG:32644", transform=from_origin(214_400.0, 3_358_000.0, res, res)) as dst:
        dst.write(rng.integers(0, 255, (3, size, size), dtype=np.uint8))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image")
    ap.add_argument("--cache", default=str(ROOT / "data" / "dem_cache"))
    ap.add_argument("--margin-deg", type=float, default=0.001, help="> 1 DEM pixel (~0.00028 deg)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else ROOT / "runs" / "geo_spine" / stamp
    out.mkdir(parents=True, exist_ok=True)

    image = Path(args.image) if args.image else synthetic_geotiff(out / "SYNTHETIC_input.tif")
    _, meta = read_raster(image)
    if not meta.is_georeferenced:
        raise SystemExit(f"{image} is not a georeferenced GeoTIFF; the geo spine needs one")

    fp = footprint_wgs84(meta, margin_deg=args.margin_deg)
    t0 = time.perf_counter()
    win = fetch_dem_window(fp, args.cache, COPERNICUS_GLO30)
    fetch_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    dem, info = dem_to_image_grid(win.path, meta)
    reproj_s = time.perf_counter() - t0

    dsm_path = write_dsm(out / "dem_on_image_grid.tif", dem, meta, kind="absolute", units="metres",
                         vertical_datum=COPERNICUS_GLO30.vertical_datum,
                         provenance={"dem_window": win.provenance, "reproject": info})
    finite = dem[np.isfinite(dem)]
    report = {
        "purpose": "geo spine run - DEM heights on the image grid; NOT an accuracy result",
        "input": {"path": str(image), "synthetic": args.image is None, "crs": meta.crs.to_string(),
                  "size": [meta.width, meta.height], "res_m": abs(meta.affine.a)},
        "footprint_wgs84_with_margin": fp,
        "dem_window": {k: win.provenance[k] for k in
                       ("tiles_used", "tiles_missing", "window_bounds", "window_crs", "shape", "nan_fraction")},
        "dem_window_file": str(win.path),
        "dem_window_file_kb": round(win.path.stat().st_size / 1024, 1),
        "reproject": {k: info[k] for k in ("resampling", "direction", "coverage_fraction", "dem_res", "image_res")},
        "dem_on_grid_stats_m": {"min": float(finite.min()), "max": float(finite.max()),
                                "mean": float(finite.mean()), "nan_count": int(np.isnan(dem).sum())},
        "vertical_datum": COPERNICUS_GLO30.vertical_datum,
        "seconds": {"fetch": round(fetch_s, 2), "reproject": round(reproj_s, 2)},
        "output": str(dsm_path),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
