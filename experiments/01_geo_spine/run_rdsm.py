#!/usr/bin/env python3
"""Phase 2 acceptance run: real GeoTIFF -> tiled model -> rDSM GeoTIFF on the IDENTICAL grid
+ DEM on the same grid + overlay PNG + explicit grid checks.

The rDSM is relative (height_sign * model output). It is NOT calibrated and NOT metres.

  PYTHONPATH=src python experiments/01_geo_spine/run_rdsm.py --image data/samples/naip_...tif
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402

from depthwizard.depth import PREDICTORS  # noqa: E402
from depthwizard.export import export_dsm  # noqa: E402
from depthwizard.geo import COPERNICUS_GLO30, dem_to_image_grid, fetch_dem_window, footprint_wgs84  # noqa: E402
from depthwizard.io import read_raster  # noqa: E402
from depthwizard.tiling import run_tiled  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def stretch(a: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(a, (2, 98))
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="da3_mono_large", choices=list(PREDICTORS))
    ap.add_argument("--tile", type=int, default=1036)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    out = ROOT / "runs" / "geo_spine" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_rdsm"
    out.mkdir(parents=True, exist_ok=True)
    data, meta = read_raster(args.image)
    if meta.count < 3:
        raise SystemExit("need an RGB image")
    rgb = np.moveaxis(data[:3], 0, -1)
    if rgb.dtype != np.uint8:
        raise SystemExit(f"expected uint8 RGB, got {rgb.dtype}; add an explicit, recorded stretch first")

    kw = {"device": args.device, "process_res": args.tile}
    if args.model == "da2_small":
        kw["fp16"] = args.device == "cuda"
    pred = PREDICTORS[args.model](**kw)
    t0 = time.perf_counter()
    mosaic, tinfo = run_tiled(rgb, pred, tile=args.tile, overlap=args.overlap)
    infer_s = time.perf_counter() - t0
    rdsm = (pred.height_sign * mosaic).astype(np.float32)

    rdsm_path = export_dsm(out / "rdsm.tif", rdsm, meta, model=args.model, calibration="none", is_metric=False,
                           provenance={"image": args.image, "tiling": {k: tinfo[k] for k in ("tile", "overlap", "n_tiles")},
                                       "polarity": pred.polarity, "height_sign": pred.height_sign,
                                       "seed": args.seed})

    fp = footprint_wgs84(meta, margin_deg=0.001)
    win = fetch_dem_window(fp, ROOT / "data" / "dem_cache", COPERNICUS_GLO30)
    dem, rinfo = dem_to_image_grid(win.path, meta)
    dem_path = export_dsm(out / "dem_copernicus_on_grid.tif", dem, meta, model="none", calibration="dem_only (bilinear)",
                          is_metric=True, vertical_datum=COPERNICUS_GLO30.vertical_datum,
                          provenance={"dem_window": win.provenance, "reproject": rinfo})

    # acceptance: identical CRS / transform / shape to the input
    checks = {}
    with rasterio.open(args.image) as src:
        ref = (src.crs, src.transform, src.width, src.height)
    for name, p in (("rdsm", rdsm_path), ("dem", dem_path)):
        with rasterio.open(p) as s:
            checks[name] = {"crs_equal": s.crs == ref[0], "transform_equal": s.transform == ref[1],
                            "shape_equal": (s.width, s.height) == ref[2:], "tags": {k: s.tags()[k] for k in
                            ("DSM_KIND", "UNITS", "IS_METRIC", "MODEL", "CALIBRATION")}}

    # overlay: DEM contours drawn on the RGB and on the rDSM, all on the same pixel grid
    levels = np.arange(np.floor(np.nanmin(dem) / 5) * 5, np.nanmax(dem) + 5, 5)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), constrained_layout=True)
    axes[0].imshow(rgb); axes[0].set_title("RGB input + Copernicus 5 m contours")
    axes[1].imshow(stretch(rdsm), cmap="terrain"); axes[1].set_title(f"rDSM ({args.model}) PREVIEW ONLY, relative")
    axes[2].imshow(rgb); axes[2].imshow(stretch(rdsm), cmap="magma", alpha=0.45)
    axes[2].set_title("RGB + rDSM blend (pixel-registered by construction)")
    for ax in axes[:2]:
        cs = ax.contour(dem, levels=levels, colors="cyan", linewidths=0.8)
        ax.clabel(cs, fmt="%.0f", fontsize=7)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{Path(args.image).name} | {meta.crs.to_string()} | 0.6 m | same grid for all layers", fontsize=10)
    overlay = out / "overlay_alignment.png"
    fig.savefig(overlay, dpi=110)
    plt.close(fig)

    report = {"purpose": "Phase 2 acceptance: grid identity + alignment overlay. rDSM is relative, NOT metres.",
              "image": args.image, "model": args.model, "device": args.device,
              "tiling": {k: tinfo[k] for k in ("tile", "overlap", "n_tiles")},
              "tile_affine_a_range": [min(t["a"] for t in tinfo["tiles"]), max(t["a"] for t in tinfo["tiles"])],
              "inference_seconds": round(infer_s, 2),
              "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if args.device == "cuda" else None,
              "rdsm_stats": {"min": float(np.nanmin(rdsm)), "max": float(np.nanmax(rdsm)), "nan": int(np.isnan(rdsm).sum())},
              "dem_stats_m": {"min": float(np.nanmin(dem)), "max": float(np.nanmax(dem)), "nan": int(np.isnan(dem).sum())},
              "grid_checks": checks,
              "all_grids_identical": all(all(v[k] for k in ("crs_equal", "transform_equal", "shape_equal")) for v in checks.values()),
              "files": {"rdsm": str(rdsm_path), "dem": str(dem_path), "overlay": str(overlay)}}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
