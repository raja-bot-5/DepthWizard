#!/usr/bin/env python3
"""Zero-shot baseline: pretrained DA-V2 Small and DA3-Mono-Large vs GAMUS LiDAR AGL heights.

Per tile: one model pass at --size (output resized back to the 1024^2 tile), then
- raw Pearson r of height_sign * prediction vs AGL (polarity check: should be > 0)
- affine-fitted metrics (fit AGL ~ a*pred + b on the SAME tile) -> OPTIMISTIC, labelled so
- the same fitted prediction stratified by GAMUS class

GAMUS is US aerial imagery, not Cartosat: these numbers rank models on shape quality,
they are not the competition metric.

  PYTHONPATH=src python experiments/03_gamus_zeroshot/run_zeroshot.py --device cpu --size 518
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from depthwizard.depth import PREDICTORS
from depthwizard.evaluation import affine_fitted_metrics, stratified_metrics

ROOT = Path(__file__).resolve().parents[2]
GAMUS = ROOT / "data" / "raw" / "gamus"
CLASS_NAMES = {1: "ground", 2: "low vegetation", 3: "buildings", 4: "water", 5: "road", 6: "tree"}


def load(rel: str) -> np.ndarray:
    with h5py.File(GAMUS / rel, "r") as h:
        return h[list(h.keys())[0]][()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="da2_small,da3_mono_large")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--exclude-value", type=float, default=None,
                    help="AGL value to treat as nodata (set only if the audit shows it is a sentinel)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    manifest = json.loads((GAMUS / "manifest.json").read_text())
    tiles = manifest["tiles"][: args.limit]
    out = ROOT / "runs" / "gamus_zeroshot" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.device}"
    out.mkdir(parents=True, exist_ok=True)
    report = {"purpose": "zero-shot shape baseline vs LiDAR AGL on GAMUS - affine-fitted, OPTIMISTIC; "
                         "US aerial imagery, not Cartosat",
              "args": vars(args), "gamus": {k: manifest[k] for k in ("repo", "revision", "split")},
              "n_tiles": len(tiles), "models": {}}

    for name in [m.strip() for m in args.models.split(",")]:
        kw = {"device": args.device, "process_res": args.size}
        if name == "da3_mono_large":
            kw["max_sky_fraction"] = 1.0  # record sky, do not abort the baseline
        if name == "da2_small":
            kw["fp16"] = args.device == "cuda"
        pred_fn = PREDICTORS[name](**kw)
        per_tile, times = [], []
        print(f"\n=== {name} ({pred_fn.polarity}, height_sign {pred_fn.height_sign:+d})", flush=True)
        for i, t in enumerate(tiles, 1):
            img, agl = load(t["image"]), load(t["height"]).astype(np.float64)
            cls = np.rint(load(t["classes"])).astype(np.int16)
            if args.exclude_value is not None:
                agl[agl == args.exclude_value] = np.nan
            t0 = time.perf_counter()
            pred = pred_fn(img).astype(np.float64)
            times.append(time.perf_counter() - t0)
            h = pred_fn.height_sign * pred
            valid = np.isfinite(h) & np.isfinite(agl)
            raw_r = float(np.corrcoef(h[valid], agl[valid])[0, 1])
            fit = affine_fitted_metrics(h, agl)
            fitted = fit["fit_a"] * h + fit["fit_b"]
            rec = {"id": t["id"], "city": t["city"], "raw_r_signed": raw_r,
                   "fitted": {k: fit[k] for k in ("n", "rmse", "mae", "nmad", "pearson_r", "fit_a", "fit_b")},
                   "by_class": {k: {kk: v[kk] for kk in ("n", "rmse", "mae", "bias") if kk in v}
                                for k, v in stratified_metrics(fitted, agl, cls, CLASS_NAMES).items()},
                   "seconds": round(times[-1], 2)}
            if hasattr(pred_fn, "last_sky_fraction"):
                rec["sky_fraction"] = pred_fn.last_sky_fraction
            per_tile.append(rec)
            print(f"[{i}/{len(tiles)}] {t['id']:<12} r={raw_r:+.3f} fitted RMSE={fit['rmse']:.2f} "
                  f"({times[-1]:.1f}s)", flush=True)
            (out / "report.json").write_text(json.dumps(report | {"models": report["models"] | {name: per_tile}}, indent=2))

        def agg(key):
            return float(np.median([r["fitted"][key] for r in per_tile]))
        summary = {"median_raw_r_signed": float(np.median([r["raw_r_signed"] for r in per_tile])),
                   "tiles_with_negative_r": sum(r["raw_r_signed"] < 0 for r in per_tile),
                   "median_fitted_rmse": agg("rmse"), "median_fitted_mae": agg("mae"),
                   "median_fitted_nmad": agg("nmad"),
                   "by_class_median_rmse": {c: float(np.median([r["by_class"][c]["rmse"] for r in per_tile
                                                                 if r["by_class"][c].get("n", 0) > 500]))
                                            for c in CLASS_NAMES.values()
                                            if any(r["by_class"][c].get("n", 0) > 500 for r in per_tile)},
                   "mean_seconds_per_tile": float(np.mean(times))}
        if name == "da3_mono_large":
            summary["max_sky_fraction"] = max(r.get("sky_fraction") or 0 for r in per_tile)
        report["models"][name] = {"summary": summary, "tiles": per_tile}
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        del pred_fn

    print("report:", out / "report.json")


if __name__ == "__main__":
    main()
