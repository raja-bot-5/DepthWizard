#!/usr/bin/env python3
"""Phase 11 T3: tile seam before/after. Real pipeline (DA3, Copernicus, c2) per site with tile_align
"sequential" (before) and "joint" (after). Reports seam statistics (seam_check.py), fine-detail amplitude,
and accuracy of the c2 DSM on held-out test blocks vs LiDAR (2 m, EGM2008) and Copernicus (30 m).

  PYTHONPATH=src python experiments/11_tile_seam/run_seam_fix.py
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
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from seam_check import read, seam_stats, step_profile, tile_edges  # noqa: E402

from depthwizard.calibration import checkerboard  # noqa: E402
from depthwizard.calibration.datum import raster_file_to_egm2008  # noqa: E402
from depthwizard.evaluation import aggregate_to_grid, metrics  # noqa: E402
from depthwizard.export import export_dsm  # noqa: E402
from depthwizard.io import read_raster  # noqa: E402
from depthwizard.pipeline import PipelineConfig, run_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KEEP = ("n", "bias", "mae", "rmse", "nmad", "le90", "pearson_r")
MODES = ("sequential", "joint", "joint_plane")


def slim(m):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in KEEP}


def hp_amp(P: np.ndarray) -> float:
    from scipy.ndimage import uniform_filter
    h = (P - uniform_filter(P, 51))[25:-25, 25:-25]
    return float(np.median(np.abs(h - np.median(h))) * 1.4826)


def main() -> None:
    cfg8 = yaml.safe_load((ROOT / "configs" / "exp8.yaml").read_text())
    out = ROOT / "runs" / "exp11_seam" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    rep = {}
    for site_name in cfg8["sites"]:
        site = json.loads((ROOT / "data" / "exp8" / site_name / "site.json").read_text())
        sdir = out / site_name
        sdir.mkdir()
        img = site["naip"]["path"]
        _, meta = read_raster(img)
        test = ~checkerboard(meta.height, meta.width, max(8, round(154 / abs(meta.affine.a))))
        lid, _ = raster_file_to_egm2008(site["lidar_dsm"]["path"], sdir / "lidar_egm2008.tif")
        profs = {}
        for mode in MODES:
            j = sdir / mode
            md = run_pipeline(img, j, PipelineConfig(tile_align=mode, mesh_max_side=64,
                                                     dem_cache=str(ROOT / "data" / "dem_cache")))
            assert md["tiling"]["align"] == mode
            cal = json.loads((j / "calibration.json").read_text())
            if not (j / "dsm.tif").exists():                      # quality gate refused: record, no metric product
                rep[f"{site_name}|{mode}"] = {"gate": cal["gate"], "s": cal.get("s")}
                print(site_name, mode, "GATE FAILED", cal["gate"]["reasons"], flush=True)
                continue
            r, z, dem = read(j / "rdsm.tif"), read(j / "dsm.tif"), read(j / "dem.tif")
            H, W = r.shape
            er, ec = tile_edges(H, 1036, 128), tile_edges(W, 1036, 128)
            row = {"tiles": json.loads((j / "metadata.json").read_text())["tiling"], "gate": cal["gate"], "s": cal.get("s")}
            for pname, P in (("rdsm", r), ("c2_detail_m", z - dem)):
                pr, pc = step_profile(P, 64, 301, 0), step_profile(P, 64, 301, 1)
                profs[(mode, pname)] = (pr, pc, er, ec)
                row[pname] = {"rows": seam_stats(pr, er, 64, 128), "cols": seam_stats(pc, ec, 64, 128)}
            row["c2_detail_amplitude_m"] = hp_amp(z - dem)
            zt = np.where(test, z, np.nan).astype(np.float32)
            tp = export_dsm(sdir / "tmp.tif", zt, meta, model="da3", calibration="c2", is_metric=True,
                            vertical_datum="EGM2008")
            with rasterio.open(tp) as ps, rasterio.open(lid) as rs:
                agg, _ = aggregate_to_grid(ps, rs, 0.9)
            with rasterio.open(lid) as rs:
                ref = rs.read(1).astype(np.float32)
            row["vs_lidar_2m"] = slim(metrics(agg, ref))
            tp.unlink()
            for f in j.glob("*.glb"):
                f.unlink()
            rep[f"{site_name}|{mode}"] = row
            print(site_name, mode, "LiDAR RMSE", row["vs_lidar_2m"]["rmse"], "detail amp", round(row["c2_detail_amplitude_m"], 3),
                  "seam rdsm rows/cols", round(row["rdsm"]["rows"]["max_edge_over_control_p95"], 2),
                  round(row["rdsm"]["cols"]["max_edge_over_control_p95"], 2),
                  "c2 rows/cols", round(row["c2_detail_m"]["rows"]["max_edge_over_control_p95"], 2),
                  round(row["c2_detail_m"]["cols"]["max_edge_over_control_p95"], 2), flush=True)
        # before/after figure: rDSM row + column step profiles
        shown = [m for m in MODES if (sdir / m / "dsm.tif").exists()]
        fig, ax = plt.subplots(len(shown), 2, figsize=(13, 3 * len(shown)), constrained_layout=True, squeeze=False)
        for i, mode in enumerate(shown):
            pr, pc, er, ec = profs[(mode, "rdsm")]
            for k, (prof, edges, lab) in enumerate(((pr, er, "row"), (pc, ec, "column"))):
                x = ax[i][k]
                x.plot(prof, lw=0.7, color="#222")
                for e in edges:
                    x.axvspan(e - 128, e + 128, color="#e6550d", alpha=0.12)
                st = rep[f"{site_name}|{mode}"]["rdsm"]["rows" if k == 0 else "cols"]
                x.axhline(st["control_p95"], color="#3182bd", ls="--", lw=0.8)
                x.axhline(-st["control_p95"], color="#3182bd", ls="--", lw=0.8)
                x.set_title(f"{mode}: rDSM {lab} step profile, edge max / control p95 = "
                            f"{st['max_edge_over_control_p95']:.2f}", fontsize=9)
        fig.suptitle(f"{site_name}: tile seam before (sequential) / after (joint); orange = tile-edge blend bands", fontsize=10)
        fig.savefig(out / f"seam_before_after_{site_name}.png", dpi=90)
        plt.close(fig)
        # visual: hillshade-like gradient of the rDSM, before | after
        fig, ax = plt.subplots(1, len(MODES), figsize=(6 * len(MODES), 6), constrained_layout=True)
        for x, mode in zip(ax, MODES):
            r = read(sdir / mode / "rdsm.tif")
            gy = np.gradient(r, axis=0)
            lim = np.nanpercentile(np.abs(gy), 98)
            x.imshow(gy, cmap="gray", vmin=-lim, vmax=lim)
            for e in tile_edges(r.shape[0], 1036, 128):
                x.axhline(e, color="#e6550d", lw=0.5, alpha=0.6)
            x.set_title(f"{mode}: d(rDSM)/dy (tile edges orange)", fontsize=9); x.set_xticks([]); x.set_yticks([])
        fig.savefig(out / f"seam_gradient_{site_name}.png", dpi=80)
        plt.close(fig)
    (out / "seam_fix.json").write_text(json.dumps(rep, indent=2, default=str))
    print("run dir:", out)


if __name__ == "__main__":
    main()
