#!/usr/bin/env python3
"""Phase 11 T3: is there a tile seam in the products? Step statistic across tile-edge bands vs all other rows/cols.

For a product P (rdsm, or the c2 detail = dsm - dem) and each row r:
    step(r) = median over columns of (P[r + h] - P[r - h])            (h = half the blend band, 64 px)
    step is detrended with a running median over `detrend` rows (removes terrain slope)
A seam = |detrended step| at a tile-edge row well above the distribution at non-edge rows (control).
Same along columns. Reports, per product: max |step| at edge rows, and the p95 / max at control rows
(control rows are >= 2h away from every tile edge).

  PYTHONPATH=src python experiments/11_tile_seam/seam_check.py <job_dir> [<job_dir> ...] --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402

from depthwizard.tiling import plan_tiles  # noqa: E402


def read(p) -> np.ndarray:
    with rasterio.open(p) as s:
        return s.read(1).astype(np.float64)


def tile_edges(n: int, tile: int, overlap: int) -> list[int]:
    """Interior positions where some tile's footprint starts or ends (where blending weights change)."""
    starts = sorted({t.row0 for t in plan_tiles(n, 1, tile, overlap)} if n > tile else {0})
    e = set()
    for s in starts:
        e.update((s, s + min(tile, n)))
    return sorted(x for x in e if 0 < x < n)


def step_profile(P: np.ndarray, h: int, detrend: int, axis: int) -> np.ndarray:
    """Detrended median step across +-h, one value per row (axis=0) or column (axis=1)."""
    A = P if axis == 0 else P.T
    n = A.shape[0]
    s = np.full(n, np.nan)
    for r in range(h, n - h):
        s[r] = np.nanmedian(A[r + h] - A[r - h])
    ok = np.isfinite(s)
    trend = np.full(n, np.nan)
    trend[ok] = median_filter(s[ok], size=detrend, mode="nearest")
    return s - trend


def seam_stats(prof: np.ndarray, edges: list[int], h: int, band: int) -> dict:
    """Per edge: max |step| within the blend band after the edge ([e - band, e + band]); control = rows
    at least band + h away from every edge."""
    n = len(prof)
    idx = np.arange(n)
    near = np.zeros(n, bool)
    per_edge = {}
    for e in edges:
        w = (idx >= e - band) & (idx <= e + band)
        near |= (idx >= e - band - h) & (idx <= e + band + h)
        v = np.abs(prof[w])
        per_edge[int(e)] = float(np.nanmax(v)) if np.isfinite(v).any() else None
    ctrl = np.abs(prof[~near & np.isfinite(prof)])
    return {"edges": per_edge, "control_p95": float(np.percentile(ctrl, 95)), "control_max": float(ctrl.max()),
            "control_n": int(ctrl.size),
            "max_edge_over_control_p95": (max(v for v in per_edge.values() if v is not None) /
                                          float(np.percentile(ctrl, 95))) if per_edge else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--detrend", type=int, default=301)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for job in map(Path, a.jobs):
        md = json.loads((job / "metadata.json").read_text())
        tile, ov = md["tiling"]["tile"], md["tiling"]["overlap"]
        products = {"rdsm (relative)": read(job / "rdsm.tif")}
        if (job / "dsm.tif").exists():
            products["c2 detail = dsm - dem (m)"] = read(job / "dsm.tif") - read(job / "dem.tif")
        H, W = next(iter(products.values())).shape
        edges = {0: tile_edges(H, tile, ov), 1: tile_edges(W, tile, ov)}
        name = f"{job.parent.name}/{job.name}"
        report[name] = {"tile": tile, "overlap": ov, "row_edges": edges[0], "col_edges": edges[1]}
        fig, ax = plt.subplots(len(products), 2, figsize=(13, 3.2 * len(products)), squeeze=False,
                               constrained_layout=True)
        for i, (pname, P) in enumerate(products.items()):
            for axis in (0, 1):
                prof = step_profile(P, a.h, a.detrend, axis)
                st = seam_stats(prof, edges[axis], a.h, ov)
                report[name][f"{pname}|{'rows' if axis == 0 else 'cols'}"] = st
                x = ax[i][axis]
                x.plot(prof, lw=0.7, color="#333")
                for e in edges[axis]:
                    x.axvspan(e - ov, e + ov, color="#e6550d", alpha=0.12)
                    x.axvline(e, color="#e6550d", lw=0.8)
                x.axhline(st["control_p95"], color="#3182bd", ls="--", lw=0.8)
                x.axhline(-st["control_p95"], color="#3182bd", ls="--", lw=0.8)
                x.set_title(f"{pname}: {'row' if axis == 0 else 'column'} step profile "
                            f"(edge max / control p95 = {st['max_edge_over_control_p95']:.2f})", fontsize=9)
                x.set_xlabel("row" if axis == 0 else "column")
        fig.suptitle(f"{name}: detrended median step across ±{a.h} px; orange = tile edges ± overlap; "
                     "blue dashed = 95th pct at non-edge rows", fontsize=9)
        fig.savefig(out / f"seam_{job.parent.name}_{job.name}.png", dpi=90)
        plt.close(fig)
    (out / "seam_report.json").write_text(json.dumps(report, indent=2))
    for k, v in report.items():
        for pk, st in v.items():
            if isinstance(st, dict):
                print(f"{k:40s} {pk:36s} edge max {max(x for x in st['edges'].values() if x is not None):.4f}  "
                      f"control p95 {st['control_p95']:.4f} max {st['control_max']:.4f}  ratio {st['max_edge_over_control_p95']:.2f}")


if __name__ == "__main__":
    main()
