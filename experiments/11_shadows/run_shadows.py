#!/usr/bin/env python3
"""Phase 11 T5b: shadow-based building heights at site A (urban), vs LiDAR, on held-out buildings.

Sun: the NAIP STAC datetime (2021-07-26T16:00:00Z) looks like a nominal round hour (UNKNOWN whether it is the real
acquisition time), so the shadow direction is MEASURED from the image (OSM footprints vs dark pixels) and the
time/elevation solved on the known date. The nominal 16:00Z sun is evaluated too.

Compared per building (same held-out buildings, footprint >= 50 m2, OSM, LiDAR 2013 DSM-DTM truth):
  c2        : current derived height (pipeline dsm.tif / ground.tif, building_heights)
  shadow    : shadow length x tan(elevation), measured sun
  shadow16z : same with the nominal 16:00Z sun
  c2xk      : c2 heights x k, k fitted on FIT-block buildings (median shadow/c2 ratio) -> calibration use
  hybrid    : shadow where it is measurable and not flagged, else c2

  PYTHONPATH=src python experiments/11_shadows/run_shadows.py <pipeline job dir for A_urban>
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "06_buildings"))
from run_buildings import lidar_heights  # noqa: E402

from depthwizard.calibration import checkerboard  # noqa: E402
from depthwizard.calibration.ndsm import building_heights  # noqa: E402
from depthwizard.calibration.shadows import (shadow_azimuth_from_footprints, shadow_heights, shadow_mask,  # noqa: E402
                                             solar_position, time_for_azimuth)
from depthwizard.evaluation import metrics  # noqa: E402
from depthwizard.geo import footprint_wgs84  # noqa: E402
from depthwizard.geo.osm import ATTRIBUTION, fetch_buildings, rasterize  # noqa: E402
from depthwizard.io import read_raster  # noqa: E402

KEEP = ("n", "bias", "mae", "rmse", "nmad", "le90", "pearson_r")
NOMINAL = datetime(2021, 7, 26, 16, 0, tzinfo=timezone.utc)


def slim(m):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k in KEEP}


def main() -> None:
    job = Path(sys.argv[1]).resolve()
    site = json.loads((ROOT / "data" / "exp8" / "A_urban" / "site.json").read_text())
    out = ROOT / "runs" / "exp11_shadows" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    data, meta = read_raster(site["naip"]["path"])
    rgb = np.moveaxis(data[:3], 0, -1)
    gsd = abs(meta.affine.a)
    w, s, e, n = footprint_wgs84(meta)
    lon, lat = (w + e) / 2, (s + n) / 2
    osm = fetch_buildings(footprint_wgs84(meta), ROOT / "data" / "osm")
    fp, ids = rasterize(osm["buildings"], meta.affine, meta.crs, (meta.height, meta.width), min_area_m2=50)
    ref = lidar_heights(site, osm["buildings"])
    blocks = checkerboard(meta.height, meta.width, round(154 / gsd))           # same split as A6c / pipeline
    held = {}
    for k in np.unique(fp[fp > 0]):
        rr, cc = np.nonzero(fp == k)
        held[int(k)] = not blocks[int(rr.mean()), int(cc.mean())]

    # sun: measured from the image, and nominal
    sh, thr = shadow_mask(rgb)
    meas = shadow_azimuth_from_footprints(sh, fp, gsd)
    t_meas, el_meas = time_for_azimuth(lon, lat, NOMINAL, meas["sun_azimuth_deg"])
    el16, az16 = solar_position(lon, lat, NOMINAL)
    sun = {"measured": {**meas, "solved_time_utc": t_meas.isoformat(), "sun_elevation_deg": round(el_meas, 2)},
           "nominal_16z": {"time_utc": NOMINAL.isoformat(), "sun_azimuth_deg": round(az16, 2),
                           "sun_elevation_deg": round(el16, 2)},
           "shadow_threshold_luminance": thr, "shadow_fraction": float(sh.mean())}
    print(json.dumps(sun, indent=1), flush=True)

    # heights
    with rasterio.open(job / "dsm.tif") as d, rasterio.open(job / "ground.tif") as g:
        z, gr = d.read(1), g.read(1)
        assert d.transform == meta.affine and d.shape == (meta.height, meta.width), "job grid differs from image"
    c2 = {k: v["height_m"] for k, v in building_heights(z, gr, fp).items()}
    shm = shadow_heights(sh, fp, gsd, meas["sun_azimuth_deg"], el_meas)
    sh16 = shadow_heights(sh, fp, gsd, az16, el16)

    def ok(r):
        return r.height_m is not None and not r.reasons

    # calibration use: one factor k from FIT-block buildings with a clean shadow and a positive c2 height
    fitk = [k for k in shm if not held.get(k) and ok(shm[k]) and c2.get(k, 0) > 1.0]
    ratios = np.array([shm[k].height_m / c2[k] for k in fitk])
    kfac = float(np.median(ratios)) if ratios.size >= 5 else None

    variants = {
        "c2": lambda k: c2.get(k),
        "shadow": lambda k: shm[k].height_m if k in shm else None,
        "shadow_clean_only": lambda k: shm[k].height_m if k in shm and ok(shm[k]) else None,
        "shadow16z": lambda k: sh16[k].height_m if k in sh16 else None,
        "c2xk": lambda k: c2[k] * kfac if (kfac and k in c2) else None,
        "hybrid": lambda k: shm[k].height_m if (k in shm and ok(shm[k])) else c2.get(k),
    }
    test_ids = [k for k in ref if held.get(k)]
    res, pairs = {}, {}
    common = [k for k in test_ids if all(f(k) is not None for name, f in variants.items() if name != "shadow_clean_only")]
    for name, f in variants.items():
        ks = [k for k in test_ids if f(k) is not None]
        est, tru = np.array([f(k) for k in ks]), np.array([ref[k] for k in ks])
        on_common = [k for k in common if f(k) is not None]
        ec, tc = np.array([f(k) for k in on_common]), np.array([ref[k] for k in on_common])
        res[name] = {"all_available": slim(metrics(est, tru)), "coverage": f"{len(ks)}/{len(test_ids)}",
                     "on_common_buildings": slim(metrics(ec, tc))}
        pairs[name] = (tru, est)
        print(f"{name:18s} coverage {len(ks):3d}/{len(test_ids)}  all: {res[name]['all_available']}", flush=True)
        print(f"{'':18s} common({len(on_common)}): {res[name]['on_common_buildings']}", flush=True)

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.3), constrained_layout=True)
    for a, name in zip(ax, ("c2", "shadow", "c2xk", "hybrid")):
        t, e_ = pairs[name]
        a.scatter(t, e_, s=9, alpha=0.6)
        lim = [0, max(30.0, float(np.nanmax(np.r_[t, e_])) * 1.05)]
        a.plot(lim, lim, "k--", lw=1); a.set_xlim(lim); a.set_ylim(lim)
        m = res[name]["all_available"]
        a.set_title(f"{name}: n {m['n']}, bias {m['bias']:+.2f}, RMSE {m['rmse']:.2f}, r {m['pearson_r']:.2f}", fontsize=9)
        a.set_xlabel("LiDAR building height (m)"); a.set_ylabel("estimated (m)")
    fig.suptitle(f"Site A, held-out buildings. Footprints {ATTRIBUTION}", fontsize=8)
    fig.savefig(out / "shadow_heights_scatter.png", dpi=90)
    plt.close(fig)
    # shadow mask overlay crop for the report
    r0, c0 = 700, 700
    crop = rgb[r0:r0 + 500, c0:c0 + 500].copy()
    crop[sh[r0:r0 + 500, c0:c0 + 500]] = (0.5 * crop[sh[r0:r0 + 500, c0:c0 + 500]] + [0, 90, 127]).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(crop)
    import cv2
    edges = cv2.Canny((fp[r0:r0 + 500, c0:c0 + 500] > 0).astype(np.uint8) * 255, 50, 150) > 0
    yy, xx = np.nonzero(edges)
    ax.scatter(xx, yy, s=0.2, c="#ff3b30")
    ax.set_title("shadow mask (cyan) + OSM footprints (red), 300 m crop", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out / "shadow_mask_crop.png", dpi=90)
    plt.close(fig)
    (out / "metrics.json").write_text(json.dumps({
        "attribution": ATTRIBUTION, "job": str(job), "sun": sun, "k_from_fit_buildings": kfac,
        "n_fit_buildings_for_k": len(fitk), "ratio_iqr": [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))] if ratios.size else None,
        "n_test_buildings": len(test_ids), "n_common": len(common), "results": res,
        "shadow_flags": {str(k): v.reasons for k, v in shm.items() if v.reasons}}, indent=2))
    print("run dir:", out)


if __name__ == "__main__":
    main()
