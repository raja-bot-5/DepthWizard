#!/usr/bin/env python3
"""Build docs/results.md from runs/exp8/<run>/metrics.json. Every number in the tables is read from
metrics.json; nothing is typed by hand.

  python experiments/08_stratified/make_results_md.py runs/exp8/<run> [runs/exp8/<pre-datum-baseline-run>]
The optional baseline run adds a "datum conversion: old vs new bias" table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITES = {"A_urban": "A urban (CU Boulder campus)", "B_hilly_forest": "B hilly forest (Flatirons foothills)",
         "C_sparse": "C sparse (east Boulder open space)"}
METHODS = {"a_dem_only": "(a) DEM only", "b_robust_affine": "(b) model scaled to DEM",
           "c_dem_plus_residual": "(c) DEM + detail, per-cell", "c2_dem_plus_smooth_residual": "(c2) DEM + detail, smooth"}
MODELS = {"none": "none", "da2_small": "DA-V2 Small", "da3_mono_large": "DA3-Mono-Large"}


def f(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    m = json.loads((run / "metrics.json").read_text())
    R = m["results"]
    base = json.loads((Path(sys.argv[2]).resolve() / "metrics.json").read_text()) if len(sys.argv) > 2 else None

    def get(site, gsd, meth, model):
        return next(r for r in R if r["site"] == site and r["gsd_m"] == gsd and r["method"] == meth and r["model"] == model)

    L = []
    L += ["# DepthWizard: stratified validation results (Phase 8)", "",
          f"Source: `{run.relative_to(ROOT)}/metrics.json`, config `configs/exp8.yaml`, seed {m['seed']}. "
          "All tables below are generated from that file by `experiments/08_stratified/make_results_md.py`.", ""]

    a_vs = {s: get(s, 0.6, "a_dem_only", "none")["lidar_2m"]["overall"]["rmse"] for s in SITES}
    c2_vs = {s: get(s, 0.6, "c2_dem_plus_smooth_residual", "da3_mono_large")["lidar_2m"]["overall"]["rmse"] for s in SITES}
    L += ["## Summary", ""]
    L += ["Against an independent LiDAR DSM, the best method is **(c2): the Copernicus DEM plus DA3-Mono-Large detail, "
          "with the detail made zero-mean by a smooth 30 m low-pass**. It has the lowest RMSE at all three sites, measured on held-out test blocks:", ""]
    for s in SITES:
        L.append(f"- {SITES[s]}: {f(c2_vs[s])} m vs DEM-only {f(a_vs[s])} m "
                 f"({100 * (c2_vs[s] - a_vs[s]) / a_vs[s]:+.1f} %)")
    L += ["",
          "- The gain is largest where there are buildings and trees on flat or moderate ground. "
          "It shrinks on steep forest and is negligible on open grassland.",
          "- The per-cell version (c) leaves 30 m steps at DEM cell edges. It **loses to DEM-only on steep forest** "
          "(see table 1). The smooth version (c2) fixes both the visual grid and the accuracy loss. "
          "The pipeline default is now (c2).",
          "- (b), scaling the model output to the DEM alone, is unusable. Monocular depth from a near-nadir image does "
          "not carry absolute terrain.",
          "- Accuracy has a floor set by the 30 m DEM. Every calibrated product inherits the DEM's bias (table 1: the bias "
          "is identical across (a), (c) and (c2)). The model only adds fine structure.", ""]

    L += ["## Setup", "",
          "| | |", "|---|---|",
          "| Imagery | USDA NAIP 2021, 0.6 m RGB, public domain (a stand-in for Cartosat-2S; see Limitations) |",
          "| Independent reference | USGS 3DEP LiDAR 2013 (survey `USGS_LPC_CO_SoPlatteRiver_Lot5_2013_LAS_2015`): DSM and DTM, 2 m, NAVD88, **converted to EGM2008** (GEOID18 + EGM2008 grids via PROJ) before scoring |",
          "| Calibration DEM | Copernicus GLO-30 (EGM2008; verified from the tile XML in Phase 10 A1) |",
          "| Land-cover strata | ESA WorldCover 2021, 10 m, CC-BY-4.0. Urban = built-up; forest = tree cover; sparse = grass/crop/shrub/bare; water = water/wetland |",
          "| Terrain strata | Slope of the LiDAR **DTM**, smoothed to ~10 m. Flat < 5°, moderate 5–15°, hilly ≥ 15°. Building walls are not counted as terrain. |",
          "| Split | Checkerboard of 154 m blocks. Anchors (used to fit (b)'s scale, which (c) and (c2) reuse) and test blocks never overlap. Only test blocks are scored. |",
          "| Scoring | Prediction area-averaged onto the reference grid. A cell counts only if it lies fully inside the image and ≥ 90 % of it is test pixels. |",
          "| Bias note | Outputs and LiDAR are both EGM2008, so bias no longer contains a datum offset. It still includes the 2013 → 2021 time gap and Copernicus's own bias. `std` = RMSE with bias removed. |",
          ""]
    if base:
        def bget(site, meth, model):
            return next(r for r in base["results"] if r["site"] == site and r["gsd_m"] == 0.6 and r["method"] == meth
                        and r["model"] == model)["lidar_2m"]["overall"]
        L += ["## 0. Datum conversion: bias before and after (vs LiDAR DSM, 2 m, GSD 0.6 m)", "",
              f"Before: `{Path(sys.argv[2]).resolve().relative_to(ROOT)}` (LiDAR left in NAVD88). "
              "After: this run (LiDAR converted to EGM2008). The geoid offset is the per-site range applied by PROJ.", "",
              "| Site | Method | Offset NAVD88 → EGM2008 (m) | Bias before (m) | Bias after (m) | RMSE before → after (m) | std before → after (m) |",
              "|---|---|---|---|---|---|---|"]
        for s in SITES:
            d = m["lidar_datum_conversion"][s]["lidar_dsm"]
            for meth, model in (("a_dem_only", "none"), ("c2_dem_plus_smooth_residual", "da3_mono_large")):
                o, n = bget(s, meth, model), get(s, 0.6, meth, model)["lidar_2m"]["overall"]
                L.append(f"| {SITES[s].split(' (')[0]} | {METHODS[meth]}{' DA3' if model != 'none' else ''} | "
                         f"{d['offset_min_m']:+.2f} … {d['offset_max_m']:+.2f} | {f(o['bias'])} | {f(n['bias'])} | "
                         f"{f(o['rmse'])} → {f(n['rmse'])} | {f(o['std_error'])} → {f(n['std_error'])} |")
        L += ["", "- The bias moves by exactly the geoid offset and `std` is unchanged, so the model outputs are identical "
              "between runs. **The remaining biases (about −1.2 / +1.9 / −1.1 m) are not a datum effect.** They are "
              "Copernicus's own local bias plus the 2013 → 2021 time gap. The RMSE changes are only that bias shift.", ""]

    L += ["## 1. Overall accuracy vs LiDAR DSM (2 m, GSD 0.6 m)", "",
          "| Site | Method | Model | n | Bias (m) | MAE (m) | RMSE (m) | std (m) | r | R² |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SITES:
        for meth in METHODS:
            for model in (["none"] if meth == "a_dem_only" else ["da2_small", "da3_mono_large"]):
                o = get(s, 0.6, meth, model)["lidar_2m"]["overall"]
                L.append(f"| {SITES[s].split(' (')[0]} | {METHODS[meth]} | {MODELS[model]} | {o['n']:,} | {f(o['bias'])} | "
                         f"{f(o['mae'])} | {f(o['rmse'])} | {f(o['std_error'])} | {f(o['pearson_r'], 3)} | {f(o['r2'], 3)} |")
    L.append("")

    L += ["## 2. Per-stratum accuracy vs LiDAR DSM (2 m, GSD 0.6 m)", "",
          "Strata are pixel-level classes within each site. Strata with fewer than 500 cells are left out.", ""]
    for key, label in (("land", "Land cover"), ("terrain", "Terrain slope (from LiDAR DTM)")):
        L += [f"### {label}", "",
              "| Site | Stratum | n | DEM only: bias / MAE / RMSE / r | (c2) DA3: bias / MAE / RMSE / r | (c2) DA-V2: RMSE | Δ RMSE (c2 DA3 vs DEM) |",
              "|---|---|---|---|---|---|---|"]
        for s in SITES:
            A = get(s, 0.6, "a_dem_only", "none")["lidar_2m"]["strata"]
            C = get(s, 0.6, "c2_dem_plus_smooth_residual", "da3_mono_large")["lidar_2m"]["strata"]
            D = get(s, 0.6, "c2_dem_plus_smooth_residual", "da2_small")["lidar_2m"]["strata"]
            for k in A:
                if not k.startswith(key) or A[k].get("n", 0) < 500:
                    continue
                a, c = A[k], C[k]
                L.append(f"| {SITES[s].split(' (')[0]} | {k.split(':')[1]} | {a['n']:,} | "
                         f"{f(a['bias'])} / {f(a['mae'])} / {f(a['rmse'])} / {f(a['pearson_r'], 3)} | "
                         f"{f(c['bias'])} / {f(c['mae'])} / {f(c['rmse'])} / {f(c['pearson_r'], 3)} | {f(D[k]['rmse'])} | "
                         f"{100 * (c['rmse'] - a['rmse']) / a['rmse']:+.1f} % |")
        L.append("")
    # pooled across sites (cell-weighted RMSE) for the four headline strata
    L += ["### Pooled over the three sites (cell-weighted RMSE, m)", "",
          "| Stratum | n | DEM only | (c2) DA3 | (c2) DA-V2 |", "|---|---|---|---|---|"]
    for k in ("land:urban", "land:sparse", "land:forest", "land:water", "terrain:flat", "terrain:moderate", "terrain:hilly"):
        tot, acc = 0, {"a": 0.0, "c": 0.0, "d": 0.0}
        for s in SITES:
            for tag, meth, model in (("a", "a_dem_only", "none"), ("c", "c2_dem_plus_smooth_residual", "da3_mono_large"),
                                     ("d", "c2_dem_plus_smooth_residual", "da2_small")):
                v = get(s, 0.6, meth, model)["lidar_2m"]["strata"].get(k)
                if v and v.get("n"):
                    acc[tag] += v["n"] * v["rmse"] ** 2
                    if tag == "a":
                        tot += v["n"]
        if tot:
            L.append(f"| {k.split(':')[1]} | {tot:,} | {f((acc['a'] / tot) ** 0.5)} | {f((acc['c'] / tot) ** 0.5)} | {f((acc['d'] / tot) ** 0.5)} |")
    L.append("")

    L += ["## 3. GSD sweep (0.6 → 1.2 → 2.4 m)", "",
          "Coarser images are exact block means of the 0.6 m image, with the transform rescaled. Every GSD is scored on the "
          "same 6 m grid (LiDAR DSM block-averaged 3×3), so rows can be compared directly. RMSE in metres:", "",
          "| Site | GSD (m) | DEM only | (c) DA3 | (c2) DA3 | (c2) DA-V2 |", "|---|---|---|---|---|---|"]
    for s in SITES:
        for g in (0.6, 1.2, 2.4):
            row = [get(s, g, "a_dem_only", "none"), get(s, g, "c_dem_plus_residual", "da3_mono_large"),
                   get(s, g, "c2_dem_plus_smooth_residual", "da3_mono_large"), get(s, g, "c2_dem_plus_smooth_residual", "da2_small")]
            L.append(f"| {SITES[s].split(' (')[0]} | {g} | " + " | ".join(f(x["lidar_6m"]["overall"]["rmse"]) for x in row) + " |")
    L += ["", "- DA3 detail still helps urban at 2.4 m, although the gain roughly halves. "
          "DA-V2 Small at 1.2–2.4 m is **worse than DEM-only** in urban.",
          "- In forest the gain fades by 2.4 m. On sparse land all methods stay within a few centimetres of each other.", ""]

    L += ["## 4. Consistency with the calibration DEM (Copernicus 30 m, GSD 0.6 m)", "",
          "This is **not** an independent check: (a) *is* Copernicus, and (c)/(c2) are built to match it at 30 m. "
          "It shows how far each product departs from the DEM the organisers may score against.", "",
          "| Site | Method | Model | RMSE vs Copernicus (m) | bias (m) |", "|---|---|---|---|---|"]
    for s in SITES:
        for meth, model in (("a_dem_only", "none"), ("c_dem_plus_residual", "da3_mono_large"),
                            ("c2_dem_plus_smooth_residual", "da3_mono_large"), ("b_robust_affine", "da3_mono_large")):
            o = get(s, 0.6, meth, model)["copernicus_30m"]["overall"]
            L.append(f"| {SITES[s].split(' (')[0]} | {METHODS[meth]} | {MODELS[model]} | {f(o['rmse'])} | {f(o['bias'])} |")
    L.append("")

    L += ["## 5. Derived products: ground estimate and height above ground", "",
          "The app offers a crude **derived** ground surface (a 30 m morphological opening of the DSM) and "
          "*height above ground* = DSM − ground. Checked against the LiDAR DTM and the LiDAR nDSM (DSM − DTM, on "
          "cells where LiDAR says objects are taller than 2 m):", "",
          "| Site | Ground vs LiDAR DTM: bias / RMSE (m) | Height above ground vs LiDAR nDSM: bias / RMSE / r | Object cells |",
          "|---|---|---|---|"]
    for gchk in m["ground_checks"]:
        g, h = gchk["ground_estimate_vs_lidar_dtm"], gchk["derived_height_vs_lidar_ndsm_on_objects_gt2m"]
        L.append(f"| {SITES[gchk['site']].split(' (')[0]} | {f(g['bias'])} / {f(g['rmse'])} | {f(h['bias'])} / {f(h['rmse'])} / "
                 f"{f(h['pearson_r'], 3)} | {100 * gchk['lidar_object_fraction']:.0f} % |")
    L += ["", "- The ground estimate is usable in urban and sparse areas. In continuous forest it stays on the canopy "
          "(+5 m bias), as expected, because a 30 m opening can't see through a forest wider than 30 m.",
          "- **Height above ground is not reliable.** It underestimates objects by 3–5 m and correlates weakly "
          "(r 0.23–0.49). The viewer already labels it *derived, not measured*. It should not be presented as a "
          "measurement until a better ground model and building-height calibration exist.", ""]

    L += ["## 6. Failure cases", "",
          "These are the worst 128 m windows for (c2) DA3 at each site, ranked by RMSE on test blocks. Each panel shows "
          "NAIP 0.6 m, the DSM at 2 m, the LiDAR DSM, and the error. Blank areas are anchor blocks, or lie outside the image.", ""]
    notes = {
        "A_urban_1": "Folsom Field stadium: the LiDAR shows the ~25 m press box and stands. The DSM stays near the DEM "
                     "(errors down to −20 m). Tall structures are strongly underestimated, because the detail scale is "
                     "fitted at 30 m. The stadium may also have changed between 2013 and 2021.",
        "A_urban_2": "Not a model error: the 2021 image shows a flat practice field, but the 2013 LiDAR shows a "
                     "~14 m rounded structure there (it looks like an air-supported practice dome). The DSM is correctly "
                     "flat. This is land change between the LiDAR and the image.",
        "B_hilly_forest_1": "Narrow forested gully: the DSM is +10–15 m too high. The 30 m DEM fills narrow valleys "
                            "and the model detail doesn't carve them back.",
        "B_hilly_forest_2": "Another narrow gully: the DSM is ~+10 m along the valley floor. Same cause as window 1 "
                            "(the 30 m DEM fills sharp valleys).",
        "C_sparse_1": "Open water: a uniform −4 m. LiDAR over water is sparse and noisy, and Copernicus flattens lakes. "
                      "Neither reference is trustworthy on water, and there is no water mask (RGB-only input).",
        "C_sparse_2": "Tree clumps in grassland: canopies (~10 m in LiDAR) are underestimated by up to ~10 m. The "
                      "model detail is too weak for isolated tall objects.",
    }
    for fi in m["failure_figures"]:
        if "rmse_c2_da3" not in fi:
            continue
        key = fi["file"].replace("failure_", "").replace(".png", "")
        L += [f"**{SITES[fi['site']].split(' (')[0]}, window {key[-1]}:** RMSE {f(fi['rmse_c2_da3'])} m "
              f"(DEM-only in the same window: {f(fi['rmse_dem_only'])} m). {notes.get(key, '')}", "",
              f"![{key}](figures/{fi['file']})", ""]
    L += ["Error maps for the whole sites (DEM-only vs (c2) DA3):", ""]
    for s in SITES:
        L += [f"![errormap {s}](figures/errormap_{s}.png)", ""]

    L += ["## Limitations", "",
          "- **Not Cartosat.** The imagery is US aerial NAIP: a different sensor, near-nadir, 2021. The competition "
          "data is Cartosat-2S satellite imagery over India. These numbers rank methods; they do not predict the "
          "competition score. No open Cartosat + LiDAR pair was found.",
          "- **Three sites, one region, one LiDAR survey, one seed.** No confidence intervals. Differences under about "
          "0.1 m (for example on sparse land) are not meaningful.",
          "- **Datums are converted** (LiDAR NAVD88 → EGM2008, +0.17 to +0.26 m here). The remaining site biases come "
          "from the calibration DEM itself, which every calibrated product inherits.",
          "- **Time gap:** LiDAR October 2013 vs imagery July 2021. Changes count as error (urban failure window 2 is "
          "one: a structure present in 2013 is gone in 2021).",
          "- **Water is not masked** (RGB only), and both references are unreliable on water.",
          "- **Buildings and tall structures are underestimated.** The model detail is scaled with one factor fitted "
          "to the smooth DEM. A building-height calibration (for example from GAMUS fine-tuning, or sparse control "
          "points such as ICESat-2) is the obvious next step.",
          "- **Mixed strata:** WorldCover is 10 m, scored on a 2 m grid, so class edges are mixed.",
          "- **Fine-tuned models are not evaluated here.** The Phase 5 sanity fine-tune only proved the pipeline, and the DA3 "
          "Kaggle fine-tune has not been run.",
          "- **ICESat-2** sparse validation over India was not done (it needs a NASA Earthdata account, which is a GATE).",
          ""]
    L += ["## Timings (RTX 3050 6 GB, tiled inference only)", "",
          "| Site | GSD (m) | Model | Tiles | Seconds |", "|---|---|---|---|---|"]
    for t in m["timings"]:
        L.append(f"| {SITES[t['site']].split(' (')[0]} | {t['gsd_m']} | {MODELS[t['model']]} | {t['tiles']} | {t['seconds']} |")
    L.append("")
    (ROOT / "docs" / "results.md").write_text("\n".join(L))
    print("wrote docs/results.md,", len(L), "lines")


if __name__ == "__main__":
    main()
