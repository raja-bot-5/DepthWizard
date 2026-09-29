# DepthWizard: technical report

**Task:** SIH 2026 PS 26175 (ISRO). Estimate a digital surface model (DSM) from one optical RGB image and present it as an interactive 3D flythrough. Scoring is 50 % DSM accuracy (RMSE, MAE, correlation across urban, sparse, hilly and forest terrain) and 50 % visualisation and UX. Evaluation imagery is Cartosat-2S GeoTIFF at about 0.6 m. GeoTIFF output is compared with an absolute DSM "such as SRTM or Copernicus". No reference data is provided.

Every number below comes from a run recorded in this repository. The source is given in brackets.

## 1. The core problem, and the approach it leads to

A single image has no absolute height information. Monocular depth networks return depth that is only defined up to an unknown scale and shift for each inference. For a near-nadir view of the ground, that depth is dominated by shading and texture cues rather than terrain.

We measured this directly. The best global fit *Z = a·R + b* of the model output *R* to Copernicus gives RMSE of **11–53 m** against LiDAR, and the correlation with LiDAR on open land is only r ≈ 0.09 [runs/exp8/20260928T202021Z, table 1 of docs/results.md]. So the model **cannot supply terrain**.

It can supply **fine structure**. The approach therefore splits the spectrum:

- **Coarse heights (≥ 30 m)** come from a free global DEM, Copernicus GLO-30, which is itself a DSM. This matches what the organisers score against.
- **Fine structure (< 30 m)** comes from the monocular model, with its own coarse component removed.

Z = D + a · (R − L₃₀(R))

- *D* is the DEM, bilinear-resampled onto the image grid.
- *R* is the model output oriented so larger = taller. DA-V2 predicts disparity, so R = +output; DA3-Mono predicts depth, so R = −output. Both were verified from the DA3 source and by a positive fitted *a*.
- *L₃₀* is a NaN-aware 30 m box low-pass.
- *a* is fitted by Huber regression of *L₃₀(R)* against *D* on low-slope anchor pixels.

By construction *Z* keeps the DEM's heights at 30 m and adds the model's detail. This is method **(c2)**, the default. The earlier variant **(c)** subtracts the mean of *R* inside each 30 m DEM cell instead. It leaves steps at every cell edge, which show as a grid in the slope view and cost accuracy on steep terrain (§4).

## 2. Models

| Model | Params | License | Role | GPU cost (RTX 3050, 518 px / 1036 px tile) |
|---|---|---|---|---|
| Depth Anything 3 Mono Large | 0.33 B | Apache-2.0 | default | 184 ms, 1.67 GB / 760 ms, 2.63 GB |
| Depth Anything V2 Small | 25 M | Apache-2.0 | fast fallback | 44 ms, 0.15 GB / 127 ms, 0.42 GB |

[runs/smoke/20260928T190346Z_cuda]. Checkpoint sha256 values match Hugging Face.

Findings from reading the DA3 source (commit 3d835ec):
- Its API always runs under autocast (bf16 on this GPU).
- Its mono head is `exp`-activated depth.
- A sky head **overwrites** "sky" pixels with the 99th-percentile depth. On overhead imagery the measured sky fraction was 0.0, but the app refuses tiles with more than 1 % sky rather than pass corrupted heights on.

Non-commercial checkpoints (DA-V2 Base/Large, multi-view DA3) are excluded.

## 3. Pipeline (`src/depthwizard/`)

1. **Read** (`io`). GeoTIFF CRS and transform are kept exactly. PNG/JPG are never treated as georeferenced. 16-bit or panchromatic input is stretched to 8-bit RGB *for the model only*.
2. **Tiled inference** (`tiling`). Tiles are 1036 px with 128 px overlap. Each new tile's output is affine-aligned to the mosaic on the overlap, because tiles disagree by scale and shift, and then blended with a cosine ramp. A crop is never resampled, and each tile keeps its own geotransform.
3. **DEM** (`geo`).
   - The footprint is converted to WGS84 with densified edges.
   - Copernicus windows are read from the public AWS Cloud-Optimized GeoTIFFs by HTTP range request: a 1.2 km scene costs about 10 KB, not a 42 MB tile.
   - The window is snapped to the source grid, whose pixel centres sit on whole arc-seconds. rasterio's `target_aligned_pixels` would have been off by half a pixel, about 15 m.
   - Neighbour tiles reached only by padding are opened too (a bug found and fixed in Phase 4).
   - The DEM is bilinear-resampled onto the **image** grid. The image grid is never changed.
   - Requests are cached and looked up **before** any network access, so a cached area works offline. This was verified with a dead proxy.
4. **Calibration** (`calibration`): method (c2) above. (a) DEM-only, (b) global affine and (c) per-cell are also available.
5. **Export** (`export`). A float32 GeoTIFF on the input grid, with tags `DSM_KIND`, `UNITS`, `VERTICAL_DATUM`, `MODEL`, `CALIBRATION` and a JSON sidecar. Writing a metric product without a datum, a relative product labelled in metres, or an array whose shape doesn't match the grid is refused.
6. **Derived products** (`reconstruction`): a textured GLB mesh (at most 512² vertices plus a 128² LOD) and a crude ground estimate (30 m morphological opening).

The **backend** (`backend/app.py`, FastAPI) queues jobs to one worker, so there is one GPU job at a time. It serves whitelisted products and answers point queries from the DSM GeoTIFF, not from the mesh.

The **frontend** (`frontend/`, Three.js 0.186 vendored, no CDN) provides:
- a metric/relative stamp
- input and calibration panels
- orbit (clamped above the horizon) and fly navigation
- point inspection and two-point measurement using true heights (a display-only exaggeration slider can't change them)
- slope and reference-difference overlays
- exports

## 4. Validation against independent LiDAR

**Protocol** [configs/exp8.yaml, runs/exp8/20260929T145210Z; re-run with datum conversion, Phase 11 T1]:
- Three 1.2 km sites near Boulder, CO: **urban**, **hilly forest** (590 m relief) and **sparse** (grassland, with a lake).
- NAIP 2021 at 0.6 m as the image; 3DEP LiDAR 2013 DSM and DTM at 2 m as the reference, converted NAVD88 → EGM2008 (GEOID18, +0.17 to +0.26 m) before scoring.
- ESA WorldCover land-cover strata, and terrain strata from the slope of the LiDAR *DTM*, so that building walls don't count as hills.
- A 154 m checkerboard splits fitting anchors from test blocks. Only test blocks are scored.
- Predictions are area-averaged onto the LiDAR grid.

| RMSE vs LiDAR, 2 m (m) | DEM only | (b) affine, DA3 | (c) per-cell, DA3 | **(c2) smooth, DA3** | (c2) smooth, DA-V2 |
|---|---|---|---|---|---|
| Urban | 3.95 | 11.47 | 3.41 | **3.24** | 3.56 |
| Hilly forest | 4.69 | 48.63 | 4.97 | **4.38** | 4.52 |
| Sparse | 2.15 | 5.97 | 2.09 | **2.10** | 2.10 |

- **(c2) DA3 is best, or tied best, at every site.** It improves on the DEM by 18 %, 7 % and 2 %. It improves every land-cover and terrain stratum except open water, where both references are unreliable [docs/results.md §2].
- **Why the smooth variant matters:** in steep forest, per-cell (c) is *worse* than the DEM (4.97 vs 4.69 m), while smooth (c2) is better (4.38 m).
- **Resolution sweep** (common 6 m scoring grid, urban): DEM 3.67–3.69 m. (c2) DA3 gives 3.01, 3.12 and 3.24 m at 0.6, 1.2 and 2.4 m GSD. DA-V2 at 2.4 m (4.03 m) is worse than the DEM alone.
- **Bias vs LiDAR** (−1.23 / +1.92 / −1.15 m, both in EGM2008) is the same for all DEM-based methods. It is inherited from the DEM. Converting the datum moved each bias by exactly the geoid offset and left the bias-free error (std) unchanged, so these biases are Copernicus's own, plus the 2013 → 2021 gap [docs/results.md §0].
- **Earlier result, same conclusion:** Experiment 0 (two sites, per-cell (c) only) found (c) DA3 3.33 m vs DEM 3.86 m in urban, and DEM-only best in forest [runs/exp0/20260928T192346Z]. Phase 8's smooth variant resolved that forest loss.

**Failure modes** (docs/results.md §6):
- **Tall structures** are underestimated by up to 20 m (a stadium press box). The single detail scale *a* is fitted at 30 m.
- **Narrow valleys** filled by the 30 m DEM come out +10–15 m too high.
- **Water** is not masked, and both references are unreliable there.
- **Isolated trees** are underestimated.
- One "failure" is actually a structure that existed in the 2013 LiDAR and is gone in the 2021 image.

**Derived heights.** The ground estimate matches the LiDAR DTM in urban and sparse areas (RMSE 2.23 / 1.23 m). It stays on the canopy in continuous forest (+5.26 m bias). *Height above ground* is **not reliable**: it underestimates by 3–5 m, with r 0.23–0.49. The UI labels it *derived*.

## 5. Fine-tuning

**GAMUS** [docs/gamus_audit.md] is used as training data: CC-BY-4.0, 80 GB, 1024² aerial tiles with LiDAR above-ground height, three US cities. The files carry no unit or pixel-size metadata, and each city uses a different floor convention. Only subsets are used on this laptop.

`training/finetune.py`:
- **Loss:** scale-and-shift-invariant L1 that keeps the sign (median/MAD normalisation, MiDaS-style).
- **Holdout by city:** train on DC + PHL, validate on NYC.
- **Tooling:** AMP (bf16), gradient checkpointing (native for DA-V2; a block wrapper for DA3, untested), seeds, and sha256 for every checkpoint.

A local DA-V2 Small sanity run (300 steps, 40 tiles, 99 s) cut the training loss from 0.82 to 0.58. On the unseen city, the affine-fitted RMSE fell from **5.84 to 4.53 m** and the correlation rose from **0.13 to 0.59** [runs/finetune/20260928T194117Z_da2_small]. These are optimistic per-crop fits, and this was a pipeline check, not a model to deploy. The DA3 fine-tune is packaged for Kaggle (`notebooks/finetune_da3_kaggle.ipynb`) and **has not been run yet**. Fine-tuned models are not part of the §4 results.

## 6. Performance and deployment

- **End to end** on a 2048² GeoTIFF: 16–35 s over five runs on the RTX 3050 (the first includes model loading), including DEM fetch, calibration and mesh. A 1024² PNG takes about 2 s [runs/jobs/*/job.json]. A cold first build in a fresh clone took 63.6 s, including model load, the first DEM fetch and the upload [docs/screenshots/fresh_clone_report.json].
- **Viewer:** 141–144 fps (median while orbiting) in Chrome on the Intel iGPU in the final recorded run [docs/screenshots/ui_report.json]. An earlier run measured 55 fps on the forest site with the whole terrain in view (not saved to a file).
- **Offline:** once the weights and the area's DEM are cached, the app runs with no network.
- **ONNX** [models/*.json]:
  - DA-V2 Small exports with relative L2 error 7.6 × 10⁻⁷ vs PyTorch.
  - DA3's full forward can't be exported, because its sky step branches on data. The network without that step exports with error 2.0 × 10⁻⁷. That variant matches the app whenever no sky is detected.
  - The app itself uses PyTorch.
- **Launch:** `bash setup/setup_env.sh` once, then `bash scripts/run_app.sh` (README).

## 7. Limitations and next steps

1. **Domain gap.** All validation uses US aerial NAIP, not Cartosat-2S over India: different sensor, viewing geometry and landscapes. No open Cartosat + LiDAR pair was found, and Cartosat-2 imagery is priced (NSIL). **Next:** run on any Cartosat scene the organisers provide, and compare with CartoDEM and Copernicus.
2. **Datum and reference DEM.** All heights are converted to EGM2008 with PROJ geoid grids (EGM96 SRTM, NAVD88 LiDAR, ellipsoidal ICESat-2), and the conversion is recorded. The reference DEM matters more than any method: our default product scores 0.25–0.93 m RMSE against Copernicus at 30 m but 2.5–4.3 m against SRTM (NASADEM). Calibrating to the wrong one costs 1.9–3.7 m [docs/reference_choice.md]. Copernicus is the default because it is clearly better against LiDAR (3.24 / 4.38 / 2.13 m vs 4.34 / 6.59 / 3.67 m calibrated to SRTM). `dem_source=nasadem` switches to SRTM. **Next:** the organisers must confirm the scoring DEM, its datum and the scoring resolution.
3. **Tall objects and derived heights.** One global detail scale, fitted at 30 m, can't recover 20–30 m structures. Every derived height is underestimated by 2–4 m. Each derived height now carries a **low / medium** confidence (there is no *high*), and the UI quotes the measured error for its level (low RMSE 4.7 m, medium 4.4 m on object cells) [docs/building_heights.md]. Shadow-based heights were prototyped and **lost** to the current method on held-out buildings (RMSE 5.47 vs 4.97 m), so they are not used. **Next:** a detail-scale estimator that doesn't depend on low-frequency agreement: the GAMUS-fine-tuned model, ICESat-2 control points (needs a NASA Earthdata account), or shadows on Cartosat with metadata sun angles.
4. **Evidence base.** Three sites, one region, one seed, no confidence intervals. The 2013 LiDAR vs 2021 imagery gap adds error that isn't the model's fault.
5. **Water** is unmasked (RGB only). The crude ground estimate fails under continuous canopy.
