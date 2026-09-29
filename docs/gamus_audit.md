# GAMUS audit

**Date:** 2026-09-29
**Source:** Hugging Face `earthflow/GAMUS` (`XShadow/GAMUS` redirects there), pinned to revision `a3c0e2511f06d909612406f436cf8abb4da805f5`.
**License:** CC-BY-4.0, as stated on the dataset card.
**Audited sample:** 30 test tiles (DC 10, NYC 10, PHL 10) and 40 train tiles (DC 20, PHL 20). Tiles were chosen evenly spaced per city. The sha256 of every file is in `data/raw/gamus/manifest_{test,train}.json`.
**Audit script:** `experiments/02_gamus_audit/audit_gamus.py`. Its output is `runs/gamus_audit/20260928T192600Z/audit.json`, which covers the test subset.

Each item below is marked **MEASURED** (computed from the files), **STATED** (from a source, cited) or **UNKNOWN — NEEDS VERIFICATION**.

## Size and splits

| Item | Value | Status |
|---|---|---|
| Total size | 80.0 GB, 26,174 files | MEASURED (HF API) |
| Per tile | image 3.15 MB + heights 4.2 MB + classes 1.9 MB (about 9.3 MB) | MEASURED (HF API) |
| Train split | DC 1439, NYC 1167, PHL 2398 (5004 tiles) | MEASURED (HF file list) |
| Val split | DC 359, PHL 500 (859 tiles) | MEASURED |
| Test split | DC 361, NYC 1000, PHL 1500 (2861 tiles) | MEASURED |
| Cities | DC, NYC, PHL only. The OMA/JAX tiles from DFC2019 were removed in this version. | MEASURED + STATED (EarthNets/RSI-MMSegmentation README) |

The paper's older count of 11,507 tiles (6304/1059/4144) does **not** match this revision.

## File format

| Item | Value | Status |
|---|---|---|
| Container | HDF5, one dataset per file, key `image` | MEASURED |
| RGB | `(1024, 1024, 3) uint8` | MEASURED |
| Height (AGL / nDSM) | `(1024, 1024) float32`, no NaNs in the sample | MEASURED |
| Classes | `(1024, 1024)`: **float32 for DC, uint8 for NYC and PHL**. All values are whole numbers 0–6. | MEASURED |
| File names | DC and PHL images are `<id>_RGB.h5`; **NYC images are `<id>_IMG.h5`** | MEASURED |
| Metadata | **None**: no HDF5 attributes, so no units, GSD, CRS or datum | MEASURED |
| Georeferencing | None | MEASURED |

## Height label values

| City (test sample) | Min | Max | Mean | Fraction < 0 |
|---|---|---|---|---|
| DC | **−5.0** (1 tile, 0.016% px) | 128.2 | 5.69 | 0.05% |
| NYC | −1.76 | 34.9 | 7.86 | **5.9%** |
| PHL | **0.0** | 141.6 | 2.84 | 0.0% |

Median height by class (sampled pixels, all cities):

| Class | Median | p95 |
|---|---|---|
| ground | 0.006 | 1.87 |
| road | 0.010 | 10.24 |
| low vegetation | 0.021 | 2.07 |
| buildings | 7.19 | 19.92 |
| tree | 11.01 | 26.46 |
| water | 0.0 | 2.08 |

- **Units: consistent with metres, but not declared anywhere in the data.** Ground sits near 0, a typical building is about 7, trees about 11, and the maximum is about 140 (tall towers). These are physically plausible for metres above ground. **UNKNOWN — NEEDS VERIFICATION** against the GAMUS paper's label definition.
- **Negative values and floors differ by city.** PHL looks clipped at 0, DC at −5 (only one tile hits it), and NYC is not clipped (5.9% negative, down to −1.76). **The cities were processed differently.** Training masks `AGL <= -5` as invalid and keeps small negatives, because they're plausibly LiDAR noise around ground level.
- **The label is height above ground, not a DSM.** It contains no terrain and no vertical datum. It's suited to training fine structure (buildings, trees), not absolute elevation.

## Things the data does not tell us

| Item | Status |
|---|---|
| GSD | **UNKNOWN — NEEDS VERIFICATION.** A third-party README says 0.5 m/px; the files don't state it. |
| Imagery sensor, date, and whether it's aerial or satellite | UNKNOWN (described as aerial in secondary sources) |
| How AGL was derived (LiDAR DSM minus DTM? which filter?) | UNKNOWN — see the GAMUS paper (arXiv:2305.14914) |
| Time gap between imagery and LiDAR | UNKNOWN |

## Consequences for training (`training/finetune.py`)

- **City-level holdout:** train on DC + PHL, validate on **NYC**. NYC is also the city with the different processing convention (negatives kept, `_IMG` names), so validation there partly tests robustness to label conventions.
- **Crops** are taken at native resolution (518 px). There's no resizing, so GSD stays about 0.5 m, close to Cartosat-2S at 0.6 m.
- **Augmentation** uses horizontal flips and 90° rotations. Both keep the label valid for straight-down imagery.
- **The loss is scale- and shift-invariant**, because the labels are relative to local ground and the models' outputs are relative too.
- **Mask:** `AGL <= -5` (the floor value) is excluded.
- The class raster must be cast from float32 or uint8 to int before use.
