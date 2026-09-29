"""One image in, all products out. The ML engine behind the backend (no web code here).

GeoTIFF (georeferenced) -> metric DSM: tiled model -> relative R -> calibrated with a DEM
PNG/JPG (or no CRS)      -> rDSM only (relative, never metres)

Products in out_dir: dsm.tif (metric only), rdsm.tif, dem.tif (metric only), ground.tif
(derived, metric only), mesh.glb, texture.jpg, metadata.json, calibration.json.
Default calibration = DEM + smooth zero-mean residual (c2): lowest RMSE vs LiDAR at all three Phase 8
sites (docs/results.md). Other methods stay selectable.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from PIL import Image

from depthwizard.calibration import (anchor_pixels, dem_cell_ids, dem_only, dem_plus_residual,
                                     dem_plus_smooth_residual, robust_affine)
from depthwizard.export import export_dsm
from depthwizard.geo import COPERNICUS_GLO30, TILEZEN_SKADI, dem_to_image_grid, fetch_dem_window, footprint_wgs84
from depthwizard.geo.planetary import NASADEM
from depthwizard.io import read_raster
from depthwizard.reconstruction import export_glb, ground_estimate
from depthwizard.tiling import run_tiled

# copernicus = Copernicus GLO-30; nasadem = SRTM (NASADEM reprocessing); skadi = Tilezen blend (USA: 3DEP, not SRTM)
DEM_SOURCES = {"copernicus": COPERNICUS_GLO30, "nasadem": NASADEM, "skadi": TILEZEN_SKADI}
CALIBRATIONS = ("dem_plus_residual", "dem_plus_smooth_residual", "dem_only", "robust_affine")


@dataclass
class PipelineConfig:
    model: str = "da3_mono_large"
    calibration: str = "dem_plus_smooth_residual"   # Phase 8: best at all 3 sites vs LiDAR
    dem_source: str = "copernicus"
    device: str = "cuda"
    tile: int = 1036
    overlap: int = 128
    # "auto" = joint_plane for non-georeferenced input (removes the rDSM tile seam; Phase 11 T3) and sequential for
    # GeoTIFF: with joint_plane the M2 detail scale (fitted on the low-pass) failed the gate at 1 of 3 sites.
    tile_align: str = "auto"            # auto | sequential | joint | joint_plane
    lowpass_m: float = 30.0
    max_slope_deg: float = 15.0
    mesh_max_side: int = 512
    dem_cache: str = "data/dem_cache"
    seed: int = 0
    gate: dict | None = None            # GateConfig overrides
    checkpoint: str | None = None       # fine-tuned weights (training/finetune.py best.pt); None = pretrained
    landcover: bool = True              # ESA WorldCover for derived-height confidence (cached; optional offline)
    band_order: str = "auto"            # auto | rgb | rgbn | bgr | bgrn | pan | "r,g,b" (1-based)
    radiometry: str = "auto"            # auto | bit_depth | stretch (model input only)


_PREDICTORS: dict[tuple, Any] = {}


def get_predictor(model: str, device: str, process_res: int, checkpoint: str | None = None):
    """One loaded model at a time (6 GB VRAM): switching models frees the previous one."""
    key = (model, device, process_res, checkpoint)
    if key not in _PREDICTORS:
        _PREDICTORS.clear()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
        from depthwizard.depth import PREDICTORS
        kw = {"device": device, "process_res": process_res}
        if model == "da2_small":
            kw["fp16"] = device == "cuda"
        pred = PREDICTORS[model](**kw)
        pred.checkpoint = None
        if checkpoint:
            import hashlib
            import torch
            sd = torch.load(checkpoint, map_location="cpu")
            sd = {k[4:] if k.startswith("net.") else k: v for k, v in sd.items()}   # training wrapper prefix
            target = pred.model if model == "da2_small" else pred.model.model
            target.load_state_dict(sd)
            pred.checkpoint = {"path": str(checkpoint),
                               "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}
        _PREDICTORS[key] = pred
    return _PREDICTORS[key]


BAND_ORDERS = {"rgb": (1, 2, 3), "rgbn": (1, 2, 3), "bgr": (3, 2, 1), "bgrn": (3, 2, 1), "pan": (1, 1, 1)}


def resolve_bands(meta, band_order: str = "auto") -> tuple[tuple[int, int, int], str, list[str]]:
    """0-based band indices feeding the model's R, G, B, how they were chosen, and warnings.

    auto: the file's colour interpretation, then band descriptions; 1 band -> panchromatic. Otherwise bands 1-3 are
    ASSUMED to be R,G,B and a warning says so (Cartosat-2S MX is B,G,R,NIR -> band_order="bgrn").
    Explicit: rgb | rgbn | bgr | bgrn | pan | "r,g,b" (1-based band numbers)."""
    n = meta.count if meta is not None else 3
    warns: list[str] = []
    if band_order != "auto":
        idx1 = BAND_ORDERS.get(band_order) or tuple(int(v) for v in band_order.split(","))
        if len(idx1) != 3 or min(idx1) < 1 or max(idx1) > n:
            raise ValueError(f"band_order {band_order!r} does not fit a {n}-band image")
        return tuple(i - 1 for i in idx1), f"band_order={band_order}: R,G,B = bands {idx1}", warns
    if n == 1:
        return (0, 0, 0), "single band repeated to RGB (panchromatic)", warns
    ci = list(getattr(meta, "colorinterp", ()) or ())
    if all(c in ci for c in ("red", "green", "blue")):
        idx = (ci.index("red"), ci.index("green"), ci.index("blue"))
        return idx, f"R,G,B = bands {tuple(i + 1 for i in idx)} (file colour interpretation)", warns
    desc = [(d or "").lower() for d in (getattr(meta, "descriptions", ()) or ())]
    found = {c: next((i for i, d in enumerate(desc) if c in d), None) for c in ("red", "green", "blue")}
    if all(v is not None for v in found.values()):
        idx = (found["red"], found["green"], found["blue"])
        return idx, f"R,G,B = bands {tuple(i + 1 for i in idx)} (band descriptions)", warns
    if n == 2:
        warns.append("2-band image: band 1 used as panchromatic; band 2 ignored")
        return (0, 0, 0), "band 1 repeated to RGB", warns
    warns.append(f"band order is not declared in the file ({n} bands, colour interpretation {ci}): ASSUMED bands "
                 "1-3 = R,G,B. Cartosat-2S multispectral is B,G,R,NIR: set band_order=bgrn")
    return (0, 1, 2), "bands 1-3 assumed R,G,B (undeclared)", warns


def to_uint8_rgb(data: np.ndarray, meta=None, band_order: str = "auto",
                 radiometry: str = "auto") -> tuple[np.ndarray, str, list[str]]:
    """(bands, H, W) any dtype -> (H, W, 3) uint8 for the MODEL only. Output grids are untouched.

    radiometry: auto (uint8 as-is; wider integers scaled by their bit depth, NBITS tag else inferred from the
    maximum; a joint stretch only if the result is clearly underexposed), bit_depth, or stretch (joint
    0.5-99.5 percentile over all three bands: keeps colour balance). Bit-depth scaling of k*uint8 data gives back
    exactly the uint8 values (Phase 11 T4 test)."""
    if radiometry not in ("auto", "bit_depth", "stretch"):
        raise ValueError("radiometry must be auto, bit_depth or stretch")
    idx, how, warns = resolve_bands(meta, band_order)
    x = data[list(idx)]
    nodata = getattr(meta, "nodata", None)
    valid = np.ones(x.shape[1:], bool) if nodata is None else ~np.any(data == nodata, axis=0)

    def stretch(v: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(v[:, valid], (0.5, 99.5)) if valid.any() else (0.0, 1.0)
        return np.clip((v.astype(np.float64) - lo) / max(hi - lo, 1e-9) * 255, 0, 255).round().astype(np.uint8)

    if x.dtype == np.uint8 and radiometry != "stretch":
        out, how = x, how + ", uint8 as-is"
    elif radiometry == "stretch" or x.dtype.kind == "f":
        out, how = stretch(x), how + f", {x.dtype} joint 0.5-99.5 percentile stretch"
    else:
        nb = getattr(meta, "nbits", None)
        vmax = int(x[:, valid].max()) if valid.any() else 255
        bits = nb or max(8, int(np.ceil(np.log2(vmax + 1))))
        out = np.clip(np.round(x.astype(np.float64) * 255.0 / (2 ** bits - 1)), 0, 255).astype(np.uint8)
        how += f", {x.dtype} scaled by bit depth ({bits} bits, {'NBITS tag' if nb else 'inferred from max'})"
        if radiometry == "auto" and valid.any() and np.percentile(out[:, valid], 99) < 64:
            out = stretch(x)
            how += " -> underexposed (p99 < 64/255): joint 0.5-99.5 percentile stretch instead"
            warns.append("image is very dark at its bit depth; a contrast stretch was applied for the model only")
    out = np.where(valid, out, 0).astype(np.uint8)
    return np.moveaxis(out, 0, -1).copy(), how, warns


def write_slope_png(path: Path, z: np.ndarray, res: tuple[float, float], max_side: int = 2048) -> None:
    """Slope in degrees as a mesh texture: viridis over 0-60 deg (perceptually uniform; same scale as layer_slope)."""
    from matplotlib import colormaps
    gy, gx = np.gradient(z.astype(np.float64), res[1], res[0])
    s = np.degrees(np.arctan(np.hypot(gx, gy)))
    rgb = (colormaps["viridis"](np.clip(np.nan_to_num(s) / 60.0, 0, 1))[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(rgb)
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    im.save(path)


def hillshade(z: np.ndarray, res: tuple[float, float], az_deg: float = 315.0, el_deg: float = 45.0) -> np.ndarray:
    """Lambertian shaded relief in [0, 1] (light from the north-west, as on printed relief maps). NaN stays NaN."""
    gy, gx = np.gradient(np.asarray(z, np.float64), res[1], res[0])
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, el = np.radians(360 - az_deg + 90), np.radians(el_deg)
    return np.clip(np.sin(el) * np.cos(slope) + np.cos(el) * np.sin(slope) * np.cos(az - aspect), 0, 1)


STAGES = [("input", "Input & metadata"), ("tiling", "Tiling"), ("depth", "Monocular depth"),
          ("dem", "Reference DEM"), ("calibration", "Calibration"), ("dsm", "DSM"), ("uncertainty", "Uncertainty"),
          ("ndsm", "Ground / nDSM"), ("slope", "Slope"), ("mesh", "Mesh")]
CAL_KEYS = {"dem_only": "M0", "robust_affine": "M1", "dem_plus_residual": "M2_cell",
            "dem_plus_smooth_residual": "M2_smooth"}


class StageRunner:
    """Emits real stage events: start / done / skipped / failed, with wall-clock seconds and layer ids."""

    def __init__(self, emit: Callable[[dict], None] | None, progress: Callable[[str, float], None] | None,
                 titles: dict[str, str]):
        self.emit_fn, self.progress, self.titles, self.log = emit, progress, titles, []

    def _emit(self, ev: dict) -> None:
        ev = {"time": time.time(), "title": self.titles.get(ev.get("stage"), ev.get("stage")), **ev}
        self.log.append(ev)
        if self.emit_fn:
            self.emit_fn(ev)
        if self.progress and ev.get("stage") in self.titles:
            i = [k for k, _ in STAGES].index(ev["stage"])
            done = ev["type"] in ("stage_done", "stage_skipped")
            self.progress(self.titles[ev["stage"]].lower(), round((i + (1 if done else 0.3)) / len(STAGES), 3))

    def run(self, key: str, fn: Callable[[], dict]) -> dict:
        self._emit({"type": "stage_start", "stage": key})
        t0 = time.perf_counter()
        try:
            out = fn() or {}
        except Exception as exc:
            self._emit({"type": "stage_failed", "stage": key, "seconds": round(time.perf_counter() - t0, 2),
                        "error": f"{type(exc).__name__}: {exc}"})
            raise
        self._emit({"type": "stage_done", "stage": key, "seconds": round(time.perf_counter() - t0, 2),
                    "layers": out.get("layers", []), "summary": out.get("summary", {})})
        return out

    def skip(self, key: str, reason: str) -> None:
        self._emit({"type": "stage_skipped", "stage": key, "reason": reason})


def _tiles_png(path: Path, tiles, h: int, w: int, max_side: int = 1024) -> None:
    from PIL import ImageDraw
    f = max(h, w) / max_side if max(h, w) > max_side else 1.0
    im = Image.new("RGBA", (int(round(w / f)), int(round(h / f))), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for t in tiles:
        d.rectangle([t.col0 / f, t.row0 / f, (t.col0 + t.width) / f - 1, (t.row0 + t.height) / f - 1],
                    outline=(0, 229, 255, 230), width=2)
    im.save(path)


def _fit_plots(out: Path, inp, rep: dict) -> list[str]:
    """Scatter of low-pass R vs DEM on fit anchors with the fitted line, and a held-out residual histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from depthwizard.calibration.methods import anchor_masks
    fit_rep = rep.get("scale_fit", rep)
    if "a" not in fit_rep:
        return []
    fit, held, _ = anchor_masks(inp)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(fit)
    idx = rng.choice(idx, min(idx.size, 5000), replace=False)
    x, y = inp.signal.low.ravel()[idx], inp.dem.ravel()[idx]
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=110)
    ax.scatter(x, y, s=2, alpha=0.35, color="#35c7ff")
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
    ax.plot(xs, fit_rep["a"] * xs + fit_rep["b"], color="#2dd4bf", lw=1.5)
    ax.set_xlabel("low-pass relative signal (unitless)"); ax.set_ylabel("DEM (m, EGM2008)")
    ax.set_title(f"robust fit on {idx.size:,} of {fit.sum():,} fit anchors", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "calib_scatter.png", transparent=True); plt.close(fig)
    resid = ((fit_rep["a"] * inp.signal.low + fit_rep["b"]) - inp.dem)[held]
    resid = resid[np.isfinite(resid)]
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=110)
    lim = float(np.percentile(np.abs(resid), 99)) if resid.size else 1.0
    ax.hist(resid, bins=60, range=(-lim, lim), color="#35c7ff")
    ax.set_xlabel("held-out residual vs DEM (m)"); ax.set_ylabel("pixels")
    ax.set_title(f"held-out anchors: {resid.size:,}", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "calib_residuals.png", transparent=True); plt.close(fig)
    return ["calib_scatter.png", "calib_residuals.png"]


def run_pipeline(image_path: str | Path, out_dir: str | Path, cfg: PipelineConfig | None = None,
                 predictor=None, dem_source=None, progress: Callable[[str, float], None] | None = None,
                 emit: Callable[[dict], None] | None = None) -> dict[str, Any]:
    from depthwizard import layers as L
    from depthwizard.calibration import checkerboard, dem_cell_ids
    from depthwizard.calibration.datum import parse_datum, to_egm2008
    from depthwizard.calibration.methods import CalibrationInputs, GateConfig, METHODS
    from depthwizard.calibration.signal import condition
    from depthwizard.geo import COPERNICUS_HEM, COPERNICUS_WBM
    from depthwizard.tiling import plan_tiles

    cfg = cfg or PipelineConfig()
    if cfg.calibration not in CALIBRATIONS:
        raise ValueError(f"calibration must be one of {CALIBRATIONS}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    pred = predictor or get_predictor(cfg.model, cfg.device, cfg.tile, cfg.checkpoint)
    adapted = bool(getattr(pred, "checkpoint", None))
    titles = dict(STAGES)
    titles["depth"] = ("Remote-sensing adapted model" if adapted else "Monocular depth") + f" ({cfg.model})"
    st = StageRunner(emit, progress, titles)
    reg = L.Registry(out)
    S: dict[str, Any] = {"files": {}}

    def stage_input():
        data, meta = read_raster(image_path)
        rgb, how, warns = to_uint8_rgb(data, meta, cfg.band_order, cfg.radiometry)
        if meta.georef_issue:
            warns.append(meta.georef_issue)
        S.update(meta=meta, rgb=rgb, rgb_how=how, georef=meta.is_georeferenced, input_warnings=warns)
        L.rgb_png(out / "layer_rgb.png", rgb)
        Image.fromarray(rgb).save(out / "texture.jpg", quality=90)
        S["files"]["texture"] = "texture.jpg"
        reg.add(L.Layer("rgb", "Input image", "input", "image", "", "layer_rgb.png"))
        summ = {"width": meta.width, "height": meta.height, "bands": meta.count, "dtype": meta.dtype,
                "georeferenced": meta.is_georeferenced, "model_input": how, "warnings": warns,
                "georef_issue": meta.georef_issue}
        if meta.is_georeferenced:
            summ.update(crs=meta.crs.to_string(), gsd_m=[abs(meta.affine.a), abs(meta.affine.e)],
                        footprint_wgs84=list(footprint_wgs84(meta)))
        return {"layers": ["rgb"], "summary": summ}

    def stage_tiling():
        meta = S["meta"]
        tiles = plan_tiles(meta.height, meta.width, cfg.tile, cfg.overlap)
        _tiles_png(out / "layer_tiles.png", tiles, meta.height, meta.width)
        reg.add(L.Layer("tiles", "Tile grid", "tiling", "overlay", "", "layer_tiles.png",
                        notes=[f"{len(tiles)} tiles of {cfg.tile} px, {cfg.overlap} px overlap"]))
        return {"layers": ["tiles"], "summary": {"n_tiles": len(tiles), "tile": cfg.tile, "overlap": cfg.overlap}}

    def stage_depth():
        meta, rgb = S["meta"], S["rgb"]
        align = cfg.tile_align if cfg.tile_align != "auto" else ("sequential" if S["georef"] else "joint_plane")
        mosaic, tinfo = run_tiled(rgb, pred, tile=cfg.tile, overlap=cfg.overlap, align=align)
        gsd = abs(meta.affine.a) if S["georef"] else 1.0
        sig = condition(mosaic, pred.height_sign, gsd, cfg.lowpass_m if S["georef"] else 30.0, pred.polarity)
        S.update(signal=sig, tinfo=tinfo)
        rd = export_dsm(out / "rdsm.tif", sig.r, meta, model=cfg.model, calibration="none", is_metric=False,
                        provenance={"height_sign": pred.height_sign, "polarity": pred.polarity,
                                    "checkpoint": getattr(pred, "checkpoint", None)})
        S["files"]["rdsm"] = rd.name
        lo, hi = L.colour_png(out / "layer_rdepth.png", sig.r, L.CMAPS["relative"])
        reg.add(L.Layer("rdepth", "Relative depth (height-oriented)", "depth", "relative", "relative (unitless)",
                        "layer_rdepth.png", rd.name, L.CMAPS["relative"], lo, hi))
        return {"layers": ["rdepth"], "summary": {"model": cfg.model, "adapted_checkpoint": getattr(pred, "checkpoint", None),
                                                  "polarity": pred.polarity, "height_sign": pred.height_sign,
                                                  "n_tiles": tinfo["n_tiles"], "tile_align": tinfo["align"]}}

    def stage_dem():
        meta = S["meta"]
        src = dem_source or DEM_SOURCES[cfg.dem_source]
        fp = footprint_wgs84(meta, margin_deg=0.001)
        win = fetch_dem_window(fp, cfg.dem_cache, src)
        dem, rinfo = dem_to_image_grid(win.path, meta)
        dem, dinfo = to_egm2008(dem, meta.affine, meta.crs, src.vertical_datum)
        water = dem_error = None
        aux = []
        if src is COPERNICUS_GLO30 or src is NASADEM:
            # the water mask is independent of the heights, so it is used for either DEM (lakes never anchor
            # the fit); the height-error map describes Copernicus only
            from rasterio.enums import Resampling
            wbm = fetch_dem_window(fp, cfg.dem_cache, COPERNICUS_WBM)
            water = np.nan_to_num(dem_to_image_grid(wbm.path, meta, Resampling.nearest)[0]) > 0
            aux = ["water mask (Copernicus WBM)"]
            if src is COPERNICUS_GLO30:
                hem = fetch_dem_window(fp, cfg.dem_cache, COPERNICUS_HEM)
                dem_error = dem_to_image_grid(hem.path, meta)[0]
                aux.append("DEM error (Copernicus HEM)")
        S.update(dem=dem, dem_win=win, water=water, dem_error=dem_error, dem_src=src, datum_info=dinfo)
        f = export_dsm(out / "dem.tif", dem, meta, model="none", calibration="dem_only (bilinear)", is_metric=True,
                       vertical_datum="EGM2008", provenance={"source": src.name, "datum_conversion": dinfo},
                       calibration_dem=src.name)
        S["files"]["dem"] = f.name
        lo, hi = L.colour_png(out / "layer_dem.png", dem, L.CMAPS["height"])
        ids = ["dem"]
        reg.add(L.Layer("dem", "Reference DEM on image grid", "dem", "metric", "m", "layer_dem.png", f.name,
                        L.CMAPS["height"], lo, hi, "EGM2008", [src.name, f"datum: {dinfo['source_datum']} -> EGM2008"]))
        if water is not None:
            L.mask_png(out / "layer_water.png", water, (53, 150, 255, 170))
            reg.add(L.Layer("water", "Water (excluded from fitting)", "dem", "class", "class", "layer_water.png",
                            notes=["Copernicus WBM"]))
            ids.append("water")
        return {"layers": ids, "summary": {"source": src.name, "coverage": rinfo["coverage_fraction"],
                                           "source_datum": dinfo["source_datum"], "aux": aux,
                                           "water_fraction": float(water.mean()) if water is not None else None}}

    def stage_calibration():
        meta, res = S["meta"], (abs(S["meta"].affine.a), abs(S["meta"].affine.e))
        blocks = checkerboard(meta.height, meta.width, max(8, round(154 / res[0])))
        inp = CalibrationInputs(signal=S["signal"], dem=S["dem"], res=res, fit_mask=blocks, eval_mask=~blocks,
                                cell_ids=dem_cell_ids(S["dem_win"].path, meta), water=S["water"],
                                dem_error=S["dem_error"], max_slope_deg=cfg.max_slope_deg, seed=cfg.seed)
        result = METHODS[CAL_KEYS[cfg.calibration]]().calibrate(inp, GateConfig(**(cfg.gate or {})))
        rep = {**result.report, "dem_source": S["dem_src"].name, "dem_datum_conversion": S["datum_info"],
               "config": asdict(cfg)}
        (out / "calibration.json").write_text(json.dumps(rep, indent=2, default=str))
        S["files"]["calibration"] = "calibration.json"
        S.update(result=result, cal_inputs=inp, cal_report=rep)
        ids = []
        from depthwizard.calibration.methods import anchor_masks
        fit, held, _ = anchor_masks(inp)
        L.mask_png(out / "layer_anchors.png", fit, (0, 229, 255, 150))
        reg.add(L.Layer("anchors", "Calibration anchors (fit)", "calibration", "overlay", "", "layer_anchors.png",
                        notes=[f"{int(fit.sum()):,} fit / {int(held.sum()):,} held-out pixels (spatial split)"]))
        ids.append("anchors")
        for p in _fit_plots(out, inp, rep):
            S["files"][p[:-4]] = p
        return {"layers": ids, "summary": {"method": result.method, "gate_passed": result.report["gate"]["passed"],
                                           "gate_reasons": result.report["gate"]["reasons"],
                                           "scale": rep.get("s", rep.get("a")),
                                           "plots": [p for p in ("calib_scatter.png", "calib_residuals.png")
                                                     if (out / p).exists()]}}

    def stage_dsm():
        meta, res_ = S["meta"], S.get("result")
        if res_ is not None and res_.is_metric:
            f = export_dsm(out / "dsm.tif", res_.z, meta, model=cfg.model, calibration=res_.method, is_metric=True,
                           vertical_datum="EGM2008", provenance={"calibration_report": "calibration.json"},
                           calibration_dem=S["dem_src"].name)
            S["files"]["dsm"] = f.name
            lo, hi = L.colour_png(out / "layer_dsm.png", res_.z, L.CMAPS["height"])
            reg.add(L.Layer("dsm", "Metric DSM", "dsm", "metric", "m", "layer_dsm.png", f.name, L.CMAPS["height"],
                            lo, hi, "EGM2008", [res_.method]))
            return {"layers": ["dsm"], "summary": {"kind": "metric", "datum": "EGM2008", "method": res_.method}}
        why = res_.report["gate"]["reasons"] if res_ is not None else ["no map coordinates"]
        lo, hi = L.colour_png(out / "layer_dsm.png", S["signal"].r, L.CMAPS["relative"])
        reg.add(L.Layer("dsm", "Relative DSM (rDSM)", "dsm", "relative", "relative (unitless)", "layer_dsm.png",
                        S["files"]["rdsm"], L.CMAPS["relative"], lo, hi, notes=why))
        return {"layers": ["dsm"], "summary": {"kind": "relative", "reasons": why}}

    def stage_uncertainty():
        res_ = S["result"]
        f = export_dsm(out / "uncertainty.tif", res_.uncertainty, S["meta"], model=cfg.model,
                       calibration=f"uncertainty ({res_.report.get('uncertainty_model')})", is_metric=True,
                       vertical_datum="EGM2008 (1-sigma estimate, metres)", calibration_dem=S["dem_src"].name)
        S["files"]["uncertainty"] = f.name
        lo, hi = L.colour_png(out / "layer_uncertainty.png", res_.uncertainty, L.CMAPS["uncertainty"], vmin=0.0)
        reg.add(L.Layer("uncertainty", "Uncertainty (1-sigma estimate)", "uncertainty", "derived", "m",
                        "layer_uncertainty.png", f.name, L.CMAPS["uncertainty"], lo, hi,
                        notes=[res_.report.get("uncertainty_model", ""), "low-confidence pixels are transparent"]))
        ids = ["uncertainty"]
        if res_.low_confidence is not None and res_.low_confidence.any():
            L.mask_png(out / "layer_lowconf.png", res_.low_confidence, (255, 176, 32, 150))
            reg.add(L.Layer("lowconf", "Low confidence", "uncertainty", "overlay", "", "layer_lowconf.png"))
            ids.append("lowconf")
        return {"layers": ids, "summary": res_.report.get("uncertainty_m", {})}

    def landcover_on_grid(meta):
        """ESA WorldCover classes on the image grid, or (None, reason). Never fatal: offline runs keep working."""
        if not cfg.landcover or dem_source is not None:      # injected DEM = test/custom run: no network
            return None, "land cover disabled for this run"
        try:
            from rasterio.enums import Resampling
            from depthwizard.geo.planetary import esa_worldcover
            win = fetch_dem_window(footprint_wgs84(meta, margin_deg=0.001), cfg.dem_cache, esa_worldcover(2021))
            return dem_to_image_grid(win.path, meta, Resampling.nearest)[0], "ESA WorldCover 10 m 2021 (CC-BY-4.0)"
        except Exception as exc:                            # network/cache miss: say so, don't guess
            return None, f"land cover unavailable ({type(exc).__name__}): tree cover NOT checked"

    def stage_ndsm():
        from depthwizard.calibration.ndsm import LEVEL_NAMES, REASONS, estimate_ground, height_confidence
        meta, z = S["meta"], S["result"].z
        res = (abs(meta.affine.a), abs(meta.affine.e))
        g = estimate_ground(z, res, water=S["water"])
        S["ground"] = g
        lc, lc_note = landcover_on_grid(meta)
        level, why = height_confidence(g, S["result"].uncertainty, lc, S["water"])
        with rasterio.open(out / "height_confidence.tif", "w", driver="GTiff", width=meta.width, height=meta.height,
                           count=2, dtype="uint8", crs=meta.crs, transform=meta.affine, nodata=0,
                           compress="deflate") as d:
            d.write(level, 1)
            d.write(why, 2)
            d.update_tags(PRODUCT="DERIVED height-above-ground confidence", LEVELS=json.dumps(LEVEL_NAMES),
                          REASON_BITS=json.dumps(REASONS), LANDCOVER=lc_note,
                          NOTE="band 1 = level, band 2 = reason bitmask; no HIGH level (not validated)")
            d.set_band_description(1, "confidence level")
            d.set_band_description(2, "reason bitmask")
        S["files"]["height_confidence"] = "height_confidence.tif"
        S["landcover_note"] = lc_note
        gf = export_dsm(out / "ground.tif", g.ground, meta, model="derived", calibration=f"DERIVED ground: {g.method}",
                        is_metric=True, vertical_datum="EGM2008", calibration_dem=S["dem_src"].name)
        nf = export_dsm(out / "ndsm.tif", g.ndsm, meta, model="derived", calibration="DERIVED nDSM = DSM - ground",
                        is_metric=True, vertical_datum="height above derived ground (metres)")
        S["files"].update(ground=gf.name, ndsm=nf.name)
        lo, hi = L.colour_png(out / "layer_ndsm.png", g.ndsm, L.CMAPS["ndsm"], vmin=0.0)
        reg.add(L.Layer("ndsm", "Object height nDSM (derived)", "ndsm", "derived", "m", "layer_ndsm.png", nf.name,
                        L.CMAPS["ndsm"], lo, hi, notes=["derived: DSM minus an estimated ground", g.method]))
        ids = ["ndsm"]
        L.mask_png(out / "layer_height_lowconf.png", level == 1, (255, 176, 32, 160))
        reg.add(L.Layer("height_lowconf", "Derived height: LOW confidence", "ndsm", "overlay", "",
                        "layer_height_lowconf.png", notes=[lc_note] + [f"{t}: {100 * float(((why & b) > 0).mean()):.0f} %"
                                                                        for b, t in REASONS.items()]))
        ids.append("height_lowconf")
        if g.low_confidence.any():
            L.mask_png(out / "layer_ground_lowconf.png", g.low_confidence, (255, 176, 32, 150))
            reg.add(L.Layer("ground_lowconf", "Ground estimate: low confidence", "ndsm", "overlay", "",
                            "layer_ground_lowconf.png", notes=g.notes))
            ids.append("ground_lowconf")
        valid = level > 0
        summ = {**g.summary, "height_confidence": {
            "low_fraction": float((level == 1).sum() / max(valid.sum(), 1)),
            "medium_fraction": float((level == 2).sum() / max(valid.sum(), 1)),
            "reason_fractions": {t: float(((why & b) > 0).sum() / max(valid.sum(), 1)) for b, t in REASONS.items()},
            "landcover": lc_note, "levels_available": ["low", "medium"]}}
        return {"layers": ids, "summary": summ}

    def stage_slope():
        meta, z = S["meta"], S["result"].z
        res = (abs(meta.affine.a), abs(meta.affine.e))
        from depthwizard.calibration import slope_deg
        sl = slope_deg(z, *res)
        f = export_dsm(out / "slope.tif", sl, meta, model="derived", calibration="DERIVED slope (degrees)",
                       is_metric=True, vertical_datum="n/a (degrees)")
        S["files"]["slope_tif"] = f.name
        write_slope_png(out / "slope.png", z, res)
        S["files"]["slope"] = "slope.png"
        L.colour_png(out / "layer_slope.png", sl, L.CMAPS["slope"], vmin=0.0, vmax=60.0)
        reg.add(L.Layer("slope", "Slope (derived)", "slope", "derived", "degrees", "layer_slope.png", f.name,
                        L.CMAPS["slope"], 0.0, 60.0))
        return {"layers": ["slope"], "summary": {"median_deg": float(np.nanmedian(sl))}}

    def stage_mesh():
        meta, rgb = S["meta"], S["rgb"]
        res_ = S.get("result")
        if res_ is not None and res_.is_metric:
            mesh_z, pixel = res_.z, (abs(meta.affine.a), abs(meta.affine.e))
            S["units"] = "metres"
        else:
            r = S["signal"].r
            span = float(np.nanmax(r) - np.nanmin(r)) or 1.0
            S["display_z_scale"] = 0.05 * max(meta.width, meta.height) / span
            mesh_z, pixel, S["units"] = r * S["display_z_scale"], (1.0, 1.0), "relative (display-scaled)"
        L.colour_png(out / "layer_hillshade.png", hillshade(mesh_z, pixel), "gray", vmin=0.0, vmax=1.0)
        reg.add(L.Layer("hillshade", "Shaded relief of the mesh surface", "mesh", "overlay", "", "layer_hillshade.png",
                        notes=["display: light from the north-west, 45 deg"]))
        minfo = export_glb(out / "mesh.glb", mesh_z, rgb, pixel, max_side=cfg.mesh_max_side)
        lod = export_glb(out / "mesh_lod1.glb", mesh_z, rgb, pixel, max_side=max(32, cfg.mesh_max_side // 4))
        minfo["lod1"] = {k: lod[k] for k in ("decimation", "grid", "n_vertices", "z_offset")}
        S.update(minfo=minfo, mesh_z=mesh_z)
        S["files"].update(mesh="mesh.glb", mesh_lod1="mesh_lod1.glb")
        return {"layers": ["hillshade"], "summary": {"vertices": minfo["n_vertices"], "faces": minfo["n_faces"],
                                          "lod1_vertices": lod["n_vertices"], "units": S["units"]}}

    st.run("input", stage_input)
    st.run("tiling", stage_tiling)
    st.run("depth", stage_depth)
    if S["georef"]:
        st.run("dem", stage_dem)
        st.run("calibration", stage_calibration)
    else:
        why_rel = S["meta"].georef_issue or "the image has no map coordinates"
        st.skip("dem", f"{why_rel}: no DEM can be placed on it")
        st.skip("calibration", f"{why_rel}: output stays relative (rDSM), never metres")
    st.run("dsm", stage_dsm)
    metric = S.get("result") is not None and S["result"].is_metric
    for key, fn in (("uncertainty", stage_uncertainty), ("ndsm", stage_ndsm), ("slope", stage_slope)):
        if metric:
            st.run(key, fn)
        else:
            st.skip(key, "needs a metric DSM (relative output only)")
    st.run("mesh", stage_mesh)

    meta, res_ = S["meta"], S.get("result")
    if not S["georef"]:
        (out / "calibration.json").write_text(json.dumps({"method": None, "is_metric": False,
                                                          "reason": "no map coordinates",
                                                          "display_z_scale": S.get("display_z_scale")}, indent=2))
        S["files"]["calibration"] = "calibration.json"
    mz = S["mesh_z"]
    finite = (res_.z if metric else S["signal"].r)
    finite = finite[np.isfinite(finite)]
    kind = "metric DSM" if metric else "relative DSM (rDSM)"
    metadata = {
        "input": {"path": str(image_path), "georeferenced": S["georef"], "width": meta.width, "height": meta.height,
                  "bands": meta.count, "dtype": meta.dtype, "model_input": S["rgb_how"],
                  "warnings": S["input_warnings"], "georef_issue": meta.georef_issue,
                  "crs": meta.crs.to_string() if S["georef"] else None,
                  "bounds": list(meta.bounds) if S["georef"] else None,
                  "footprint_wgs84": list(footprint_wgs84(meta)) if S["georef"] else None,
                  "gsd_m": [abs(meta.affine.a), abs(meta.affine.e)] if S["georef"] else None},
        "product": {"kind": kind, "units": "metres" if metric else "relative (unitless)",
                    "vertical_datum": "EGM2008" if metric else "none (relative)",
                    "model": cfg.model, "adapted_checkpoint": getattr(pred, "checkpoint", None),
                    "calibration": res_.method if res_ is not None else "none",
                    "calibration_dem": S["dem_src"].name if S.get("dem_src") is not None else None,
                    "gate": res_.report["gate"] if res_ is not None else None,
                    "height_range": [float(finite.min()), float(finite.max())]},
        "mesh": {**S["minfo"], "units": "metres" if metric else "pixels (z display-scaled)",
                 "origin": "image upper-left corner; x east, y north, z = height - z_offset",
                 "note": "display product; query the DSM for measurements"},
        "tiling": {k: S["tinfo"][k] for k in ("tile", "overlap", "n_tiles", "align")},
        "stages": st.log, "layers": reg.as_list(),
        "config": asdict(cfg), "files": S["files"], "seconds": round(time.perf_counter() - t_start, 2),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    if emit:
        emit({"type": "done", "time": time.time(), "seconds": metadata["seconds"], "product": metadata["product"]})
    return metadata


def query_point(job_dir: str | Path, row: int | None = None, col: int | None = None,
                x: float | None = None, y: float | None = None) -> dict[str, Any]:
    """Elevation at a pixel (row/col) or CRS coordinate (x/y) from the job's products."""
    job_dir = Path(job_dir)
    main = job_dir / ("dsm.tif" if (job_dir / "dsm.tif").exists() else "rdsm.tif")
    with rasterio.open(main) as src:
        if row is None:
            if x is None or y is None:
                raise ValueError("give row/col or x/y")
            row, col = src.index(x, y)
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise IndexError("point outside the image")
        v = float(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
        cx, cy = src.xy(row, col)
        tags = src.tags()
    outd: dict[str, Any] = {"row": int(row), "col": int(col), "value": None if np.isnan(v) else v,
                            "units": tags.get("UNITS"), "kind": tags.get("DSM_KIND"),
                            "vertical_datum": tags.get("VERTICAL_DATUM")}
    if tags.get("DSM_KIND") == "absolute":
        outd["x"], outd["y"] = cx, cy
        with rasterio.open(job_dir / "ground.tif") as g:
            gv = float(g.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
        if np.isfinite(gv) and not np.isnan(v):
            from depthwizard.calibration.ndsm import LEVEL_NAMES, describe_reasons
            outd["ground_estimate"] = gv
            outd["derived_height_above_ground"] = v - gv
            outd["derived_note"] = ("DERIVED: DSM minus an estimated local ground (low percentile of smooth pixels in "
                                    "60 m windows); not measured")
            conf = {"level": "unknown", "reasons": ["no confidence map for this job (older run)"]}
            if (job_dir / "height_confidence.tif").exists():
                with rasterio.open(job_dir / "height_confidence.tif") as c:
                    lv, bits = (int(b) for b in c.read(window=((row, row + 1), (col, col + 1)))[:, 0, 0])
                    lc = c.tags().get("LANDCOVER", "")
                conf = {"level": LEVEL_NAMES.get(lv, "none"), "reasons": describe_reasons(bits),
                        "landcover": lc, "note": "levels are low / medium only; no derived height is rated high"}
                ev = Path(__file__).parent / "calibration" / "height_confidence_evidence.json"
                if ev.exists():                               # measured error for this level (named run)
                    e = json.loads(ev.read_text())
                    if conf["level"] in e["levels"]:
                        conf["measured_error"] = {**e["levels"][conf["level"]], "scope": e["scope"], "source": e["source"]}
            outd["derived_height_confidence"] = conf
    return outd


def compare_reference(job_dir: str | Path, ref_path: str | Path) -> dict[str, Any]:
    """Compare a job's metric DSM with a user-supplied reference DSM GeoTIFF.

    For DISPLAY and pixel metrics the reference is bilinear-resampled onto the job grid (recorded).
    If the reference is coarser, metrics after area-averaging the DSM onto the reference grid are
    also reported (the fairer comparison for 30 m DEMs). Writes reference_diff.png + reference_metrics.json.
    """
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from depthwizard.evaluation import evaluate_rasters, metrics

    job_dir = Path(job_dir)
    dsm_path = job_dir / "dsm.tif"
    if not dsm_path.exists():
        raise ValueError("reference comparison needs a metric DSM (this job produced a relative rDSM)")
    with rasterio.open(dsm_path) as d, rasterio.open(ref_path) as r:
        if r.crs is None:
            raise ValueError("reference has no CRS")
        dsm = d.read(1).astype(np.float32)
        ref = np.full(dsm.shape, np.nan, np.float32)
        reproject(rasterio.band(r, 1), ref, src_transform=r.transform, src_crs=r.crs,
                  src_nodata=r.nodata if r.nodata is not None else np.nan, dst_transform=d.transform,
                  dst_crs=d.crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
        coarser = abs(r.res[0]) > abs(d.res[0]) * 1.5 if r.crs.is_projected == d.crs.is_projected else True
        ref_info = {"crs": r.crs.to_string()[:120], "res": list(r.res), "shape": [r.height, r.width],
                    "vertical_datum_tag": r.tags().get("VERTICAL_DATUM")}
        dsm_datum = d.tags().get("VERTICAL_DATUM")
    if not np.isfinite(ref).any():
        raise ValueError("reference does not overlap the DSM")
    diff = dsm - ref
    rep: dict[str, Any] = {"reference": ref_info, "dsm_vertical_datum": dsm_datum,
                           "on_dsm_grid": metrics(dsm, ref),
                           "on_dsm_grid_note": "reference bilinear-resampled onto the DSM grid for this comparison",
                           "coverage_fraction": float(np.isfinite(ref).mean())}
    if coarser:
        try:
            agg = evaluate_rasters(dsm_path, ref_path, mode="aggregate")
            rep["aggregated_to_reference_grid"] = agg["overall"]
            rep["datum_check"] = agg["vertical_datum"]["check"]
        except ValueError as exc:
            rep["aggregated_to_reference_grid"] = f"not computed: {exc}"
    if "datum_check" not in rep:
        from depthwizard.evaluation import datum_check
        rep["datum_check"] = datum_check(dsm_datum, ref_info["vertical_datum_tag"])
    # diverging overlay, perceptually uniform (Crameri "berlin"), symmetric +-p98, transparent where no data
    from depthwizard import layers as L
    lim = float(np.nanpercentile(np.abs(diff), 98)) or 1.0
    L.colour_png(job_dir / "reference_diff.png", diff, "berlin", vmin=-lim, vmax=lim, max_side=2048)
    rep["diff_colour_limit_m"] = lim
    layers_path = job_dir / "layers.json"
    if layers_path.exists():                               # make the difference a layer for the 2D compare
        lay = [x for x in json.loads(layers_path.read_text()) if x["id"] != "diff"]
        lay.append({"id": "diff", "title": "DSM minus reference", "stage": "validation", "kind": "derived",
                    "units": "m", "preview": "reference_diff.png", "data": None, "colormap": "berlin",
                    "vmin": -lim, "vmax": lim, "datum": rep.get("datum_check"),
                    "notes": [f"reference: {Path(ref_path).name}"]})
        layers_path.write_text(json.dumps(lay, indent=2))
    (job_dir / "reference_metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep
