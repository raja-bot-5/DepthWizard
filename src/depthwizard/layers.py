"""Layer registry + decimated PNG previews for the UI.

Every layer the pipeline produces is registered with its kind, units, colormap and value range, so the
UI can draw a legend with real units. Relative layers are never labelled in metres; derived layers say
so. Full-resolution data stays in the GeoTIFFs (downloaded on demand); previews are decimated to at most
`max_side` pixels with an exact block mean (no interpolation of values across NaNs).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from PIL import Image

KINDS = ("image", "metric", "relative", "derived", "class", "overlay")
# perceptually uniform only (Phase 11 T6): viridis / cividis / magma; diverging = Crameri "berlin"
CMAPS = {"height": "viridis", "relative": "cividis", "uncertainty": "magma", "slope": "viridis",
         "diverging": "berlin", "ndsm": "magma"}


@dataclass
class Layer:
    id: str
    title: str
    stage: str
    kind: str                         # one of KINDS
    units: str                        # "m", "relative (unitless)", "degrees", "class", "" for images
    preview: str                      # PNG file name in the job folder
    data: str | None = None           # GeoTIFF with the real values, if any
    colormap: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    datum: str | None = None          # "EGM2008" for metric heights
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"bad layer kind {self.kind!r}")
        if self.kind == "relative" and "m" == self.units.strip():
            raise ValueError("a relative layer cannot be in metres")
        if self.kind == "metric" and not self.datum:
            raise ValueError("a metric height layer must state its datum")


def decimate(a: np.ndarray, max_side: int) -> np.ndarray:
    f = max(1, int(np.ceil(max(a.shape[:2]) / max_side)))
    if f == 1:
        return a
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    b = a[:h, :w].reshape(h // f, f, w // f, f, *a.shape[2:]).astype(np.float64)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3)) if np.isnan(b).any() else b.mean(axis=(1, 3))


def colour_png(path: Path, values: np.ndarray, cmap: str, vmin: float | None = None, vmax: float | None = None,
               max_side: int = 1024, symmetric: bool = False) -> tuple[float, float]:
    """Colour-map a float raster into an RGBA PNG (NaN -> transparent). Returns the (vmin, vmax) used."""
    v = decimate(np.asarray(values, np.float32), max_side)
    finite = v[np.isfinite(v)]
    if vmin is None or vmax is None:
        lo, hi = (np.percentile(finite, (2, 98)) if finite.size else (0.0, 1.0))
        if symmetric:
            m = float(max(abs(lo), abs(hi)))
            lo, hi = -m, m
        vmin = float(lo) if vmin is None else vmin
        vmax = float(hi) if vmax is None else vmax
    t = np.clip((v - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    rgba = (colormaps[cmap](np.nan_to_num(t)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(v), 255, 0)
    Image.fromarray(rgba, "RGBA").save(path)
    return float(vmin), float(vmax)


def rgb_png(path: Path, rgb: np.ndarray, max_side: int = 1024) -> None:
    im = Image.fromarray(rgb)
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    im.save(path)


def mask_png(path: Path, mask: np.ndarray, rgba=(0, 229, 255, 160), max_side: int = 1024) -> None:
    m = decimate(mask.astype(np.float32), max_side) > 0.5
    out = np.zeros((*m.shape, 4), np.uint8)
    out[m] = rgba
    Image.fromarray(out, "RGBA").save(path)


class Registry:
    def __init__(self, job_dir: Path):
        self.dir = Path(job_dir)
        self.layers: list[Layer] = []

    def add(self, layer: Layer) -> Layer:
        self.layers = [x for x in self.layers if x.id != layer.id] + [layer]
        self.save()
        return layer

    def save(self) -> None:
        (self.dir / "layers.json").write_text(json.dumps([asdict(x) for x in self.layers], indent=2))

    def as_list(self) -> list[dict[str, Any]]:
        return [asdict(x) for x in self.layers]
