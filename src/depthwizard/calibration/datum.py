"""Vertical datums: every height is converted to EGM2008 orthometric before any fit or comparison.

Conversions use PROJ with its geoid grids (installed in ~/.local/share/proj; hashes in
setup/geoid_grids.sha256). PROJ silently falls back to a no-op "ballpark" vertical transformation when a
grid is missing, so every transformer is checked and a missing grid raises DatumGridMissing, with the
grid name, size and source. Nothing is converted silently.

  Copernicus GLO-30 : EGM2008 (native; tile XML: vertical "WGS 84 Geoid EGM08", EPSG::1027)
  SRTM / skadi      : EGM96  -> converted
  ICESat-2 / GEDI   : WGS84 ellipsoidal heights -> converted
  USGS 3DEP LiDAR   : NAVD88 (GEOID18) -> converted
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from pyproj import Transformer
from pyproj.transformer import TransformerGroup

TARGET = "EGM2008"
# compound CRS with geographic lon/lat + vertical datum; ellipsoidal = 3D geographic
# NAVD88 uses WGS84 lon/lat input (EPSG:4269+5703 returns inf in PROJ 9.5; the <1 m horizontal NAD83/WGS84
# difference is irrelevant for a geoid that varies over kilometres)
CRS3D = {"EGM2008": "EPSG:4326+3855", "EGM96": "EPSG:4326+5773", "NAVD88": "EPSG:4326+5703",
         "WGS84_ELLIPSOID": "EPSG:4979"}
KNOWN_GRIDS = {  # for the error message when PROJ cannot find a grid
    "us_nga_egm08_25.tif": "80.6 MB",
    "us_nga_egm96_15.tif": "2.7 MB",
    "us_noaa_g2018u0.tif": "16.7 MB",
}


class DatumGridMissing(RuntimeError):
    pass


def parse_datum(text: str | None) -> str:
    """Map a free-text VERTICAL_DATUM tag to a key of CRS3D. Unknown -> ValueError (never guess)."""
    t = (text or "").upper()
    if "EGM2008" in t or "EGM08" in t:
        return "EGM2008"
    if "EGM96" in t:
        return "EGM96"
    if "NAVD88" in t or "NAVD 88" in t:
        return "NAVD88"
    if "ELLIPSOID" in t or t in ("WGS84", "WGS 84"):
        return "WGS84_ELLIPSOID"
    raise ValueError(f"UNKNOWN vertical datum {text!r} - NEEDS VERIFICATION; refusing to convert")


@lru_cache(maxsize=16)
def transformer(src: str, dst: str = TARGET) -> Transformer:
    """Best-available transformer, refused if PROJ would use a ballpark (no-op) vertical step."""
    t = Transformer.from_crs(CRS3D[src], CRS3D[dst], always_xy=True, only_best=True)
    if src != dst and "ballpark" in t.description.lower():
        missing = []
        for op in TransformerGroup(CRS3D[src], CRS3D[dst], always_xy=True).unavailable_operations:
            for g in op.grids:
                if not g.available:
                    missing.append(f"{g.short_name} ({KNOWN_GRIDS.get(g.short_name, 'size unknown')}, {g.url or 'cdn.proj.org'})")
        raise DatumGridMissing(f"{src} -> {dst} needs geoid grid(s) {sorted(set(missing)) or 'UNKNOWN'}; "
                               "PROJ would otherwise apply a no-op ballpark transform")
    return t


def convert(h: np.ndarray, lon: np.ndarray, lat: np.ndarray, src: str, dst: str = TARGET) -> np.ndarray:
    """Point-wise height conversion (float64 in, float32 out). NaN heights stay NaN."""
    h = np.asarray(h, dtype=np.float64)
    if src == dst:
        return h.astype(np.float32)
    _, _, z = transformer(src, dst).transform(np.asarray(lon, np.float64), np.asarray(lat, np.float64), h)
    z = np.asarray(z, dtype=np.float64)
    if np.any(np.isfinite(h) & ~np.isfinite(z)):
        raise DatumGridMissing(f"{src} -> {dst} produced non-finite heights for finite input (grid coverage or "
                               "CRS definition problem); refusing to continue")
    return z.astype(np.float32)


@dataclass(frozen=True)
class DatumShift:
    src: str
    dst: str
    offset: np.ndarray            # dst_height - src_height on the raster grid (float32, metres)
    info: dict[str, Any]


def raster_shift(transform, crs, height: int, width: int, src: str, dst: str = TARGET,
                 step_px: int = 64) -> DatumShift:
    """Datum offset (dst - src) for every pixel of a raster grid.

    Geoid models vary over tens of km, so the offset is evaluated exactly at a coarse lattice of pixel
    centres (every step_px) and bilinearly interpolated in between. The max interpolation error is
    reported (checked against exact values at lattice midpoints).
    """
    from pyproj import Transformer as T
    if src == dst:
        return DatumShift(src, dst, np.zeros((height, width), np.float32), {"method": "identity"})
    to_ll = T.from_crs(crs, "EPSG:4326", always_xy=True)
    rr = np.unique(np.r_[np.arange(0, height, step_px), height - 1])
    cc = np.unique(np.r_[np.arange(0, width, step_px), width - 1])
    R, C = np.meshgrid(rr, cc, indexing="ij")
    x, y = transform * (C + 0.5, R + 0.5)
    lon, lat = to_ll.transform(x, y)
    zero = np.zeros_like(lon)
    coarse = convert(zero, lon, lat, src, dst)             # height 0 in src -> height in dst = offset
    offset = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR) if coarse.size > 1 else \
        np.full((height, width), float(coarse.ravel()[0]), np.float32)
    # interpolation check at lattice-cell midpoints
    if len(rr) > 1 and len(cc) > 1:
        mr, mc = (rr[:-1] + rr[1:]) // 2, (cc[:-1] + cc[1:]) // 2
        MR, MC = np.meshgrid(mr, mc, indexing="ij")
        mx, my = transform * (MC + 0.5, MR + 0.5)
        mlon, mlat = to_ll.transform(mx, my)
        exact = convert(np.zeros_like(mlon), mlon, mlat, src, dst)
        interp_err = float(np.abs(exact - offset[MR, MC]).max())
    else:
        interp_err = 0.0
    t = transformer(src, dst)
    return DatumShift(src, dst, offset.astype(np.float32), {
        "method": f"PROJ exact on a {step_px}-px lattice + bilinear",
        "pipeline": t.description if "unavailable" not in t.description else
        TransformerGroup(CRS3D[src], CRS3D[dst], always_xy=True).transformers[0].description,
        "offset_min_m": float(offset.min()), "offset_max_m": float(offset.max()),
        "max_interp_error_m": interp_err})


def to_egm2008(z: np.ndarray, transform, crs, src_datum_text: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert a raster of heights (on the given grid) to EGM2008. Returns (heights, metadata record)."""
    src = parse_datum(src_datum_text)
    shift = raster_shift(transform, crs, z.shape[0], z.shape[1], src, TARGET)
    out = (np.asarray(z, np.float32) + shift.offset).astype(np.float32)
    return out, {"source_datum": src, "source_datum_text": src_datum_text, "target_datum": TARGET, **shift.info}


def raster_datum_text(tags: dict, crs) -> str:
    """Vertical datum of a raster: the VERTICAL_DATUM tag if present, else the vertical part of a compound CRS.

    Only the vertical sub-CRS name is used (never the whole WKT, which always contains words like ELLIPSOID).
    No tag and no vertical CRS -> ValueError (never guess).
    """
    if tags.get("VERTICAL_DATUM"):
        return tags["VERTICAL_DATUM"]
    from pyproj import CRS
    c = CRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
    vert = [s.name for s in c.sub_crs_list if s.is_vertical]
    if vert:
        return vert[0]
    raise ValueError("raster has no VERTICAL_DATUM tag and no vertical CRS - UNKNOWN datum, NEEDS VERIFICATION")


def raster_file_to_egm2008(path, out) -> tuple[Any, dict[str, Any]]:
    """Write an EGM2008 copy of a single-band height GeoTIFF (same grid, float32, NaN nodata)."""
    import json

    import rasterio
    with rasterio.open(path) as s:
        z = s.read(1).astype(np.float32)
        if s.nodata is not None and not np.isnan(s.nodata):
            z[z == s.nodata] = np.nan
        text = raster_datum_text(s.tags(), s.crs)
        conv, info = to_egm2008(z, s.transform, s.crs, text)
        prof = s.profile.copy()
    prof.update(dtype="float32", nodata=np.nan, count=1)
    # drop the source vertical CRS (it would contradict the new datum); horizontal part and grid unchanged
    from pyproj import CRS
    c = CRS.from_user_input(prof["crs"].to_wkt())
    horiz = [s for s in c.sub_crs_list if not s.is_vertical]
    if horiz:
        prof["crs"] = rasterio.crs.CRS.from_wkt(horiz[0].to_wkt())
    with rasterio.open(out, "w", **prof) as d:
        d.write(conv, 1)
        d.update_tags(VERTICAL_DATUM=f"EGM2008 (converted from {info['source_datum']})", UNITS="metres",
                      DATUM_CONVERSION=json.dumps(info, default=str))
    return out, info
