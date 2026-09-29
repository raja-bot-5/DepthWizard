"""Raster I/O that never loses georeferencing.

- PNG/JPG -> not georeferenced; results are relative (rDSM), never metres.
- GeoTIFF -> CRS + affine transform are kept exactly as read. Nothing here
  resamples, reprojects or resizes; that is geo.reproject's job and it is explicit.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.errors import NotGeoreferencedWarning

GEOTIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class RasterMetadata:
    width: int
    height: int
    count: int
    dtype: str
    crs_wkt: str | None                 # None -> not georeferenced
    transform: tuple[float, ...] | None  # affine (a, b, c, d, e, f); None -> not georeferenced
    nodata: float | None
    area_or_point: str | None           # GDAL AREA_OR_POINT tag as read
    source: str
    tags: dict[str, str] = field(default_factory=dict)
    colorinterp: tuple[str, ...] = ()   # per band, as declared in the file ("red", "gray", "undefined", ...)
    descriptions: tuple[str | None, ...] = ()
    nbits: int | None = None            # GDAL IMAGE_STRUCTURE NBITS (e.g. 11 for an 11-bit sensor), if declared
    georef_issue: str | None = None     # why a GeoTIFF is NOT treated as georeferenced (RPC-only, rotated, ...)

    @property
    def is_georeferenced(self) -> bool:
        return self.crs_wkt is not None and self.transform is not None

    @property
    def affine(self) -> Affine:
        if self.transform is None:
            raise ValueError(f"{self.source} is not georeferenced")
        return Affine(*self.transform[:6])

    @property
    def crs(self) -> rasterio.crs.CRS:
        if self.crs_wkt is None:
            raise ValueError(f"{self.source} has no CRS")
        return rasterio.crs.CRS.from_wkt(self.crs_wkt)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) of the pixel *edges* in the raster's CRS."""
        return rasterio.transform.array_bounds(self.height, self.width, self.affine)

    def pixel_to_geo(self, row: float, col: float) -> tuple[float, float]:
        """Pixel (row, col) -> CRS (x, y). Integer row/col is the pixel's upper-left corner;
        add 0.5 for the centre."""
        x, y = self.affine * (col, row)
        return x, y

    def geo_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        col, row = ~self.affine * (x, y)
        return row, col

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def read_raster(path: str | Path) -> tuple[np.ndarray, RasterMetadata]:
    """Read a raster as (bands, rows, cols) in its native dtype, plus metadata.

    Georeferencing is taken only from GeoTIFFs. A PNG/JPG with a world file is still
    treated as not georeferenced, so an rDSM can never be mislabelled as metric.
    """
    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            data = src.read()
            is_tif = path.suffix.lower() in GEOTIFF_SUFFIXES
            georef = is_tif and src.crs is not None
            issue = None
            if is_tif and src.crs is None and (src.rpcs is not None or src.gcps[0]):
                # e.g. a Cartosat Level-1 product: sensor geometry + RPCs, NOT orthorectified. A DSM on this grid
                # cannot be placed on a map without orthorectification (which needs a DEM and changes the grid).
                issue = ("image is not orthorectified: it has " + ("RPCs" if src.rpcs is not None else "GCPs") +
                         " but no map grid. Output is RELATIVE only. Orthorectify it first (e.g. gdalwarp -rpc with "
                         "a DEM) to get a metric DSM.")
            elif georef and (src.transform.b != 0 or src.transform.d != 0):
                georef, issue = False, ("rotated/sheared geotransform: only north-up grids are supported for metric "
                                        "output. Warp it to a north-up grid first. Output is RELATIVE only.")
            nb = src.tags(ns="IMAGE_STRUCTURE").get("NBITS")
            meta = RasterMetadata(
                width=src.width, height=src.height, count=src.count, dtype=src.dtypes[0],
                crs_wkt=src.crs.to_wkt() if georef else None,
                transform=tuple(src.transform)[:6] if georef else None,
                nodata=src.nodata,
                area_or_point=src.tags().get("AREA_OR_POINT"),
                source=str(path), tags=dict(src.tags()),
                colorinterp=tuple(c.name for c in src.colorinterp), descriptions=tuple(src.descriptions),
                nbits=int(nb) if nb else None, georef_issue=issue,
            )
    return data, meta


def write_dsm(path: str | Path, dsm: np.ndarray, meta: RasterMetadata, *, kind: str,
              units: str, vertical_datum: str, provenance: dict[str, Any] | None = None,
              extra_tags: dict[str, str] | None = None) -> Path:
    """Write a single-band float32 DSM on exactly the grid described by `meta`.

    kind: "absolute" (metric heights) or "relative" (rDSM, unitless).
    NaN is the nodata value. Datum/units/provenance are written as GeoTIFF tags so
    the file is self-describing.
    """
    if kind not in {"absolute", "relative"}:
        raise ValueError("kind must be 'absolute' or 'relative'")
    if kind == "relative" and units.lower() in {"m", "metre", "metres", "meter", "meters"}:
        raise ValueError("a relative DSM cannot be labelled in metres")
    if dsm.shape != (meta.height, meta.width):
        raise ValueError(f"DSM shape {dsm.shape} does not match grid {(meta.height, meta.width)}; "
                         "refusing to write it with that grid's transform")
    profile = {"driver": "GTiff", "width": meta.width, "height": meta.height, "count": 1,
               "dtype": "float32", "nodata": np.nan, "compress": "deflate", "tiled": True,
               "blockxsize": 256, "blockysize": 256}
    if meta.is_georeferenced:
        profile.update(crs=meta.crs, transform=meta.affine)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(dsm.astype(np.float32), 1)
            dst.update_tags(DSM_KIND=kind, UNITS=units, VERTICAL_DATUM=vertical_datum,
                            PROVENANCE=json.dumps(provenance or {}, default=str), **(extra_tags or {}))
            if meta.area_or_point:
                dst.update_tags(AREA_OR_POINT=meta.area_or_point)
    return path


def downsample_geotiff(src_path: str | Path, factor: int, out_path: str | Path) -> Path:
    """Coarser-GSD copy of a GeoTIFF by exact f x f block mean. The transform is SCALED by f
    (same origin, f x pixel size) - never the old transform. Ragged edge rows/cols are cropped."""
    if factor < 1 or int(factor) != factor:
        raise ValueError("factor must be a positive integer")
    with rasterio.open(src_path) as src:
        h, w = (src.height // factor) * factor, (src.width // factor) * factor
        a = src.read(window=((0, h), (0, w))).astype(np.float64)
        b = a.reshape(src.count, h // factor, factor, w // factor, factor).mean(axis=(2, 4))
        prof = src.profile.copy()
        prof.update(width=w // factor, height=h // factor, transform=src.transform * Affine.scale(factor),
                    blockxsize=min(256, w // factor), blockysize=min(256, h // factor))
        dtype = src.dtypes[0]
        tags = src.tags()
    out = np.rint(b).astype(dtype) if np.issubdtype(np.dtype(dtype), np.integer) else b.astype(dtype)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out)
        dst.update_tags(**tags, DOWNSAMPLED=f"block mean x{factor} from {Path(src_path).name}")
    return Path(out_path)
