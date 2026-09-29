"""Fetch the DEM window covering an image, on the DEM's own native grid.

Only the needed window is read from each Cloud-Optimized GeoTIFF (HTTP range
requests), so a 0.6 m scene costs a few MB, not 42 MB per 1x1 degree tile.

The window is snapped to the source pixel grid, so the cached mosaic holds the
DEM's exact values. No resampling happens here; putting the DEM on the image
grid is geo.reproject's explicit job.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.merge import merge
from rasterio.warp import transform_bounds

from depthwizard.geo.bounds import copernicus_tile_names

# Remote-read settings: flaky connections were seen on this machine, so retry hard.
GDAL_REMOTE_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.gz",  # .gz: skadi .hgt.gz tiles
    "GDAL_HTTP_MAX_RETRY": "6",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_TIMEOUT": "60",
}


@dataclass(frozen=True)
class DEMSource:
    name: str
    url_template: str                     # "{tile}" is replaced by the tile name
    tile_names: Callable[[tuple[float, float, float, float]], list[str]]
    horizontal_crs: str
    vertical_datum: str
    units: str
    pixel_convention: str
    reference: str

    def url(self, tile: str) -> str:
        return self.url_template.format(tile=tile)

    def describe(self) -> dict[str, str]:
        d = asdict(self)
        d.pop("tile_names")
        return d


COPERNICUS_GLO30 = DEMSource(
    name="Copernicus DEM GLO-30 (AWS Open Data)",
    url_template="https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif",
    tile_names=copernicus_tile_names,
    horizontal_crs="EPSG:4326",
    # Not present in the file metadata (tags only carry AREA_OR_POINT=Point).
    vertical_datum="EGM2008 geoid (verified: Copernicus tile XML vertical \"WGS 84 Geoid EGM08\", EPSG::1027)",
    units="metres",
    pixel_convention="AREA_OR_POINT=Point in file; GDAL exposes an edge-based transform "
                     "(first pixel centre on the whole degree)",
    reference="https://registry.opendata.aws/copernicus-dem/",
)


def skadi_tile_names(bounds_wgs84: tuple[float, float, float, float]) -> list[str]:
    """Tilezen 'skadi' 1x1 degree tiles, SRTM naming (south-west corner): N40/N40W106."""
    w, s, e, n = bounds_wgs84
    names = []
    for lat in range(math.floor(s), math.ceil(n)):
        for lon in range(math.floor(w), math.ceil(e)):
            la = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            names.append(f"{la}/{la}{'E' if lon >= 0 else 'W'}{abs(lon):03d}")
    return names


# SRTM-style alternative. NOT pure SRTM: a blend (USA mostly 3DEP/NED; elsewhere SRTM, GMTED, ...).
# Whole gzipped tiles are streamed (gzip is not range-readable): ~13-18 MB per 1x1 degree tile.
TILEZEN_SKADI = DEMSource(
    name="Tilezen/Mapzen terrain tiles, skadi 1 arc-second (AWS Open Data)",
    url_template="/vsigzip//vsicurl/https://elevation-tiles-prod.s3.amazonaws.com/skadi/{tile}.hgt.gz",
    tile_names=skadi_tile_names,
    horizontal_crs="EPSG:4326",
    vertical_datum="EGM96 geoid (per tilezen/joerd docs/formats.md: 'referenced to the WGS84/EGM96 geoid')",
    units="metres",
    pixel_convention="SRTM HGT: 3601x3601 samples, pixel-is-point (GDAL SRTMHGT driver shifts by half a pixel)",
    reference="https://registry.opendata.aws/terrain-tiles/ ; attribution required: "
              "https://github.com/tilezen/joerd/blob/master/docs/attribution.md",
)


def _copernicus_aux(kind: str):
    def names(bounds_wgs84):
        return [f"https://copernicus-dem-30m.s3.amazonaws.com/{t}/AUXFILES/{t[:-4]}_{kind}.tif"
                for t in copernicus_tile_names(bounds_wgs84)]
    return names


# Auxiliary layers on the exact DEM grid (same tiles, AUXFILES/ folder).
COPERNICUS_WBM = DEMSource(
    name="Copernicus DEM GLO-30 water body mask (WBM)", url_template="{tile}", tile_names=_copernicus_aux("WBM"),
    horizontal_crs="EPSG:4326", vertical_datum="n/a (classes: 0 land, 1 ocean, 2 lake, 3 river)",
    units="class code", pixel_convention="same grid as the DEM", reference="https://registry.opendata.aws/copernicus-dem/")
COPERNICUS_HEM = DEMSource(
    name="Copernicus DEM GLO-30 height error mask (HEM)", url_template="{tile}", tile_names=_copernicus_aux("HEM"),
    horizontal_crs="EPSG:4326",
    vertical_datum="n/a (per-pixel height error in metres; 1-sigma interpretation per product handbook - NEEDS VERIFICATION)",
    units="metres", pixel_convention="same grid as the DEM; nodata -32767 (e.g. edited water)",
    reference="https://registry.opendata.aws/copernicus-dem/")


@dataclass(frozen=True)
class DEMWindow:
    path: Path
    provenance: dict[str, Any]


def _snap_outward(lo: float, hi: float, origin: float, step: float, pad_px: int) -> tuple[float, float]:
    """Expand [lo, hi] to whole source pixels (grid defined by origin + k*step), plus padding."""
    k0 = math.floor((lo - origin) / step + 1e-9) - pad_px
    k1 = math.ceil((hi - origin) / step - 1e-9) + pad_px
    return origin + k0 * step, origin + k1 * step


def fetch_dem_window(bounds_wgs84: tuple[float, float, float, float], cache_dir: str | Path,
                     source: DEMSource = COPERNICUS_GLO30, pad_px: int = 2) -> DEMWindow:
    """Mosaic the DEM tiles covering `bounds_wgs84` (west, south, east, north), cropped to
    whole source pixels (+ `pad_px` on each side), and cache it as a float32 GeoTIFF.

    Tiles that do not exist (e.g. open ocean) are skipped and recorded; their area is NaN.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Offline reuse: look the REQUEST up before touching the network (tile grids need remote reads).
    index_path = cache_dir / "index.json"
    req_key = f"{source.name}|{pad_px}|" + "|".join(f"{v:.7f}" for v in bounds_wgs84)
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    hit = index.get(req_key)
    if hit and (cache_dir / hit).exists() and (cache_dir / hit).with_suffix(".json").exists():
        return DEMWindow(cache_dir / hit, json.loads((cache_dir / hit).with_suffix(".json").read_text()))

    def remember(path: Path) -> None:
        idx = json.loads(index_path.read_text()) if index_path.exists() else {}
        idx[req_key] = path.name
        index_path.write_text(json.dumps(idx, indent=1))

    tiles = source.tile_names(bounds_wgs84)
    if not tiles:
        raise ValueError(f"no DEM tiles for bounds {bounds_wgs84}")

    with rasterio.Env(**GDAL_REMOTE_ENV):
        datasets, used, missing = [], [], []
        for t in tiles:
            try:
                datasets.append(rasterio.open(source.url(t)))
                used.append(t)
            except RasterioIOError as exc:
                missing.append({"tile": t.split("?")[0], "error": str(exc)[:300]})
        if not datasets:
            raise RuntimeError(f"none of the DEM tiles could be opened: {missing}")
        try:
            resolutions = {ds.res for ds in datasets}
            if len(resolutions) != 1:
                # GLO-30 widens longitude spacing above 50 deg latitude; not needed for India
                raise NotImplementedError(f"mixed DEM resolutions {resolutions}")
            src_crs = datasets[0].crs
            if any(ds.crs != src_crs for ds in datasets):
                raise NotImplementedError("DEM tiles in different CRSs")
            xres, yres = datasets[0].res
            x0, y0 = datasets[0].transform.c, datasets[0].transform.f
            # snap in the SOURCE CRS (degrees for Copernicus, metres for UTM LiDAR rasters)
            w, s, e, n = (bounds_wgs84 if src_crs.is_geographic else
                          transform_bounds("EPSG:4326", src_crs, *bounds_wgs84, densify_pts=21))
            left, right = _snap_outward(w, e, x0, xres, pad_px)
            # rows grow southwards: snap in "distance below the top edge" space
            top_off, bottom_off = _snap_outward(y0 - n, y0 - s, 0.0, yres, pad_px)
            top, bottom = y0 - top_off, y0 - bottom_off

            # Padding/snapping can push the window into a neighbouring tile that the requested
            # bounds did not touch (e.g. one row south of a degree line): open those too.
            final_wgs84 = ((left, bottom, right, top) if src_crs.is_geographic else
                           transform_bounds(src_crs, "EPSG:4326", left, bottom, right, top, densify_pts=21))
            seen = {t.split("?")[0] for t in used} | {m["tile"].split("?")[0] for m in missing}
            for t in source.tile_names(final_wgs84):
                if t.split("?")[0] in seen:
                    continue
                try:
                    ds = rasterio.open(source.url(t))
                except RasterioIOError as exc:
                    missing.append({"tile": t.split("?")[0], "error": str(exc)[:300]})
                    continue
                if ds.res != (xres, yres) or ds.crs != src_crs:
                    ds.close()
                    raise NotImplementedError(f"neighbour tile {t} has a different grid/CRS")
                datasets.append(ds)
                used.append(t)

            key = hashlib.sha1(f"{source.name}|{left:.9f}|{bottom:.9f}|{right:.9f}|{top:.9f}"
                               .encode()).hexdigest()[:16]
            out = cache_dir / f"dem_{key}.tif"
            meta_path = out.with_suffix(".json")
            if out.exists() and meta_path.exists():
                remember(out)
                return DEMWindow(out, json.loads(meta_path.read_text()))

            mosaic, transform = merge(datasets, bounds=(left, bottom, right, top), res=(xres, yres),
                                      nodata=np.nan, dtype="float32", resampling=rasterio.enums.Resampling.nearest)
            src_nodata = [ds.nodata for ds in datasets]
            src_tags = datasets[0].tags()
        finally:
            for ds in datasets:
                ds.close()

    # The output grid must coincide with the source grid, otherwise "nearest" moved data.
    for got, want, step in ((transform.c, left, xres), (transform.f, top, yres)):
        if abs(got - want) > 1e-6 * step:
            raise RuntimeError(f"DEM mosaic grid is off the source grid ({got} vs {want})")

    band = mosaic[0]
    provenance = {
        "source": source.describe(),
        # query strings are stripped: signed access tokens must not land in provenance files
        "tiles_used": [{"tile": t.split("?")[0], "url": source.url(t).split("?")[0]} for t in used],
        "tiles_missing": missing,
        "requested_bounds_wgs84": list(bounds_wgs84),
        "window_bounds": [left, bottom, right, top],
        "window_crs": src_crs.to_string(),
        "pad_px": pad_px,
        "native_res_deg": [xres, yres],
        "shape": list(band.shape),
        "source_nodata_as_read": src_nodata,
        "source_crs_wkt": src_crs.to_wkt(),
        "source_tags": src_tags,
        "nan_fraction": float(np.isnan(band).mean()),
        "resampling": "none (window snapped to the source grid; values copied)",
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
    }
    profile = {"driver": "GTiff", "width": band.shape[1], "height": band.shape[0], "count": 1,
               "dtype": "float32", "crs": src_crs, "transform": transform,
               "nodata": np.nan, "compress": "deflate"}
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(band, 1)
        dst.update_tags(VERTICAL_DATUM=source.vertical_datum, UNITS=source.units,
                        DEM_SOURCE=source.name, AREA_OR_POINT=src_tags.get("AREA_OR_POINT", ""))
    meta_path.write_text(json.dumps(provenance, indent=2, default=str))
    remember(out)
    return DEMWindow(out, provenance)
