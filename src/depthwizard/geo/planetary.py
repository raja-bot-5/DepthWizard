"""Microsoft Planetary Computer helpers: STAC search + anonymous SAS signing (no account)."""
from __future__ import annotations

import json
import time
import urllib.request

from depthwizard.geo.dem import DEMSource

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"

_tokens: dict[str, tuple[str, float]] = {}


def _json(url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def token(collection: str) -> str:
    """Anonymous read token, cached for 30 min (tokens last ~45 min)."""
    tok, t = _tokens.get(collection, ("", 0.0))
    if time.time() - t > 1800:
        tok = _json(SAS.format(collection=collection))["token"]
        _tokens[collection] = (tok, time.time())
    return tok


def search(collection: str, bbox: tuple[float, float, float, float], limit: int = 100) -> list[dict]:
    return _json(f"{STAC}/search", {"collections": [collection], "bbox": list(bbox), "limit": limit})["features"]


def naip_window(lon: float, lat: float, size: int, out_dir, gsd: float = 0.6, item_id: str | None = None):
    """Pixel-window crop (RGB, native CRS/grid, no resampling) of the newest NAIP item at `gsd`
    covering (lon, lat), or of `item_id` if given. Returns (tif_path, provenance)."""
    from datetime import datetime, timezone
    from pathlib import Path

    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window

    feats = _json(f"{STAC}/search", {"collections": ["naip"], "limit": 50,
                                     "intersects": {"type": "Point", "coordinates": [lon, lat]},
                                     "sortby": [{"field": "datetime", "direction": "desc"}]})["features"]
    feats = [f for f in feats if abs(f["properties"].get("gsd", 0) - gsd) < 1e-6
             and (item_id is None or f["id"] == item_id)]
    if not feats:
        raise RuntimeError(f"no NAIP item (gsd {gsd}, id {item_id}) covers {lon},{lat}")
    item = feats[0]
    href = item["assets"]["image"]["href"]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="6", GDAL_HTTP_RETRY_DELAY="2"):
        with rasterio.open(f"{href}?{token('naip')}") as src:
            x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
            row, col = src.index(x, y)
            r0 = min(max(row - size // 2, 0), src.height - size)
            c0 = min(max(col - size // 2, 0), src.width - size)
            win = Window(c0, r0, size, size)
            data = src.read([1, 2, 3], window=win)
            profile = {"driver": "GTiff", "width": size, "height": size, "count": 3, "dtype": data.dtype,
                       "crs": src.crs, "transform": src.window_transform(win), "compress": "deflate",
                       "tiled": True, "photometric": "RGB"}
            src_info = {"crs": src.crs.to_string(), "res": list(src.res), "shape": [src.height, src.width]}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tif = out / f"naip_{item['id']}_r{r0}_c{c0}_{size}.tif"
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(SOURCE="USDA NAIP via Microsoft Planetary Computer", ITEM=item["id"],
                        WINDOW=f"row {r0} col {c0} size {size}", LICENSE="Public Domain (USDA FSA)")
    prov = {"item": item["id"], "datetime": item["properties"].get("datetime"), "href": href,
            "gsd": item["properties"].get("gsd"), "window": {"row0": r0, "col0": c0, "size": size},
            "bands_kept": ["Red", "Green", "Blue"], "source": src_info,
            "license": {"stac_field": "proprietary", "license_link": "Public Domain (USDA FSA)"},
            "resampling": "none (pixel window of the source grid)",
            "fetched_utc": datetime.now(timezone.utc).isoformat()}
    tif.with_suffix(".json").write_text(json.dumps(prov, indent=2, default=str))
    return tif, prov


def usgs_3dep(survey: str, product: str = "dsm") -> DEMSource:
    """3DEP LiDAR-derived 2 m DSM or DTM COGs, restricted to ONE named survey (e.g.
    'USGS_LPC_CO_SoPlatteRiver_Lot5_2013_LAS_2015') so tiles from other surveys/years never mix."""
    if product not in ("dsm", "dtm"):
        raise ValueError("product must be 'dsm' or 'dtm'")
    coll = f"3dep-lidar-{product}"

    def tile_names(bounds_wgs84):
        items = [f for f in search(coll, bounds_wgs84) if f["properties"].get("3dep:usgs_id") == survey]
        tok = token(coll)
        return [f"{f['assets']['data']['href']}?{tok}" for f in items]

    how = ("pdal filters.range, noise class removed" if product == "dsm"
           else "pdal filters.smrf ground classification")
    return DEMSource(
        name=f"USGS 3DEP LiDAR {product.upper()} 2 m ({survey}) via Planetary Computer",
        url_template="{tile}",
        tile_names=tile_names,
        horizontal_crs="from file (compound CRS, e.g. NAD83 / UTM 13N + NAVD88 height)",
        vertical_datum="NAVD88 height (explicit in the file's compound CRS)",
        units="metres",
        pixel_convention=f"AREA_OR_POINT=Point (file tag); {how}",
        reference=f"https://planetarycomputer.microsoft.com/dataset/{coll} ; USGS 3DEP",
    )


def usgs_3dep_dsm(survey: str) -> DEMSource:
    return usgs_3dep(survey, "dsm")


def _nasadem_tiles(bounds_wgs84):
    tok = token("nasadem")
    return [f"{f['assets']['elevation']['href']}?{tok}" for f in search("nasadem", bounds_wgs84)]


# Pure SRTM (reprocessed, void-filled) - unlike Tilezen skadi, which is 3DEP/NED over the USA.
NASADEM = DEMSource(
    name="NASADEM HGT v001 (reprocessed SRTM, 1 arc-second) via Planetary Computer",
    url_template="{tile}",
    tile_names=_nasadem_tiles,
    horizontal_crs="EPSG:4326",
    # file has no datum tag; NASADEM User Guide V1 (LP DAAC, Jan 2020) Table 1: NASADEM_HGT "meters (relative
    # to the EGM96 geoid)" (guide sha256 064ffd7356e648bf470ccebca0e3acb6d03578f60ba4256d89e229cff2b970de)
    vertical_datum="EGM96 geoid (verified: NASADEM User Guide V1, Table 1, NASADEM_HGT)",
    units="metres",
    pixel_convention="AREA_OR_POINT=Point (file tag); 3601x3601 per 1x1 degree, first pixel centre on the degree",
    reference="https://planetarycomputer.microsoft.com/dataset/nasadem ; NASA LP DAAC "
              "(https://lpdaac.usgs.gov/data/data-citation-and-policies/: no restrictions on reuse)",
)


ESA_WORLDCOVER_CLASSES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up",
                          60: "bare / sparse vegetation", 70: "snow and ice", 80: "permanent water",
                          90: "herbaceous wetland", 95: "mangroves", 100: "moss and lichen"}


def esa_worldcover(year: int = 2021) -> DEMSource:
    """ESA WorldCover 10 m land-cover classes (CC-BY-4.0). Served through the DEM window machinery
    (float32 class codes, nearest, source grid) - it is a label raster, not heights."""
    ver = {2020: "v100", 2021: "v200"}[year]

    def tile_names(bounds_wgs84):
        items = [f for f in search("esa-worldcover", bounds_wgs84) if f"_{year}_{ver}_" in f["id"]]
        tok = token("esa-worldcover")
        return [f"{f['assets']['map']['href']}?{tok}" for f in items]

    return DEMSource(
        name=f"ESA WorldCover 10 m {year} {ver} via Planetary Computer", url_template="{tile}",
        tile_names=tile_names, horizontal_crs="EPSG:4326", vertical_datum="n/a (land-cover classes)",
        units="class code", pixel_convention="area",
        reference="https://planetarycomputer.microsoft.com/dataset/esa-worldcover ; CC-BY-4.0, ESA WorldCover project")
