"""Image footprint in WGS84 and the DEM tiles that cover it."""
from __future__ import annotations

import math

from rasterio.warp import transform_bounds

from depthwizard.io import RasterMetadata

WGS84 = "EPSG:4326"


def footprint_wgs84(meta: RasterMetadata, margin_deg: float = 0.0,
                    densify_pts: int = 21) -> tuple[float, float, float, float]:
    """(west, south, east, north) in EPSG:4326 covering the image's pixel edges.

    Edges are densified before transforming so a UTM footprint's curved edges are
    fully covered. `margin_deg` pads every side (use >= one DEM pixel so bilinear
    resampling at the image border has neighbours).
    """
    w, s, e, n = transform_bounds(meta.crs, WGS84, *meta.bounds, densify_pts=densify_pts)
    if e < w:
        raise NotImplementedError("footprint crosses the antimeridian")
    return (w - margin_deg, max(s - margin_deg, -90.0), e + margin_deg, min(n + margin_deg, 90.0))


def _hemi(value: int, pos: str, neg: str, width: int) -> str:
    return f"{pos if value >= 0 else neg}{abs(value):0{width}d}"


def copernicus_tile_names(bounds_wgs84: tuple[float, float, float, float]) -> list[str]:
    """Copernicus GLO-30 1x1 degree tile names intersecting the bounds.

    A tile is named by its south-west corner, e.g. N28_00_E077_00 covers
    lat 28..29, lon 77..78; S01_00_W072_00 covers lat -1..0, lon -72..-71.
    """
    w, s, e, n = bounds_wgs84
    names = []
    # ceil() keeps a north/east bound lying exactly on a degree line from pulling in the next tile
    for lat in range(math.floor(s), math.ceil(n)):
        for lon in range(math.floor(w), math.ceil(e)):
            names.append(f"Copernicus_DSM_COG_10_{_hemi(lat, 'N', 'S', 2)}_00_"
                         f"{_hemi(lon, 'E', 'W', 3)}_00_DEM")
    return names
