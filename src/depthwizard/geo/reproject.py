"""Put a DEM onto an image's grid.

Direction matters: the IMAGE grid is the reference and is never changed. The DEM
is resampled onto it with an explicitly named method, and that is recorded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from depthwizard.io import RasterMetadata


def dem_to_image_grid(dem_path: str | Path, meta: RasterMetadata,
                      resampling: Resampling = Resampling.bilinear) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the DEM sampled on the image grid as float32 (H, W), NaN where no DEM data.

    Bilinear is the default because a 30 m DEM on a 0.6 m grid is a ~50x upsample:
    nearest would give 30 m terraces. Neither creates detail the DEM does not have.
    """
    if not meta.is_georeferenced:
        raise ValueError("image is not georeferenced; a DEM cannot be placed on it")
    dst = np.full((meta.height, meta.width), np.nan, dtype=np.float32)
    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
            dst_transform=meta.affine, dst_crs=meta.crs, dst_nodata=np.nan,
            resampling=resampling,
        )
        info = {
            "dem_path": str(dem_path),
            "dem_crs": src.crs.to_string(),
            "dem_res": list(src.res),
            "dem_tags": src.tags(),
            "image_crs": meta.crs.to_string(),
            "image_res": [abs(meta.affine.a), abs(meta.affine.e)],
            "resampling": resampling.name,
            "direction": "DEM -> image grid (image grid unchanged)",
            "coverage_fraction": float(np.isfinite(dst).mean()),
        }
    return dst, info
