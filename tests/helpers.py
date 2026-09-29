"""Synthetic georeferenced rasters for tests (no network, no real data)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

# Near Dehradun, UTM 44N. 0.6 m pixels like Cartosat-2S.
UTM44N = "EPSG:32644"
DEHRADUN_UTM = (214_400.0, 3_358_000.0)  # upper-left corner, ~78.03E 30.32N (zone 44 CM is 81E)


def write_utm_image(path: Path, width: int = 200, height: int = 150, res: float = 0.6,
                    origin: tuple[float, float] = DEHRADUN_UTM, count: int = 3) -> Path:
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(count, height, width), dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=count,
                       dtype="uint8", crs=UTM44N, transform=from_origin(*origin, res, res)) as dst:
        dst.write(data)
    return path


def plane(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """A tilted plane in lon/lat; bilinear resampling reproduces it exactly."""
    return 100.0 + 1000.0 * (lon - 78.0) + 500.0 * (lat - 30.0)


def write_fake_tile(path: Path, lat: int, lon: int, px_per_deg: int = 36) -> Path:
    """A 1x1 degree 'Copernicus-like' tile: pixel centres on k/px_per_deg degrees, so the
    edge-based grid starts half a pixel west/north of the whole degree."""
    res = 1.0 / px_per_deg
    left, top = lon - res / 2, lat + 1 + res / 2
    cols = left + (np.arange(px_per_deg) + 0.5) * res
    rows = top - (np.arange(px_per_deg) + 0.5) * res
    lon_g, lat_g = np.meshgrid(cols, rows)
    data = plane(lon_g, lat_g).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", width=px_per_deg, height=px_per_deg, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(left, top, res, res)) as dst:
        dst.write(data, 1)
    return path
