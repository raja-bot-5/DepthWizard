"""Height grid -> textured mesh (GLB), plus a crude ground estimate for derived heights.

Mesh coordinates are LOCAL: x east, y north, in metres from the image's upper-left corner
(metric DSM) or in pixel units (relative DSM); z = height minus z_offset. The offsets and
units are returned so a viewer can show true values. The mesh is a display product;
measurements should query the DSM GeoTIFF, not the decimated mesh.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def _block_mean(z: np.ndarray, f: int) -> np.ndarray:
    """NaN-aware f x f block mean (crops the ragged edge)."""
    h, w = (z.shape[0] // f) * f, (z.shape[1] // f) * f
    b = z[:h, :w].reshape(h // f, f, w // f, f)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3)) if np.isnan(b).any() else b.mean(axis=(1, 3))


def grid_mesh(z: np.ndarray, pixel_size: tuple[float, float], max_side: int = 512
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Regular-grid triangle mesh. Returns (vertices, faces, uv, info). Quads touching NaN are dropped."""
    f = max(1, int(np.ceil(max(z.shape) / max_side)))
    zz = _block_mean(z.astype(np.float64), f) if f > 1 else z.astype(np.float64)
    rows, cols = zz.shape
    finite = np.isfinite(zz)
    if not finite.any():
        raise ValueError("height grid is all NaN")
    z_off = float(np.nanmin(zz))
    r, c = np.mgrid[0:rows, 0:cols]
    sx, sy = pixel_size[0] * f, pixel_size[1] * f
    verts = np.stack([(c + 0.5) * sx, -(r + 0.5) * sy, np.where(finite, zz - z_off, 0.0)], -1).reshape(-1, 3)
    uv = np.stack([(c + 0.5) / cols, 1.0 - (r + 0.5) / rows], -1).reshape(-1, 2)
    idx = np.arange(rows * cols).reshape(rows, cols)
    a, b, cc, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    ok = finite[:-1, :-1] & finite[:-1, 1:] & finite[1:, :-1] & finite[1:, 1:]
    faces = np.concatenate([np.stack([a[ok], cc[ok], b[ok]], -1), np.stack([b[ok], cc[ok], d[ok]], -1)])
    info = {"decimation": f, "grid": [rows, cols], "vertex_spacing": [sx, sy], "z_offset": z_off,
            "n_vertices": int(verts.shape[0]), "n_faces": int(faces.shape[0])}
    return verts.astype(np.float32), faces.astype(np.int64), uv.astype(np.float32), info


def export_glb(path, z: np.ndarray, rgb: np.ndarray, pixel_size: tuple[float, float],
               max_side: int = 512, texture_max: int = 2048) -> dict[str, Any]:
    import trimesh
    verts, faces, uv, info = grid_mesh(z, pixel_size, max_side)
    tex = Image.fromarray(rgb)
    if max(tex.size) > texture_max:
        tex.thumbnail((texture_max, texture_max), Image.LANCZOS)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, image=tex))
    mesh.export(path)
    info["texture_size"] = list(tex.size)
    return info


def ground_estimate(dsm: np.ndarray, pixel_m: float, window_m: float = 30.0) -> np.ndarray:
    """Crude DERIVED ground: grey morphological opening (min then max filter) over window_m.
    Removes objects narrower than the window (most buildings); large canopies/blocks survive.
    THIS SHOULD BE TESTED against a LiDAR DTM before trusting derived heights."""
    k = max(3, int(round(window_m / pixel_m)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    filled = np.where(np.isfinite(dsm), dsm, np.nanmax(dsm)).astype(np.float32)
    opened = cv2.dilate(cv2.erode(filled, kernel), kernel)
    return np.where(np.isfinite(dsm), np.minimum(opened, filled), np.nan).astype(np.float32)
