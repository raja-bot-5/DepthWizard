"""Tiled inference for scenes larger than one model pass (Cartosat scenes are ~10k px).

Monocular depth is only defined up to an affine map per inference, so two tiles
disagree by scale and shift, not just noise. Two alignment modes, then blending with
weights that fade towards the tile edges:

  "sequential" (align=True): each new tile is least-squares fitted (a*pred + b) to the
      mosaic built so far. Regression dilution shrinks later tiles (Phase 11 T3: fine
      detail kept at 0.14-0.78 of tile 0's) and a per-tile tilt is not modelled -> seams.
  "joint": all tiles at once. Scales from the ratio of robust high-pass spreads on each
      shared overlap, solved in log space (unbiased, no shrinkage); then offset + plane
      (b + c*x + d*y) per tile by least squares over all overlaps. Tile 0 is the reference.
      Phase 11 T3 (3 NAIP sites): seam step at tile edges 1.06-1.74 x control p95
      (was 1.08-28.5), fine detail kept 0.68-0.84.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
from affine import Affine
from scipy.ndimage import uniform_filter


@dataclass(frozen=True)
class Tile:
    row0: int
    col0: int
    height: int
    width: int

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.row0, self.row0 + self.height), slice(self.col0, self.col0 + self.width)

    def transform(self, parent: Affine) -> Affine:
        """Geotransform of this tile: the parent's, shifted to the tile's upper-left pixel.
        Same pixel size, same CRS; a tile is a crop, never a resample."""
        return parent * Affine.translation(self.col0, self.row0)


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)  # last tile flush with the edge, never padded
    return sorted(set(starts))


def plan_tiles(height: int, width: int, tile: int, overlap: int) -> list[Tile]:
    """Row-major tiles of at most `tile` px that cover the image, with >= `overlap` px overlap."""
    if not 0 <= overlap < tile:
        raise ValueError("need 0 <= overlap < tile")
    stride = tile - overlap
    return [Tile(r, c, min(tile, height), min(tile, width))
            for r in _starts(height, tile, stride) for c in _starts(width, tile, stride)]


def blend_weights(height: int, width: int, overlap: int) -> np.ndarray:
    """Per-pixel blending weight for one tile, shape (height, width), float32.

    Each output pixel = sum(w_i * pred_i) / sum(w_i) over the tiles covering it.
    Cosine (Hann) ramp over the `overlap` pixels at each edge, 1 in the interior: seams
    fade smoothly and the slope is smooth too, so shaded 3D renders show no creases.
    Never 0, because outer-border pixels have only one tile (0/0 would be NaN).
    """
    def ramp(n: int) -> np.ndarray:
        r = max(1, min(overlap, n // 2))            # tiles narrower than 2*overlap
        d = np.minimum(np.arange(n), n - 1 - np.arange(n))  # distance to nearest edge
        w = 0.5 - 0.5 * np.cos(np.pi * (d + 1) / (r + 1))   # d=0 -> small but > 0
        return np.where(d < r, w, 1.0)

    return np.minimum.outer(ramp(height), ramp(width)).astype(np.float32)


def fit_affine(src: np.ndarray, ref: np.ndarray, min_pixels: int = 64) -> tuple[float, float]:
    """Least-squares (a, b) minimising |a*src + b - ref| over finite pixels."""
    m = np.isfinite(src) & np.isfinite(ref)
    if m.sum() < min_pixels or np.ptp(src[m]) == 0:
        return 1.0, 0.0
    A = np.stack([src[m], np.ones(m.sum(), dtype=src.dtype)], axis=1).astype(np.float64)
    (a, b), *_ = np.linalg.lstsq(A, ref[m].astype(np.float64), rcond=None)
    return float(a), float(b)


def _overlaps(tiles: list[Tile]) -> list[tuple[int, int, int, int, int, int]]:
    """(i, j, r0, r1, c0, c1) for every pair of tiles that share pixels."""
    out = []
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            a, b = tiles[i], tiles[j]
            r0, r1 = max(a.row0, b.row0), min(a.row0 + a.height, b.row0 + b.height)
            c0, c1 = max(a.col0, b.col0), min(a.col0 + a.width, b.col0 + b.width)
            if r1 > r0 and c1 > c0:
                out.append((i, j, r0, r1, c0, c1))
    return out


def _crop(pred: np.ndarray, t: Tile, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    return pred[r0 - t.row0:r1 - t.row0, c0 - t.col0:c1 - t.col0]


def _robust_spread(v: np.ndarray) -> float:
    return float(np.median(np.abs(v - np.median(v)))) * 1.4826


def joint_align(preds: list[np.ndarray], tiles: list[Tile], H: int, W: int, plane: bool = True, hp_px: int = 51,
                min_strip_px: int = 100, step: int = 4) -> np.ndarray:
    """Per-tile (a, b, c, d) so that a*pred + b + c*col/W + d*row/H agree on all overlaps. Tile 0 = (1, 0, 0, 0).
    plane=False fits offsets only (c = d = 0).

    Scales: log a_i - log a_j = log(spread_j / spread_i), spreads = robust std of the high-pass
    (pred - box mean over hp_px) on the shared overlap, strips thinner than min_strip_px skipped.
    Offsets/planes: normal equations accumulated pair by pair (memory O(n_tiles^2), fine for
    Cartosat-size scenes), tiny ridge so a tile without usable overlaps stays at (a, 0, 0, 0).
    """
    n = len(tiles)
    pairs = _overlaps(tiles)
    trim = hp_px // 2
    rows, rhs = [], []
    for i, j, r0, r1, c0, c1 in pairs:
        if min(r1 - r0, c1 - c0) < min_strip_px:
            continue
        sp = []
        for t in (i, j):
            v = _crop(preds[t], tiles[t], r0, r1, c0, c1).astype(np.float64)
            sp.append(_robust_spread((v - uniform_filter(v, hp_px))[trim:-trim, trim:-trim]))
        if min(sp) <= 0 or not np.isfinite(sp).all():
            continue
        e = np.zeros(n)
        e[i], e[j] = 1.0, -1.0
        rows.append(e)
        rhs.append(np.log(sp[1] / sp[0]))
    log_a = np.zeros(n)
    if rows:
        log_a[1:] = np.linalg.lstsq(np.array(rows)[:, 1:], np.array(rhs), rcond=None)[0]
    a = np.exp(log_a)

    k = 3 if plane else 1                                   # b[, c, d] per tile
    N = np.zeros((n * k, n * k))
    g = np.zeros(n * k)
    for i, j, r0, r1, c0, c1 in pairs:
        pi = _crop(preds[i], tiles[i], r0, r1, c0, c1)[::step, ::step].astype(np.float64).ravel() * a[i]
        pj = _crop(preds[j], tiles[j], r0, r1, c0, c1)[::step, ::step].astype(np.float64).ravel() * a[j]
        rr, cc = np.mgrid[r0:r1:step, c0:c1:step]
        X = np.stack([np.ones(pi.size), cc.ravel() / W, rr.ravel() / H], axis=1)[:, :k]
        A = np.zeros((pi.size, n * k))
        A[:, i * k:(i + 1) * k], A[:, j * k:(j + 1) * k] = X, -X
        w = 1.0 / pi.size                                   # every overlap counts equally
        N += w * (A.T @ A)
        g += w * (A.T @ (pj - pi))
    sub = N[k:, k:] + np.eye((n - 1) * k) * 1e-9 * max(np.trace(N), 1.0)
    bcd = np.zeros((n, k))
    if n > 1:
        bcd[1:] = np.linalg.solve(sub, g[k:]).reshape(n - 1, k)
    if not plane:
        bcd = np.column_stack([bcd, np.zeros((n, 2))])
    return np.column_stack([a, bcd])


def run_tiled(image: np.ndarray, predict: Callable[[np.ndarray], np.ndarray], tile: int,
              overlap: int, align: bool | str = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Run `predict` (HxWxC crop -> HxW float map of the same size) over tiles and merge.

    align: True / "sequential" (fit each tile to the mosaic so far), "joint" (scale + offset, see joint_align),
    "joint_plane" (scale + offset + plane), or False.
    Returns the float32 (H, W) mosaic and a record of every tile's window and fitted parameters.
    """
    mode = {True: "sequential", False: "none"}.get(align, align)
    if mode not in ("sequential", "joint", "joint_plane", "none"):
        raise ValueError("align must be True/'sequential', 'joint', 'joint_plane' or False")
    H, W = image.shape[:2]
    tiles = plan_tiles(H, W, tile, overlap)
    acc = np.zeros((H, W), dtype=np.float64)
    wsum = np.zeros((H, W), dtype=np.float64)
    records = []

    def predict_tile(t: Tile) -> np.ndarray:
        pred = np.asarray(predict(image[t.slices]), dtype=np.float32)
        if pred.shape != (t.height, t.width):
            raise ValueError(f"predict returned {pred.shape} for a {(t.height, t.width)} tile; "
                             "resample the model output back to the tile size first")
        return pred

    def add(t: Tile, pred: np.ndarray) -> None:
        rs, cs = t.slices
        w = blend_weights(t.height, t.width, overlap).astype(np.float64)
        if w.shape != pred.shape or not np.all(w > 0):
            raise ValueError("blend_weights must be strictly positive and match the tile shape")
        acc[rs, cs] += w * pred
        wsum[rs, cs] += w

    if mode in ("joint", "joint_plane"):
        preds = [predict_tile(t) for t in tiles]            # float32; ~4.3 MB per 1036 px tile
        params = joint_align(preds, tiles, H, W, plane=mode == "joint_plane")
        for t, p, (a, b, c, d) in zip(tiles, preds, params):
            rr, cc = np.mgrid[t.row0:t.row0 + t.height, t.col0:t.col0 + t.width]
            add(t, a * p.astype(np.float64) + b + c * cc / W + d * rr / H)
            records.append({**asdict(t), "a": float(a), "b": float(b), "c_per_width": float(c),
                            "d_per_height": float(d)})
    else:
        for t in tiles:
            pred = predict_tile(t).astype(np.float64)
            a, b = 1.0, 0.0
            rs, cs = t.slices
            seen = wsum[rs, cs] > 0
            if mode == "sequential" and seen.any():
                current = np.where(seen, acc[rs, cs] / np.where(seen, wsum[rs, cs], 1.0), np.nan)
                a, b = fit_affine(pred, current)
                pred = a * pred + b
            add(t, pred)
            records.append({**asdict(t), "a": a, "b": b, "overlap_px": int(seen.sum())})
    mosaic = (acc / wsum).astype(np.float32)
    return mosaic, {"tile": tile, "overlap": overlap, "align": mode, "n_tiles": len(tiles),
                    "tiles": records}
