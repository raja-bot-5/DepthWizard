"""Shadow-based building heights (Phase 11 T5 prototype): height = shadow length x tan(sun elevation).

Inputs: an RGB image on a metric grid, building footprints rasterised on that grid (e.g. OpenStreetMap), and the
sun position. Sun position comes from image metadata (Cartosat products carry sun elevation/azimuth) or from the
acquisition date+time and location via solar_position() (NOAA general solar position equations). When the time of
day is not trustworthy, the shadow direction can be MEASURED from the image (shadow_azimuth_from_footprints) and the
time solved for on the known date (time_for_azimuth).

Assumptions (each one a failure mode, reported per building):
  - flat ground around the building (shadow on a slope is stretched / shortened)
  - the shadow falls on open ground, not on another building or tree (then it is truncated or merged)
  - the wall base is at the footprint edge. In a non-true ortho image, tall roofs lean and can hide part of the shadow
  - shadow pixels are the darkest non-vegetation class of the scene (3-class Otsu); water and dark roofs also are
Everything produced here is labelled DERIVED.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np


# ------------------------------------------------------------------ sun position (NOAA)
def _julian_day(t: datetime) -> float:
    t = t.astimezone(timezone.utc)
    return t.timestamp() / 86400.0 + 2440587.5


def solar_position(lon: float, lat: float, t: datetime) -> tuple[float, float]:
    """(elevation_deg, azimuth_deg clockwise from north) of the sun centre; no refraction. t must be tz-aware.

    NOAA General Solar Position calculations (the NOAA solar calculator spreadsheet); stated accuracy ~0.01 deg
    for years 1800-2100 (tested here against physical identities, see tests/test_shadows.py)."""
    if t.tzinfo is None:
        raise ValueError("t must be timezone-aware (UTC)")
    jc = (_julian_day(t) - 2451545.0) / 36525.0
    L0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    Mr = math.radians(M)
    C = (math.sin(Mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + math.sin(2 * Mr) * (0.019993 - 0.000101 * jc)
         + math.sin(3 * Mr) * 0.000289)
    omega = math.radians(125.04 - 1934.136 * jc)
    app_long = L0 + C - 0.00569 - 0.00478 * math.sin(omega)
    obliq = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obliq_c = math.radians(obliq + 0.00256 * math.cos(omega))
    decl = math.asin(math.sin(obliq_c) * math.sin(math.radians(app_long)))
    y = math.tan(obliq_c / 2) ** 2
    L0r = math.radians(L0)
    eq_time = 4 * math.degrees(y * math.sin(2 * L0r) - 2 * e * math.sin(Mr) + 4 * e * y * math.sin(Mr) * math.cos(2 * L0r)
                               - 0.5 * y * y * math.sin(4 * L0r) - 1.25 * e * e * math.sin(2 * Mr))   # minutes
    tu = t.astimezone(timezone.utc)
    minutes = tu.hour * 60 + tu.minute + tu.second / 60 + tu.microsecond / 6e7
    tst = (minutes + eq_time + 4 * lon) % 1440
    ha = math.radians(tst / 4 - 180 if tst / 4 >= 0 else tst / 4 + 180)
    latr = math.radians(lat)
    cos_z = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_z)))
    denom = math.cos(latr) * math.sin(zen)
    if abs(denom) < 1e-12:
        az = 180.0
    else:
        a = math.degrees(math.acos(max(-1.0, min(1.0, (math.sin(latr) * math.cos(zen) - math.sin(decl)) / denom))))
        az = (a + 180) % 360 if ha > 0 else (540 - a) % 360
    return 90.0 - math.degrees(zen), az


def time_for_azimuth(lon: float, lat: float, day_utc: datetime, azimuth_deg: float) -> tuple[datetime, float]:
    """Daytime instant on `day_utc`'s date (UTC) whose sun azimuth is closest to `azimuth_deg` (1-minute search).
    Returns (time, elevation_deg). Azimuth grows monotonically through the day at mid latitudes, so the answer
    is unique while the sun is up."""
    d0 = datetime(day_utc.year, day_utc.month, day_utc.day, tzinfo=timezone.utc) - timedelta(hours=12)
    best = None
    for m in range(0, 48 * 60):                              # +-24 h around the UTC date covers any longitude
        t = d0 + timedelta(minutes=m)
        el, az = solar_position(lon, lat, t)
        if el <= 5:
            continue
        if (t + timedelta(hours=lon / 15)).date() != day_utc.date():   # keep the LOCAL calendar day
            continue
        d = abs((az - azimuth_deg + 180) % 360 - 180)
        if best is None or d < best[0]:
            best = (d, t, el)
    if best is None:
        raise ValueError("sun never above 5 deg on that date")
    return best[1], best[2]


# ------------------------------------------------------------------ shadows in the image
def _otsu3(lum: np.ndarray) -> tuple[int, int]:
    """Two thresholds maximising between-class variance of a 3-class split of a uint8 image."""
    p = np.bincount(lum.ravel(), minlength=256).astype(np.float64)
    p /= p.sum()
    i = np.arange(256)
    P, S = np.cumsum(p), np.cumsum(p * i)                     # class weight / first moment prefix sums
    best, t = -1.0, (85, 170)
    for t1 in range(1, 254):
        for t2 in range(t1 + 1, 255):
            v = 0.0
            for w, m in ((P[t1 - 1], S[t1 - 1]), (P[t2 - 1] - P[t1 - 1], S[t2 - 1] - S[t1 - 1]),
                         (1 - P[t2 - 1], S[-1] - S[t2 - 1])):
                if w > 0:
                    v += m * m / w
            if v > best:
                best, t = v, (t1, t2)
    return t


def shadow_mask(rgb: np.ndarray, method: str = "otsu3_noveg", min_area_px: int = 6,
                max_exg: float = 0.05) -> tuple[np.ndarray, float]:
    """Shadow candidates. Returns (mask, luminance threshold).

    otsu3_noveg (default): darkest class of a 3-class Otsu split of luminance, minus vegetation (excess-green
        index (2G-R-B)/(R+G+B) >= max_exg). Chosen at site A by a LABEL-FREE criterion (sharpness of the
        footprint-shift azimuth peak, 1.80), not by LiDAR: a 2-class Otsu marked 58 % of the scene (trees, lawns,
        asphalt) and made shadows 16 m too long. Known failure: shadow falling on grass can be removed.
    otsu2: darkest class of a 2-class Otsu (kept for comparison).
    """
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    if method == "otsu2":
        thr, m = cv2.threshold(lum, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "otsu3_noveg":
        thr, _ = _otsu3(lum)
        f = rgb.astype(np.float32)
        exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / (f.sum(-1) + 1e-6)
        m = ((lum < thr) & (exg < max_exg)).astype(np.uint8)
    else:
        raise ValueError("method must be otsu3_noveg or otsu2")
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
    return keep[lab], float(thr)


def _direction(az_deg: float) -> tuple[float, float]:
    """Unit step (drow, dcol) on a north-up grid pointing TOWARDS azimuth az (clockwise from north)."""
    a = math.radians(az_deg)
    return -math.cos(a), math.sin(a)


def shadow_azimuth_from_footprints(shadow: np.ndarray, footprints: np.ndarray, gsd_m: float,
                                   dist_m: tuple[float, float] = (1.5, 6.0), step_deg: float = 1.0) -> dict:
    """Measure the direction shadows extend from buildings: the azimuth d maximising the shadow fraction in the
    footprints shifted by 1.5-6 m along d (outside any footprint). Sun azimuth = d - 180."""
    bld = footprints > 0
    free = ~bld
    H, W = bld.shape
    scores = []
    for d in np.arange(0, 360, step_deg):
        dr, dc = _direction(d)
        hit = tot = 0
        for dist in np.linspace(dist_m[0], dist_m[1], 4):
            sr, sc = int(round(dr * dist / gsd_m)), int(round(dc * dist / gsd_m))
            M = np.float32([[1, 0, sc], [0, 1, sr]])
            shifted = cv2.warpAffine(bld.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
            band = shifted & free
            hit += int((band & shadow).sum())
            tot += int(band.sum())
        scores.append(hit / max(tot, 1))
    scores = np.array(scores)
    d_best = float(np.arange(0, 360, step_deg)[scores.argmax()])
    return {"shadow_direction_deg": d_best, "sun_azimuth_deg": (d_best + 180) % 360,
            "shadow_fraction_best": float(scores.max()), "shadow_fraction_median": float(np.median(scores)),
            "contrast": float(scores.max() / max(np.median(scores), 1e-9))}


@dataclass
class ShadowHeight:
    height_m: float | None
    length_m: float | None
    n_rays: int
    n_rays_in_shadow: int
    reasons: list[str]


def shadow_heights(shadow: np.ndarray, footprints: np.ndarray, gsd_m: float, sun_az_deg: float,
                   sun_el_deg: float, max_len_m: float = 80.0, min_rays: int = 5,
                   blocked: np.ndarray | None = None) -> dict[int, ShadowHeight]:
    """Per building: march from each footprint-edge pixel on the shadow side along the shadow direction and count
    the contiguous shadow run. Length = 75th percentile of the runs of rays that start in shadow (roof lean and
    partial occlusion shorten runs; the upper quartile is closer to the full length). height = length * tan(el).

    A run that reaches another building (or `blocked`, e.g. water) or max_len_m is cut there and the building
    is flagged (shadow may be truncated)."""
    d = (sun_az_deg + 180) % 360
    dr, dc = _direction(d)
    H, W = footprints.shape
    tan_el = math.tan(math.radians(sun_el_deg))
    nmax = int(max_len_m / gsd_m)
    other = footprints > 0
    if blocked is not None:
        other = other | blocked
    out = {}
    k3 = np.ones((3, 3), np.uint8)
    ids = np.unique(footprints[footprints > 0])
    for k in ids:
        m = (footprints == k).astype(np.uint8)
        edge = (m > 0) & ~(cv2.erode(m, k3) > 0)
        rr, cc = np.nonzero(edge)
        # shadow-side edge: the next pixel along d is outside this footprint
        nr = np.clip(np.round(rr + dr * 1.5).astype(int), 0, H - 1)
        nc = np.clip(np.round(cc + dc * 1.5).astype(int), 0, W - 1)
        side = footprints[nr, nc] != k
        rr, cc = rr[side], cc[side]
        runs, truncated = [], 0
        for r0, c0 in zip(rr, cc):
            n = 0
            for s in range(1, nmax + 1):
                r, c = int(round(r0 + dr * s)), int(round(c0 + dc * s))
                if not (0 <= r < H and 0 <= c < W):
                    truncated += 1
                    break
                if footprints[r, c] == k:
                    continue                                   # still inside own footprint (concave shape)
                if other[r, c] and footprints[r, c] != k:
                    if n > 0:
                        truncated += 1
                    break
                if not shadow[r, c]:
                    break
                n = s
            if n > 0:
                runs.append(n)
        reasons = []
        if len(runs) < min_rays:
            reasons.append(f"only {len(runs)} shadow rays (< {min_rays}): no measurable shadow")
            out[int(k)] = ShadowHeight(None, None, int(rr.size), len(runs), reasons)
            continue
        if truncated > 0.3 * len(runs):
            reasons.append(f"{truncated} of {len(runs)} rays cut by another building / image edge: may be too short")
        L = float(np.percentile(runs, 75)) * gsd_m
        out[int(k)] = ShadowHeight(L * tan_el, L, int(rr.size), len(runs), reasons)
    return out
