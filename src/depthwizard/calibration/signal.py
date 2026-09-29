"""Relative-signal conditioning: one polarity for every backend, split at the DEM's resolution.

R = height_sign * model_output, so that a larger R means a HIGHER surface for every backend:
  DA-V2 Small   disparity-like (higher = closer to a nadir camera = taller)  -> height_sign +1
  DA3-Mono      depth-like (verified in DA3 source @3d835ec: exp head; sky -> p99 depth) -> -1
Then R = low(R) + high(R), with low = NaN-aware box mean over the DEM's effective resolution
(30 m for Copernicus GLO-30, converted to pixels from the image GSD). Units stay relative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from depthwizard.calibration.core import lowpass


@dataclass
class Signal:
    r: np.ndarray                 # height-like relative signal (float32, unitless)
    low: np.ndarray               # low-pass at the DEM resolution
    high: np.ndarray              # r - low: fine structure the DEM cannot represent
    lowpass_px: int
    info: dict = field(default_factory=dict)


def lowpass_pixels(gsd_m: float, dem_res_m: float = 30.0) -> int:
    if gsd_m <= 0:
        raise ValueError("GSD must be positive")
    return max(1, int(round(dem_res_m / gsd_m)))


def condition(prediction: np.ndarray, height_sign: int, gsd_m: float, dem_res_m: float = 30.0,
              polarity: str = "") -> Signal:
    if height_sign not in (-1, 1):
        raise ValueError("height_sign must be +1 or -1")
    r = (height_sign * np.asarray(prediction, np.float32)).astype(np.float32)
    px = lowpass_pixels(gsd_m, dem_res_m)
    low = lowpass(r, px)
    return Signal(r=r, low=low, high=(r - low).astype(np.float32), lowpass_px=px,
                  info={"height_sign": height_sign, "polarity": polarity, "gsd_m": gsd_m,
                        "dem_res_m": dem_res_m, "lowpass_px": px, "units": "relative (unitless)"})
