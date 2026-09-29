"""Depth predictors behind one contract, usable directly by tiling.run_tiled.

predictor(crop HxWx3 uint8) -> float32 HxW in the MODEL's relative units, resized
back to the crop size. Values are never metres. `height_sign` states how to turn
the output into something that grows with terrain height for a nadir view:
  disparity-like (DA-V2): closer = higher ground -> +1
  depth-like     (DA3-Mono): farther = lower ground -> -1
THIS SHOULD BE TESTED against the DEM in Phase 4; the sign is not a calibration.

Both wrappers were verified in experiments/00_smoke (APIs read from source, DA3 @ 3d835ec).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


class DepthPredictor(Protocol):
    name: str
    polarity: str
    height_sign: int

    def __call__(self, crop: np.ndarray) -> np.ndarray: ...


def _to_crop_size(pred: np.ndarray, h: int, w: int) -> np.ndarray:
    """Model output -> crop size. Bilinear on model output (not on geodata)."""
    pred = np.asarray(pred, dtype=np.float32)
    if pred.shape == (h, w):
        return pred
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)


def _check_crop(crop: np.ndarray) -> None:
    if crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8:
        raise ValueError(f"expected HxWx3 uint8 RGB, got {crop.shape} {crop.dtype}")


@dataclass
class DA2Small:
    device: str = "cuda"
    fp16: bool = True
    process_res: int = 518
    name: str = "da2_small"
    polarity: str = "disparity-like (higher = closer)"
    height_sign: int = +1
    repo: str = "depth-anything/Depth-Anything-V2-Small-hf"

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self._torch = torch
        self.dtype = torch.float16 if (self.fp16 and self.device == "cuda") else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(self.repo)
        self.model = (AutoModelForDepthEstimation.from_pretrained(self.repo, dtype=self.dtype)
                      .to(self.device).eval())

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        _check_crop(crop)
        s = self.process_res
        with self._torch.inference_mode():
            px = self.processor(images=crop, return_tensors="pt", size={"height": s, "width": s})
            out = self.model(pixel_values=px["pixel_values"].to(self.device, dtype=self.dtype))
        return _to_crop_size(out.predicted_depth[0].float().cpu().numpy(), *crop.shape[:2])


@dataclass
class DA3MonoLarge:
    device: str = "cuda"
    process_res: int = 518
    name: str = "da3_mono_large"
    polarity: str = "depth-like (higher = farther)"
    height_sign: int = -1
    repo: str = "depth-anything/DA3MONO-LARGE"
    max_sky_fraction: float = 0.01

    def __post_init__(self) -> None:
        import torch
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(self.repo).to(device=torch.device(self.device)).eval()
        self.last_sky_fraction: float | None = None

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        _check_crop(crop)
        pred = self.model.inference([crop], process_res=self.process_res,
                                    process_res_method="upper_bound_resize")
        # DA3 overwrites "sky" pixels with p99 depth; overhead scenes have no sky, so
        # a large sky fraction means corrupted heights. Refuse rather than pass them on.
        self.last_sky_fraction = float(np.mean(pred.sky)) if pred.sky is not None else None
        if self.last_sky_fraction is not None and self.last_sky_fraction > self.max_sky_fraction:
            raise RuntimeError(f"DA3 flagged {self.last_sky_fraction:.1%} of the tile as sky "
                               f"(> {self.max_sky_fraction:.0%}); its depth there was overwritten")
        return _to_crop_size(pred.depth[0], *crop.shape[:2])


PREDICTORS = {"da2_small": DA2Small, "da3_mono_large": DA3MonoLarge}
