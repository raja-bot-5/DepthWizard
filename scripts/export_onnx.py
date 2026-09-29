#!/usr/bin/env python3
"""Optional: export a depth model to ONNX and prove numerical equivalence with PyTorch.

  PYTHONPATH=src python scripts/export_onnx.py --model da2_small [--size 518] [--image data/samples/x.png]

Both runs use fp32 on the CPU on the same normalised input. PASS needs relative L2 error < 1e-3.
Writes models/<model>_<size>.onnx + models/<model>_<size>.json (sha256, opset, metrics).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


class DA2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        self.net = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").eval()

    def forward(self, x):
        return self.net(pixel_values=x).predicted_depth


class DA3(torch.nn.Module):
    def __init__(self, no_sky: bool):
        super().__init__()
        from depth_anything_3.api import DepthAnything3
        self.net = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE").model.eval()
        if no_sky:
            # The sky step branches on the data (sky-pixel count), which a static ONNX graph cannot hold.
            # Disabled on BOTH the PyTorch reference and the export, so the check compares like with like.
            self.net._process_mono_sky_estimation = lambda output: output

    def forward(self, x):
        d = self.net(x[:, None])["depth"]
        return d.reshape(d.shape[0], *d.shape[-2:])


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["da2_small", "da3_mono_large"], default="da2_small")
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--image", default=str(ROOT / "data" / "samples" / "gamus_DC_03_26_rgb.png"))
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--no-sky-postprocess", action="store_true",
                    help="DA3 only: drop the data-dependent sky overwrite (differs from the app when sky is detected)")
    a = ap.parse_args()
    assert a.size % 14 == 0, "size must be a multiple of 14 (ViT patch)"
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    tag = f"{a.model}_{a.size}" + ("_nosky" if a.no_sky_postprocess else "")
    onnx_path = out_dir / f"{tag}.onnx"

    img = Image.open(a.image).convert("RGB").resize((a.size, a.size), Image.BICUBIC)
    x = ((np.asarray(img, np.float32)[None].transpose(0, 3, 1, 2) / 255.0) - MEAN) / STD
    x = x.astype(np.float32)
    model = (DA2() if a.model == "da2_small" else DA3(a.no_sky_postprocess)).eval()
    report = {"model": a.model, "size": a.size, "opset": a.opset, "input": a.image, "precision": "fp32 CPU",
              "sky_postprocess": not a.no_sky_postprocess if a.model == "da3_mono_large" else None}
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
        t0 = time.perf_counter()
        try:
            torch.onnx.export(model, (torch.from_numpy(x),), str(onnx_path), opset_version=a.opset,
                              input_names=["pixel_values"], output_names=["depth"], dynamo=True)
        except Exception as exc:  # noqa: BLE001 - record why an export is not possible
            report.update(status="EXPORT_FAILED", error=f"{type(exc).__name__}: {str(exc)[:1500]}")
            (out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            return
        report["export_seconds"] = round(time.perf_counter() - t0, 1)

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(str(onnx_path))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    ort_s = time.perf_counter() - t0
    diff = got.astype(np.float64) - ref.astype(np.float64)
    rel = float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12))
    report.update(status="PASS" if rel < 1e-3 else "FAIL", onnx=str(onnx_path.relative_to(ROOT)),
                  files={q.name: {"mb": round(q.stat().st_size / 1e6, 1), "sha256": sha256(q)}
                         for q in (onnx_path, onnx_path.with_name(onnx_path.name + ".data")) if q.exists()},
                  output_shape=list(got.shape), rel_l2_error=rel, max_abs_diff=float(np.abs(diff).max()),
                  ref_range=[float(ref.min()), float(ref.max())],
                  pearson_r=float(np.corrcoef(got.ravel(), ref.ravel())[0, 1]), ort_cpu_seconds=round(ort_s, 3),
                  threshold="rel_l2_error < 1e-3")
    (out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
