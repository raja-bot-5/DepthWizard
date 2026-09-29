#!/usr/bin/env python3
"""Phase 1c smoke test: prove the depth models run on this machine and measure cost.

This is a RUNTIME test, NOT an accuracy test. Outputs are relative depth/disparity
in model units — never metres.

Per model x size: warmup, timed end-to-end runs (preprocess + forward + copy to
CPU, with cuda synchronize), peak VRAM, output stats, raw float32 .npy and a
normalized PNG that is for viewing only.

Models
  da2_small       depth-anything/Depth-Anything-V2-Small-hf  (transformers)
                  output: predicted_depth, disparity-like (higher = closer)
  da3_mono_large  depth-anything/DA3MONO-LARGE              (depth_anything_3.api)
                  output: prediction.depth, depth-like (higher = farther).
                  Verified in third_party/depth-anything-3 @ 3d835ec: DPT head
                  activation "exp" (model/dpt.py), and sky pixels are set to the
                  p99 depth, i.e. the far end (model/da3.py _process_mono_sky_estimation).

Usage
  python experiments/00_smoke/smoke_test_models.py --image scene.tif --fp16
  python experiments/00_smoke/smoke_test_models.py --device cpu     # no CUDA: 518 only
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[2]
DA3_DIR = ROOT / "third_party" / "depth-anything-3"

MODELS = {
    "da2_small": {
        "repo": "depth-anything/Depth-Anything-V2-Small-hf",
        "polarity": "disparity-like (higher = closer)",
        "polarity_evidence": "DA-V2 is trained on affine-invariant inverse depth (DA-V2 paper / HF card)",
    },
    "da3_mono_large": {
        "repo": "depth-anything/DA3MONO-LARGE",
        "polarity": "depth-like (higher = farther)",
        "polarity_evidence": "third_party/depth-anything-3 src: model/dpt.py activation='exp' on head "
                             "'depth'; model/da3.py sets sky pixels to p99 (max) depth",
    },
}


# ----------------------------------------------------------------- helpers
def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def env_info(device: str) -> dict:
    import transformers
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "transformers": transformers.__version__,
        "driver": run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) or "UNAVAILABLE",
        "da3_commit": run(["git", "-C", str(DA3_DIR), "rev-parse", "HEAD"]) or "UNKNOWN",
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_total_vram_gb=round(props.total_memory / 1e9, 2),
                    bf16_supported=torch.cuda.is_bf16_supported())
    try:
        import psutil
        info["ram_available_gb_at_start"] = round(psutil.virtual_memory().available / 1e9, 2)
    except ImportError:
        pass
    return info


def load_image(path: str | None, seed: int) -> tuple[np.ndarray, dict]:
    """Return an HxWx3 uint8 RGB array plus a description of where it came from."""
    if path:
        p = Path(path)
        if p.suffix.lower() in (".tif", ".tiff"):
            import rasterio
            with rasterio.open(p) as src:
                arr = src.read(indexes=[1, 2, 3] if src.count >= 3 else [1] * 3)
                src_dtype = src.dtypes[0]
            arr = np.moveaxis(arr, 0, -1)
            if arr.dtype != np.uint8:
                # preview-style stretch only to feed the model; the source raster is untouched
                lo, hi = np.percentile(arr, (1, 99))
                arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            desc = {"source": str(p), "kind": "tiff", "source_dtype": src_dtype}
        else:
            arr = np.asarray(Image.open(p).convert("RGB"))
            desc = {"source": str(p), "kind": "image"}
        desc["shape"] = list(arr.shape)
        return arr, desc

    # Synthetic fallback: smooth "terrain" + flat bright "roofs". Timing only.
    rng = np.random.default_rng(seed)
    n = 1036
    base = rng.normal(size=(n // 37, n // 37, 3)).astype(np.float32)
    img = np.asarray(Image.fromarray(((base - base.min()) / np.ptp(base) * 255).astype(np.uint8))
                     .resize((n, n), Image.BICUBIC), dtype=np.float32)
    img = 0.6 * img + 0.4 * rng.integers(60, 120, size=(n, n, 3))
    for _ in range(40):
        y, x = rng.integers(0, n - 80, 2)
        h, w = rng.integers(20, 80, 2)
        img[y:y + h, x:x + w] = rng.integers(150, 240, 3)
    arr = np.clip(img, 0, 255).astype(np.uint8)
    return arr, {"source": f"SYNTHETIC (seed={seed})", "kind": "synthetic", "shape": list(arr.shape),
                 "note": "no real image given - runtime test only, outputs are meaningless"}


def save_outputs(out_dir: Path, tag: str, depth: np.ndarray, polarity: str) -> dict:
    depth = depth.astype(np.float32)
    npy = out_dir / f"{tag}_raw_float32.npy"
    np.save(npy, depth)
    finite = depth[np.isfinite(depth)]
    lo, hi = (np.percentile(finite, (2, 98)) if finite.size else (0.0, 1.0))
    norm = np.clip((depth - lo) / max(hi - lo, 1e-12), 0, 1)
    png = out_dir / f"{tag}_PREVIEW_ONLY.png"
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Description", f"PREVIEW ONLY - p2/p98 normalized, not physical units. {polarity}. "
                                 f"Raw values: {npy.name}")
    Image.fromarray((np.nan_to_num(norm) * 255).astype(np.uint8)).save(png, pnginfo=meta)
    return {"raw_npy": npy.name, "preview_png": png.name, "preview_norm_p2_p98": [float(lo), float(hi)]}


def stats(depth: np.ndarray) -> dict:
    finite = depth[np.isfinite(depth)]
    return {"shape": list(depth.shape), "dtype": str(depth.dtype),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "nan_count": int(np.isnan(depth).sum()), "inf_count": int(np.isinf(depth).sum())}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_info(repo: str) -> dict:
    """Resolved HF commit + sha256 of the cached safetensors (provenance)."""
    from huggingface_hub import HfApi, try_to_load_from_cache
    out = {"repo": repo}
    try:
        out["hf_commit"] = HfApi().model_info(repo).sha
    except Exception as exc:  # noqa: BLE001 - offline is fine, record it
        out["hf_commit"] = f"UNKNOWN ({type(exc).__name__})"
    cached = try_to_load_from_cache(repo, "model.safetensors")
    if isinstance(cached, str):
        out["safetensors_sha256"] = sha256(Path(cached))
        out["safetensors_mb"] = round(Path(cached).stat().st_size / 1e6, 1)
    return out


# ----------------------------------------------------------------- runners
class DA2Small:
    def __init__(self, device: str, fp16: bool):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        repo = MODELS["da2_small"]["repo"]
        self.device = device
        self.dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo, dtype=self.dtype).to(device).eval()
        self.precision = f"weights+inputs {str(self.dtype).replace('torch.', '')}"

    @torch.inference_mode()
    def __call__(self, img: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
        inputs = self.processor(images=img, return_tensors="pt", size={"height": size, "width": size})
        pix = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        out = self.model(pixel_values=pix).predicted_depth  # (1, H, W)
        return out[0].float().cpu().numpy(), {"model_input_hw": list(pix.shape[-2:])}


class DA3MonoLarge:
    def __init__(self, device: str, fp16: bool):
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(MODELS["da3_mono_large"]["repo"]).to(device=torch.device(device))
        self.model.eval()
        bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        # DA3 api.forward always autocasts (bf16 if cuda bf16 supported, else fp16); --fp16 cannot change it
        self.precision = (f"fp32 weights, autocast {'bf16' if bf16 else 'fp16'} forced by DA3 api.forward "
                          f"(--fp16={fp16} has no effect)")

    def __call__(self, img: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
        pred = self.model.inference([img], process_res=size, process_res_method="upper_bound_resize")
        # output_processor uses getattr(addict.Dict, "is_metric", 0); addict returns an empty Dict
        # (not the default) when the model never set it, so only trust real numbers here
        is_metric = pred.is_metric if isinstance(pred.is_metric, (int, float, bool)) else "not set by model"
        extra = {"model_input_hw": list(pred.depth.shape[-2:]), "is_metric": is_metric}
        if pred.sky is not None:
            # sky pixels get their depth overwritten with p99 depth -> must be ~0 on overhead imagery
            extra["sky_fraction"] = float(np.mean(pred.sky))
        return pred.depth[0].astype(np.float32), extra


RUNNERS = {"da2_small": DA2Small, "da3_mono_large": DA3MonoLarge}


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def free(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="PNG/JPG/TIF. If omitted a synthetic image is used (runtime test only).")
    ap.add_argument("--models", default="da2_small,da3_mono_large")
    ap.add_argument("--sizes", default="518,1036")
    ap.add_argument("--fp16", action="store_true", help="fp16 for DA-V2 on CUDA (DA3 precision is fixed by its API)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default runs/smoke/<UTC timestamp>/")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = [int(s) for s in args.sizes.split(",")]
    unknown = [m for m in models if m not in RUNNERS]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; choose from {list(RUNNERS)}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[error] CUDA is not available to PyTorch (driver not loaded?). "
              "Fix the driver, or rerun with --device cpu (size 518 only).", file=sys.stderr)
        return 2
    if args.device == "cpu" and sizes != [518]:
        print(f"[warn] --device cpu: restricting sizes {sizes} -> [518]")
        sizes = [518]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else ROOT / "runs" / "smoke" / f"{stamp}_{args.device}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"purpose": "RUNTIME smoke test - NOT an accuracy test; outputs are relative, not metres",
              "args": vars(args), "env": env_info(args.device)}
    ram = report["env"].get("ram_available_gb_at_start")
    if ram is not None and ram < 5:
        print(f"[warn] only {ram} GB RAM available (<5). Close other apps if this gets killed.")

    img, img_desc = load_image(args.image, args.seed)
    report["input"] = img_desc
    if img_desc["kind"] == "synthetic":
        print("[warn] no --image given: using a SYNTHETIC image. Runtime test only, outputs are meaningless.")
    if args.device == "cpu":
        print("[warn] CPU run: timings are CPU timings, not representative of GPU deployment.")

    report["results"] = []
    report_path = out_dir / "report.json"

    def flush() -> None:
        report_path.write_text(json.dumps(report, indent=2))

    for name in models:
        meta = MODELS[name]
        print(f"\n=== {name} ({meta['repo']}) on {args.device}")
        free(args.device)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            runner = RUNNERS[name](args.device, args.fp16)
        except Exception as exc:  # noqa: BLE001 - record and continue with next model
            status = "OOM" if is_oom(exc) else "LOAD_ERROR"
            report["results"].append({"model": name, "status": status, "error": repr(exc)[:2000],
                                      "traceback": traceback.format_exc()[-4000:]})
            print(f"[{status}] {exc!r}")
            flush()
            continue
        sync(args.device)
        load_s = time.perf_counter() - t0
        weights_vram = torch.cuda.memory_allocated() / 1e9 if args.device == "cuda" else None
        ckpt = checkpoint_info(meta["repo"])

        for size in sizes:
            tag = f"{name}_{size}_{args.device}"
            rec = {"model": name, "size": size, "device": args.device, "precision": runner.precision,
                   "polarity": meta["polarity"], "polarity_evidence": meta["polarity_evidence"],
                   "checkpoint": ckpt, "load_seconds": round(load_s, 2),
                   "weights_vram_gb": round(weights_vram, 3) if weights_vram is not None else None}
            try:
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                for _ in range(args.warmup):
                    runner(img, size)
                sync(args.device)
                times = []
                depth, extra = None, {}
                for _ in range(args.runs):
                    sync(args.device)
                    t = time.perf_counter()
                    depth, extra = runner(img, size)
                    sync(args.device)
                    times.append((time.perf_counter() - t) * 1000)
                rec.update(status="OK", **extra,
                           latency_ms={"mean": round(float(np.mean(times)), 2),
                                       "std": round(float(np.std(times)), 2),
                                       "min": round(float(np.min(times)), 2),
                                       "median": round(float(np.median(times)), 2),
                                       "all": [round(x, 2) for x in times]},
                           latency_scope="end-to-end: preprocess + forward + output to CPU numpy",
                           peak_vram_gb=(round(torch.cuda.max_memory_allocated() / 1e9, 3)
                                         if args.device == "cuda" else None),
                           output=stats(depth), files=save_outputs(out_dir, tag, depth, meta["polarity"]))
                print(f"[OK] {tag}: {rec['latency_ms']['mean']:.1f} ms mean, "
                      f"peak VRAM {rec['peak_vram_gb']} GB, out {rec['output']['shape']} "
                      f"min {rec['output']['min']:.4g} max {rec['output']['max']:.4g} NaN {rec['output']['nan_count']}")
            except Exception as exc:  # noqa: BLE001 - record OOM/errors per config and continue
                rec.update(status="OOM" if is_oom(exc) else "ERROR", error=repr(exc)[:2000],
                           traceback=traceback.format_exc()[-4000:])
                print(f"[{rec['status']}] {tag}: {exc!r}")
            finally:
                free(args.device)
            report["results"].append(rec)
            flush()

        del runner
        free(args.device)

    flush()
    ok = sum(r.get("status") == "OK" for r in report["results"])
    print(f"\n{ok}/{len(report['results'])} configurations OK - report: {report_path}")
    return 0 if ok == len(report["results"]) else 1


if __name__ == "__main__":
    sys.exit(main())
