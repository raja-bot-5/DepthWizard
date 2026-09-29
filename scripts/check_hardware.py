#!/usr/bin/env python3
"""DepthWizard hardware/environment readiness check.

Reports GPU, VRAM, RAM, disk, CPU, PRIME mode and library availability, and
flags anything that will block Phase 1. Writes a JSON report so the result is
recorded alongside experiments.

Usage:
    python scripts/check_hardware.py [--out runs/env/hardware.json]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Thresholds derived from the Phase 1 plan (DA3-Mono-Large fp16 inference,
# DA-V2 Small fine-tuning). They are guidance, not hard limits.
MIN_FREE_VRAM_GB = 3.0
MIN_AVAILABLE_RAM_GB = 5.0
MIN_FREE_DISK_GB = 25.0

LIBS = ["torch", "torchvision", "transformers", "rasterio", "pyproj",
        "cv2", "numpy", "scipy", "sklearn", "psutil", "fastapi", "depth_anything_3"]


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def lib_versions() -> dict[str, str]:
    out = {}
    for name in LIBS:
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            out[name] = f"MISSING ({type(exc).__name__})"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/env/hardware.json")
    args = ap.parse_args()

    report: dict = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "os": platform.platform(), "python": platform.python_version()}
    issues: list[str] = []
    warnings: list[str] = []

    # CPU / RAM
    report["cpu_logical_cores"] = os.cpu_count()
    try:
        import psutil
        vm = psutil.virtual_memory()
        report["ram_total_gb"] = round(vm.total / 1e9, 2)
        report["ram_available_gb"] = round(vm.available / 1e9, 2)
        swap = psutil.swap_memory()
        report["swap_total_gb"] = round(swap.total / 1e9, 2)
        if vm.available / 1e9 < MIN_AVAILABLE_RAM_GB:
            warnings.append(
                f"Only {vm.available/1e9:.1f} GB RAM available (<{MIN_AVAILABLE_RAM_GB}). "
                "Close browsers/IDEs before running models, or add swap.")
    except ImportError:
        issues.append("psutil missing — run setup/setup_env.sh")

    # Disk
    free = shutil.disk_usage(".").free / 1e9
    report["disk_free_gb"] = round(free, 1)
    if free < MIN_FREE_DISK_GB:
        warnings.append(f"Only {free:.0f} GB disk free (<{MIN_FREE_DISK_GB}).")

    # PRIME mode (iGPU drives display -> full VRAM for compute)
    prime = run(["prime-select", "query"])
    report["prime_mode"] = prime or "unknown (prime-select not found)"
    if prime and prime != "on-demand":
        warnings.append(f"PRIME mode is '{prime}'. 'on-demand' lets the Intel iGPU drive the "
                        "display so more NVIDIA VRAM is free: sudo prime-select on-demand && reboot")

    # nvidia-smi
    smi = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
               "--format=csv,noheader,nounits"])
    report["nvidia_smi"] = smi or "not available"
    if not smi:
        issues.append("nvidia-smi unavailable — NVIDIA driver not installed/loaded.")

    # torch view of the GPU
    try:
        import torch
        report["torch"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            props = torch.cuda.get_device_properties(0)
            report["gpu_name"] = props.name
            report["gpu_compute_capability"] = f"{props.major}.{props.minor}"
            report["vram_total_gb"] = round(total_b / 1e9, 2)
            report["vram_free_gb"] = round(free_b / 1e9, 2)
            report["bf16_supported"] = torch.cuda.is_bf16_supported()
            if free_b / 1e9 < MIN_FREE_VRAM_GB:
                warnings.append(f"Only {free_b/1e9:.1f} GB VRAM free — another process is using the GPU "
                                "(check nvidia-smi).")
        else:
            issues.append("PyTorch cannot see CUDA. Wrong wheel channel or driver too old.")
    except ImportError:
        issues.append("torch missing — run setup/setup_env.sh")

    report["libraries"] = lib_versions()
    for name, ver in report["libraries"].items():
        if ver.startswith("MISSING") and name != "depth_anything_3":
            issues.append(f"Library missing: {name}")
    if report["libraries"]["depth_anything_3"].startswith("MISSING"):
        warnings.append("depth_anything_3 not importable — DA3 will be skipped in the smoke test.")

    report["blocking_issues"] = issues
    report["warnings"] = warnings
    report["ready"] = not issues

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if k != "libraries"}, indent=2))
    print("\nLibraries:")
    for k, v in report["libraries"].items():
        print(f"  {k:18s} {v}")
    print(f"\n{'READY' if report['ready'] else 'NOT READY'} — report written to {out}")


if __name__ == "__main__":
    main()
