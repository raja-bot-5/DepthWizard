#!/usr/bin/env python3
"""Fine-tune a monocular depth model on GAMUS AGL heights (nDSM), portable local <-> Kaggle/Colab.

  Local (DA-V2 Small, 6 GB):  PYTHONPATH=src python training/finetune.py --config configs/finetune_da2_small_local.yaml
  Kaggle (DA3-Mono-Large):    PYTHONPATH=src python training/finetune.py --config configs/finetune_da3_kaggle.yaml \
                                  --data-root /kaggle/working/gamus --out /kaggle/working/runs

Loss: scale-and-shift-invariant L1 (MiDaS-style median/MAD normalisation per image) between
height_sign * prediction and AGL. Sign is preserved, so the model learns "higher value = taller".
Validation: a HELD-OUT CITY; affine-fitted RMSE (metres, OPTIMISTIC by construction) and Pearson r.
Step 0 (zero-shot) is always evaluated so improvement is measured, not assumed.
Records config, seed, per-step log, metrics.json and sha256 of every saved checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
AGL_FLOOR = -5.0  # audit: a clipping floor value in GAMUS AGL; treated as invalid


# ------------------------------------------------------------------ data
def load_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h:
        return h[list(h.keys())[0]][()]


class Gamus(torch.utils.data.Dataset):
    """GAMUS tiles from one or more manifests, filtered by city. Random crops keep native GSD."""

    def __init__(self, root: Path, manifests: list[str], cities: list[str], crop: int, train: bool,
                 crops_per_tile: int = 4):
        self.root, self.crop, self.train = Path(root), crop, train
        tiles = []
        for m in manifests:
            man = json.loads((self.root / m).read_text())
            tiles += [t for t in man["tiles"] if t["city"] in cities]
        if not tiles:
            raise ValueError(f"no tiles for cities {cities} in {manifests}")
        self.tiles = tiles
        self.crops_per_tile = 1 if train else crops_per_tile

    def __len__(self) -> int:
        return len(self.tiles) * self.crops_per_tile

    def __getitem__(self, i: int):
        t = self.tiles[i // self.crops_per_tile]
        img = load_h5(self.root / t["image"])
        agl = load_h5(self.root / t["height"]).astype(np.float32)
        H, W = agl.shape
        c = self.crop
        if self.train:
            r0, c0 = np.random.randint(0, H - c + 1), np.random.randint(0, W - c + 1)
        else:  # deterministic corner crops
            k = i % self.crops_per_tile
            r0, c0 = (0 if k < 2 else H - c), (0 if k % 2 == 0 else W - c)
        img, agl = img[r0:r0 + c, c0:c0 + c], agl[r0:r0 + c, c0:c0 + c]
        if self.train:
            if np.random.rand() < 0.5:
                img, agl = img[:, ::-1], agl[:, ::-1]
            k = np.random.randint(4)  # nadir imagery: rotations are label-preserving
            img, agl = np.rot90(img, k), np.rot90(agl, k)
        valid = np.isfinite(agl) & (agl > AGL_FLOOR)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return x, torch.from_numpy(np.ascontiguousarray(agl)), torch.from_numpy(np.ascontiguousarray(valid))


# ------------------------------------------------------------------ models
class DA2SmallTrain(torch.nn.Module):
    height_sign = +1  # disparity-like

    def __init__(self, repo: str, grad_ckpt: bool):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        self.net = AutoModelForDepthEstimation.from_pretrained(repo)
        if grad_ckpt:
            self.net.gradient_checkpointing_enable()

    def forward(self, x):  # x: normalised (B,3,H,W)
        return self.net(pixel_values=x).predicted_depth  # (B,H,W)

    def param_groups(self, lr_backbone, lr_head):
        bb = [p for n, p in self.net.named_parameters() if n.startswith("backbone.")]
        hd = [p for n, p in self.net.named_parameters() if not n.startswith("backbone.")]
        return [{"params": bb, "lr": lr_backbone}, {"params": hd, "lr": lr_head}]


class DA3MonoTrain(torch.nn.Module):
    height_sign = -1  # depth-like (verified from DA3 source @3d835ec)

    def __init__(self, repo: str, grad_ckpt: bool):
        super().__init__()
        from depth_anything_3.api import DepthAnything3
        # api.forward is @torch.inference_mode -> train the underlying network directly
        self.net = DepthAnything3.from_pretrained(repo).model
        if grad_ckpt:  # DA3 has no switch; wrap the 24 DINOv2 blocks (THIS SHOULD BE TESTED on Kaggle)
            from torch.utils.checkpoint import checkpoint
            for blk in self.net.backbone.pretrained.blocks:
                f = blk.forward
                blk.forward = (lambda f: lambda *a, **k: checkpoint(f, *a, use_reentrant=False, **k))(f)

    def forward(self, x):
        out = self.net(x[:, None])  # (B, N=1, 3, H, W)
        d = out["depth"]
        return d.reshape(d.shape[0], *d.shape[-2:])

    def param_groups(self, lr_backbone, lr_head):
        bb = [p for n, p in self.net.named_parameters() if n.startswith("backbone.")]
        hd = [p for n, p in self.net.named_parameters() if not n.startswith("backbone.")]
        return [{"params": bb, "lr": lr_backbone}, {"params": hd, "lr": lr_head}]


MODELS = {"da2_small": (DA2SmallTrain, "depth-anything/Depth-Anything-V2-Small-hf"),
          "da3_mono_large": (DA3MonoTrain, "depth-anything/DA3MONO-LARGE")}


# ------------------------------------------------------------------ loss / eval
def ssi_normalise(y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(y - median) / mean|y - median| over valid pixels, per image."""
    out = torch.zeros_like(y)
    for b in range(y.shape[0]):
        v = y[b][m[b]]
        if v.numel() < 16:
            continue
        t = v.median()
        s = (v - t).abs().mean().clamp_min(1e-6)
        out[b] = (y[b] - t) / s
    return out


def ssi_l1(h: torch.Tensor, agl: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    diff = (ssi_normalise(h, m) - ssi_normalise(agl, m)).abs()
    return (diff * m).sum() / m.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> dict:
    model.eval()
    rmse, mae, rr = [], [], []
    for x, agl, m in loader:
        x = ((x - MEAN) / STD).to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
            p = model(x)
        h = (model.height_sign * p.float()).cpu().numpy()
        for b in range(h.shape[0]):
            v = m[b].numpy()
            if v.sum() < 100:
                continue
            ph, t = h[b][v].astype(np.float64), agl[b].numpy()[v].astype(np.float64)
            A = np.stack([ph, np.ones_like(ph)], 1)
            (a, c), *_ = np.linalg.lstsq(A, t, rcond=None)
            d = a * ph + c - t
            rmse.append(float(np.sqrt((d ** 2).mean()))); mae.append(float(np.abs(d).mean()))
            rr.append(float(np.corrcoef(ph, t)[0, 1]) if ph.std() > 0 and t.std() > 0 else float("nan"))
    model.train()
    return {"n_crops": len(rmse), "median_fitted_rmse_m": float(np.median(rmse)),
            "median_fitted_mae_m": float(np.median(mae)), "median_r": float(np.nanmedian(rr)),
            "note": "affine fitted per crop on the eval data itself -> OPTIMISTIC"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", help="overrides config data.root")
    ap.add_argument("--out", help="overrides config out_dir")
    ap.add_argument("--steps", type=int, help="overrides config train.steps")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root: cfg["data"]["root"] = args.data_root
    if args.out: cfg["out_dir"] = args.out
    if args.steps: cfg["train"]["steps"] = args.steps

    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    amp = cfg["train"].get("amp", True) and device == "cuda"
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if amp else None

    out = Path(cfg["out_dir"]) / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{cfg['model']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    d = cfg["data"]
    tr = Gamus(d["root"], d["train_manifests"], d["train_cities"], d["crop"], train=True)
    va = Gamus(d["root"], d["val_manifests"], d["val_cities"], d["crop"], train=False)
    g = torch.Generator().manual_seed(seed)
    tl = torch.utils.data.DataLoader(tr, batch_size=cfg["train"]["batch"], shuffle=True, generator=g,
                                     num_workers=d.get("workers", 2), drop_last=True,
                                     worker_init_fn=lambda w: np.random.seed(seed + w), persistent_workers=True)
    vl = torch.utils.data.DataLoader(va, batch_size=cfg["train"]["batch"], shuffle=False, num_workers=d.get("workers", 2))

    cls, repo = MODELS[cfg["model"]]
    model = cls(repo, cfg["train"].get("grad_ckpt", True)).to(device)
    opt = torch.optim.AdamW(model.param_groups(cfg["train"]["lr_backbone"], cfg["train"]["lr_head"]),
                            weight_decay=cfg["train"].get("weight_decay", 0.01))
    steps, warm = cfg["train"]["steps"], cfg["train"].get("warmup", 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    accum = cfg["train"].get("accum", 1)

    prov = {"python": platform.python_version(), "torch": torch.__version__, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "amp_dtype": str(amp_dtype),
            "seed": seed, "train_tiles": len(tr.tiles), "val_tiles": len(va.tiles),
            "train_cities": d["train_cities"], "val_cities": d["val_cities"], "repo": repo,
            "trainable_params_M": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2), flush=True)

    history = {"val": [], "train": []}
    v0 = evaluate(model, vl, device, amp_dtype)
    history["val"].append({"step": 0, **v0})
    print(f"[val step 0 / zero-shot] {v0}", flush=True)
    best, best_path = v0["median_fitted_rmse_m"], None
    log = open(out / "log.jsonl", "w")
    step, t0, it = 0, time.time(), iter(tl)
    model.train()
    while step < steps:
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            try:
                x, agl, m = next(it)
            except StopIteration:
                it = iter(tl); x, agl, m = next(it)
            x, agl, m = ((x - MEAN) / STD).to(device), agl.to(device), m.to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp):
                p = model(x)
            loss = ssi_l1(model.height_sign * p.float(), agl, m) / accum
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("clip", 1.0))
        scaler.step(opt); scaler.update(); sched.step()
        step += 1
        rec = {"step": step, "loss": tot, "lr": sched.get_last_lr()[0], "s": round(time.time() - t0, 1)}
        if device == "cuda":
            rec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        log.write(json.dumps(rec) + "\n"); log.flush()
        history["train"].append(rec)
        if step % cfg["train"].get("log_every", 10) == 0:
            print(rec, flush=True)
        if step % cfg["train"]["eval_every"] == 0 or step == steps:
            v = evaluate(model, vl, device, amp_dtype)
            history["val"].append({"step": step, **v})
            print(f"[val step {step}] {v}", flush=True)
            if v["median_fitted_rmse_m"] < best:
                best = v["median_fitted_rmse_m"]
                best_path = out / "best.pt"
                torch.save(model.state_dict(), best_path)
    log.close()
    last = out / "last.pt"
    torch.save(model.state_dict(), last)

    def window_mean(xs):
        return float(np.mean(xs)) if xs else None
    losses = [r["loss"] for r in history["train"]]
    n = max(1, len(losses) // 5)
    ckpts = {p.name: {"sha256": sha256(p), "mb": round(p.stat().st_size / 1e6, 1)} for p in (best_path, last) if p}
    metrics = {"steps": steps, "loss_first_20pct_mean": window_mean(losses[:n]),
               "loss_last_20pct_mean": window_mean(losses[-n:]), "val": history["val"],
               "best_val_median_fitted_rmse_m": best, "zero_shot_val_median_fitted_rmse_m": v0["median_fitted_rmse_m"],
               "checkpoints": ckpts, "provenance": prov}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("loss_first_20pct_mean", "loss_last_20pct_mean",
                                               "zero_shot_val_median_fitted_rmse_m", "best_val_median_fitted_rmse_m",
                                               "checkpoints")}, indent=2))
    print("run dir:", out)
    if cfg.get("keep_only_best", False) and best_path:
        last.unlink()


if __name__ == "__main__":
    main()
