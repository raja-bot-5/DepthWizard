#!/usr/bin/env python3
"""Download a small, reproducible GAMUS subset (image + AGL height + classes per tile).

Tiles are picked evenly spaced per city from one split, pinned to a dataset commit,
and every file's sha256 is written to data/raw/gamus/manifest.json.

  PYTHONPATH=src python scripts/fetch_gamus_subset.py --split test --per-city 10

GAMUS: HF earthflow/GAMUS (CC-BY-4.0). ~9.3 MB per tile triplet.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "earthflow/GAMUS"
ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tile_id(image_name: str) -> str:
    # images are "<id>_RGB.h5" (DC, PHL) or "<id>_IMG.h5" (NYC)
    stem = Path(image_name).stem
    return stem.rsplit("_", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--per-city", type=int, default=10)
    ap.add_argument("--revision", default="a3c0e2511f06d909612406f436cf8abb4da805f5")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "gamus"))
    ap.add_argument("--max-gb", type=float, default=1.0, help="refuse if the estimate exceeds this")
    ap.add_argument("--cities", default=None, help="comma list, e.g. DC,PHL (default: all)")
    args = ap.parse_args()

    info = HfApi().dataset_info(REPO, revision=args.revision, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings}
    images = sorted(f for f in sizes if f.startswith(f"images/{args.split}/"))
    by_city: dict[str, list[str]] = collections.defaultdict(list)
    for f in images:
        by_city[Path(f).name.split("_")[0]].append(f)

    chosen = []
    keep = set(args.cities.split(",")) if args.cities else None
    for city, files in sorted(by_city.items()):
        if keep and city not in keep:
            continue
        n = min(args.per_city, len(files))
        idx = [round(i * (len(files) - 1) / max(n - 1, 1)) for i in range(n)]
        for i in sorted(set(idx)):
            img = files[i]
            tid = tile_id(img)
            trip = {"city": city, "id": tid, "image": img,
                    "height": f"heights/{args.split}/{tid}_AGL.h5",
                    "classes": f"classes/{args.split}/{tid}_CLS.h5"}
            missing = [trip[k] for k in ("height", "classes") if trip[k] not in sizes]
            if missing:
                print(f"[skip] {tid}: missing {missing}")
                continue
            chosen.append(trip)

    est = sum(sizes[t[k]] for t in chosen for k in ("image", "height", "classes")) / 1e9
    free = shutil.disk_usage(ROOT).free / 1e9
    print(f"{len(chosen)} tiles, estimated {est:.2f} GB, disk free {free:.1f} GB")
    if est > args.max_gb:
        raise SystemExit(f"estimate {est:.2f} GB > --max-gb {args.max_gb}; not downloading")

    out = Path(args.out)
    manifest = {"repo": REPO, "revision": args.revision, "license": "CC-BY-4.0 (HF card)",
                "split": args.split, "per_city": args.per_city,
                "fetched_utc": datetime.now(timezone.utc).isoformat(), "tiles": []}
    for n, t in enumerate(chosen, 1):
        rec = dict(t)
        for k in ("image", "height", "classes"):
            p = Path(hf_hub_download(REPO, t[k], repo_type="dataset", revision=args.revision, local_dir=out))
            rec[f"{k}_sha256"] = sha256(p)
        manifest["tiles"].append(rec)
        print(f"[{n}/{len(chosen)}] {t['id']}", flush=True)
        (out / f"manifest_{args.split}.json").write_text(json.dumps(manifest, indent=2))
    print("done:", out / f"manifest_{args.split}.json")


if __name__ == "__main__":
    main()
