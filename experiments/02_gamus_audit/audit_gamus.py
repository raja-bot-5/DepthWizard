#!/usr/bin/env python3
"""Audit the downloaded GAMUS subset: layout, dtypes, value ranges, sentinels, classes.

Nothing here assumes units: the files carry no metadata, so every claim in the
report is either measured from the data or marked as coming from the paper.

  PYTHONPATH=src python experiments/02_gamus_audit/audit_gamus.py
"""
from __future__ import annotations

import collections
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GAMUS = ROOT / "data" / "raw" / "gamus"
CLASS_NAMES = {0: "others", 1: "ground", 2: "low vegetation", 3: "buildings", 4: "water", 5: "road", 6: "tree"}


def read(p: Path) -> tuple[np.ndarray, dict]:
    with h5py.File(p, "r") as h:
        keys = list(h.keys())
        return h[keys[0]][()], {"keys": keys, "file_attrs": dict(h.attrs), "ds_attrs": dict(h[keys[0]].attrs)}


def main() -> None:
    manifest = json.loads((GAMUS / "manifest.json").read_text())
    rows, layout = [], collections.Counter()
    height_counter: collections.Counter = collections.Counter()
    class_px: collections.Counter = collections.Counter()
    heights_by_class: dict[int, list] = collections.defaultdict(list)
    for t in manifest["tiles"]:
        img, mi = read(GAMUS / t["image"])
        agl, mh = read(GAMUS / t["height"])
        cls_f, mc = read(GAMUS / t["classes"])
        layout[(tuple(mi["keys"]), img.shape, str(img.dtype), agl.shape, str(agl.dtype), cls_f.shape, str(cls_f.dtype))] += 1
        if any(m["file_attrs"] or m["ds_attrs"] for m in (mi, mh, mc)):
            print("NOTE metadata found:", t["id"], mi, mh, mc)
        cls_int = np.rint(cls_f).astype(np.int16)
        vals, cnt = np.unique(np.round(agl, 3), return_counts=True)
        top = sorted(zip(cnt, vals), reverse=True)[:3]
        for c, v in top:
            height_counter[float(v)] += int(c)
        for k, n in zip(*np.unique(cls_int, return_counts=True)):
            class_px[int(k)] += int(n)
        rng = np.random.default_rng(0)
        idx = rng.choice(agl.size, 20000, replace=False)
        for k in CLASS_NAMES:
            sel = cls_int.ravel()[idx] == k
            heights_by_class[k].append(agl.ravel()[idx][sel])
        rows.append({
            "id": t["id"], "city": t["city"],
            "agl_min": float(np.nanmin(agl)), "agl_max": float(np.nanmax(agl)),
            "agl_mean": float(np.nanmean(agl)), "agl_nan": int(np.isnan(agl).sum()),
            "agl_eq_minus5_frac": float(np.mean(agl == -5.0)),
            "agl_lt0_frac": float(np.mean(agl < 0)),
            "agl_most_common": [[float(v), int(c)] for c, v in top],
            "classes_integral": bool(np.all(cls_f == np.rint(cls_f))),
            "class_values": sorted(int(v) for v in np.unique(cls_int)),
            "rgb_mean": [float(x) for x in img.reshape(-1, 3).mean(0)],
        })

    by_class = {}
    for k, name in CLASS_NAMES.items():
        a = np.concatenate(heights_by_class[k]) if heights_by_class[k] else np.array([])
        by_class[name] = ({"n_sampled": int(a.size), "p5": float(np.percentile(a, 5)),
                           "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                           "frac_eq_minus5": float(np.mean(a == -5.0))} if a.size else {"n_sampled": 0})

    total_px = sum(class_px.values())
    report = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "source": {k: manifest[k] for k in ("repo", "revision", "license", "split")},
        "n_tiles": len(rows),
        "tiles_per_city": dict(collections.Counter(r["city"] for r in rows)),
        "layouts_seen": [{"keys": k[0], "image": [list(k[1]), k[2]], "agl": [list(k[3]), k[4]],
                          "classes": [list(k[5]), k[6]], "tiles": n} for k, n in layout.items()],
        "metadata_in_files": "none found (no HDF5 attrs): units / GSD / CRS not stated in the data",
        "agl_global": {"min": min(r["agl_min"] for r in rows), "max": max(r["agl_max"] for r in rows),
                       "nan_total": sum(r["agl_nan"] for r in rows),
                       "tiles_with_minus5": sum(r["agl_eq_minus5_frac"] > 0 for r in rows),
                       "mean_frac_minus5": float(np.mean([r["agl_eq_minus5_frac"] for r in rows])),
                       "mean_frac_below0": float(np.mean([r["agl_lt0_frac"] for r in rows]))},
        "agl_by_class_sampled": by_class,
        "class_pixel_share": {CLASS_NAMES.get(k, str(k)): round(v / total_px, 4) for k, v in sorted(class_px.items())},
        "all_classes_integral": all(r["classes_integral"] for r in rows),
        "tiles": rows,
    }
    out = ROOT / "runs" / "gamus_audit" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "tiles"}, indent=2))
    print("per-tile rows in", out / "audit.json")


if __name__ == "__main__":
    main()
