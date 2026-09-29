#!/usr/bin/env python3
"""Cut a window from a NAIP Cloud-Optimized GeoTIFF (Microsoft Planetary Computer, anonymous SAS token).

Pure pixel crop: native CRS, native 0.6 m grid, the window's own transform, RGB only.

  PYTHONPATH=src python scripts/fetch_naip_window.py --lon -105.27 --lat 40.01 --size 2048

NAIP: USDA FSA, license link "Public Domain" (STAC `license` field says "proprietary").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from depthwizard.geo.planetary import naip_window

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--gsd", type=float, default=0.6)
    ap.add_argument("--item")
    ap.add_argument("--out", default=str(ROOT / "data" / "samples"))
    a = ap.parse_args()
    tif, prov = naip_window(a.lon, a.lat, a.size, a.out, gsd=a.gsd, item_id=a.item)
    print(json.dumps({"file": str(tif), "kb": round(tif.stat().st_size / 1024), **{k: prov[k] for k in ("item", "gsd", "window")}}, indent=2))


if __name__ == "__main__":
    main()
