#!/usr/bin/env python3
"""Score a predicted DSM GeoTIFF against a reference DSM/DEM GeoTIFF.

  PYTHONPATH=src python scripts/evaluate.py PRED.tif REF.tif --mode aggregate [--relative] [--out report.json]

--mode native     grids must be identical (nothing is resampled)
--mode aggregate  prediction area-averaged onto the (coarser) reference grid, e.g. Copernicus 30 m
--relative        also report affine-fitted metrics for an rDSM (OPTIMISTIC, labelled as such)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from depthwizard.evaluation import evaluate_rasters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred")
    ap.add_argument("ref")
    ap.add_argument("--mode", choices=["native", "aggregate"], default="native")
    ap.add_argument("--relative", action="store_true")
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument("--out")
    args = ap.parse_args()
    rep = evaluate_rasters(args.pred, args.ref, mode=args.mode, relative=args.relative,
                           min_coverage=args.min_coverage)
    text = json.dumps(rep, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
