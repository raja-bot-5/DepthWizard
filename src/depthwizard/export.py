"""DSM export: one self-describing GeoTIFF + a JSON sidecar per product.

Every output states what it is: which model produced it, how it was calibrated,
whether its values are metres or relative, and in which vertical datum. An rDSM
(any non-georeferenced input, or any uncalibrated output) can never be labelled metres.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from depthwizard.io import RasterMetadata, write_dsm

RELATIVE_UNITS = "relative (unitless, model-dependent scale and offset)"


def export_dsm(path: str | Path, dsm: np.ndarray, meta: RasterMetadata, *, model: str, calibration: str,
               is_metric: bool, vertical_datum: str | None = None,
               provenance: dict[str, Any] | None = None, calibration_dem: str | None = None) -> Path:
    """Write `dsm` on exactly `meta`'s grid, with product tags and a `<name>.json` sidecar.

    is_metric=True requires a georeferenced grid and a vertical datum; units are metres.
    is_metric=False writes an rDSM: relative units, datum "none (relative)".
    """
    if is_metric:
        if not meta.is_georeferenced:
            raise ValueError("a metric DSM needs a georeferenced input; PNG/JPG give rDSM only")
        if not vertical_datum:
            raise ValueError("a metric DSM must state its vertical datum")
        kind, units, datum = "absolute", "metres", vertical_datum
    else:
        kind, units, datum = "relative", RELATIVE_UNITS, "none (relative)"
    tags = {"MODEL": model, "CALIBRATION": calibration, "IS_METRIC": str(bool(is_metric)).lower()}
    if calibration_dem:                 # which DEM the heights were tied to (Copernicus / SRTM / ...)
        tags["CALIBRATION_DEM"] = calibration_dem
    path = write_dsm(path, np.asarray(dsm, dtype=np.float32), meta, kind=kind, units=units,
                     vertical_datum=datum, provenance=provenance, extra_tags=tags)
    finite = np.asarray(dsm)[np.isfinite(dsm)]
    sidecar = {
        "file": path.name, "kind": kind, "units": units, "vertical_datum": datum, **tags,
        "grid": {"crs": meta.crs_wkt, "transform": meta.transform, "width": meta.width,
                 "height": meta.height, "source": meta.source},
        "stats": ({"min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean()),
                   "nan_count": int(np.isnan(dsm).sum())} if finite.size else {"nan_count": int(np.size(dsm))}),
        "provenance": provenance or {},
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=str))
    return path
