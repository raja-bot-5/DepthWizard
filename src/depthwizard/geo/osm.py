"""OpenStreetMap building footprints (ODbL): fetch for a bounding box, cache, rasterise onto any grid.

Source: the OSM API 0.6 `map` call (light use per the OSM API usage policy; one call per area, cached).
Overpass mirrors were tried first and timed out. Buildings are closed ways tagged `building=*`;
multipolygon relations are counted and reported as skipped (not silently lost).
Attribution required: "© OpenStreetMap contributors, ODbL 1.0".
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from defusedxml import ElementTree as ET   # safe against XXE / entity-expansion attacks

ATTRIBUTION = "© OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)"
API = "https://api.openstreetmap.org/api/0.6/map?bbox={w:.6f},{s:.6f},{e:.6f},{n:.6f}"


def parse_osm_buildings(xml_bytes: bytes) -> tuple[list[dict[str, Any]], dict[str, int]]:
    root = ET.fromstring(xml_bytes)
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    out, skipped_rel, open_ways = [], 0, 0
    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.iter("tag")}
        if "building" not in tags:
            continue
        refs = [nd.get("ref") for nd in way.iter("nd")]
        if len(refs) < 4 or refs[0] != refs[-1] or any(r not in nodes for r in refs):
            open_ways += 1
            continue
        out.append({"id": int(way.get("id")), "coords": [nodes[r] for r in refs],
                    "building": tags.get("building"), "height_tag": tags.get("height"),
                    "levels_tag": tags.get("building:levels")})
    for rel in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in rel.iter("tag")}
        if "building" in tags:
            skipped_rel += 1
    return out, {"ways": len(out), "skipped_multipolygon_relations": skipped_rel, "skipped_open_or_incomplete": open_ways}


def fetch_buildings(bounds_wgs84: tuple[float, float, float, float], cache_dir: str | Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    w, s, e, n = bounds_wgs84
    key = hashlib.sha1(f"{w:.6f},{s:.6f},{e:.6f},{n:.6f}".encode()).hexdigest()[:16]
    path = cache_dir / f"osm_buildings_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    url = API.format(w=w, s=s, e=e, n=n)
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "DepthWizard-research/0.1"}),
                                 timeout=120).read()
    buildings, counts = parse_osm_buildings(raw)
    rec = {"source": url, "attribution": ATTRIBUTION, "license": "ODbL-1.0",
           "fetched_utc": datetime.now(timezone.utc).isoformat(), "bbox_wgs84": list(bounds_wgs84),
           "counts": counts, "xml_sha256": hashlib.sha256(raw).hexdigest(), "buildings": buildings}
    path.write_text(json.dumps(rec))
    return rec


def rasterize(buildings: list[dict[str, Any]], transform, crs, shape: tuple[int, int],
              min_area_m2: float = 0.0) -> tuple[np.ndarray, dict[int, int]]:
    """Burn footprints into an int32 raster (0 = none, k = 1..N). Returns (raster, {k: osm_id})."""
    from pyproj import Transformer
    from rasterio.features import rasterize as rio_rasterize
    to = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    shapes, ids = [], {}
    for k, b in enumerate(buildings, start=1):
        xs, ys = to.transform(*zip(*b["coords"]))
        xs, ys = np.asarray(xs), np.asarray(ys)
        area = 0.5 * abs(np.dot(xs[:-1], ys[1:]) - np.dot(xs[1:], ys[:-1]))      # shoelace
        if not np.all(np.isfinite(xs)) or area < max(min_area_m2, 1e-6):
            continue
        shapes.append(({"type": "Polygon", "coordinates": [list(zip(xs.tolist(), ys.tolist()))]}, k))
        ids[k] = b["id"]
    if not shapes:
        return np.zeros(shape, np.int32), {}
    r = rio_rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")
    return r, ids
