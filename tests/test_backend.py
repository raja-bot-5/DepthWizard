import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from backend.app import create_app  # noqa: E402
from depthwizard.geo.dem import DEMSource  # noqa: E402
from depthwizard.pipeline import PipelineConfig  # noqa: E402
from helpers import write_fake_tile, write_utm_image  # noqa: E402


class FakePredictor:
    """Stand-in for a depth model: 'height' = red channel. No weights, CPU, instant."""
    height_sign = +1
    polarity = "fake (red channel)"

    def __call__(self, crop):
        return crop[..., 0].astype(np.float32)


class TestBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        tiles = cls.tmp / "tiles"
        tiles.mkdir()
        for lat, lon in ((30, 77), (30, 78)):
            write_fake_tile(tiles / f"T_{lat}_{lon}.tif", lat, lon)
        dem = DEMSource(name="fake", url_template=str(tiles / "{tile}.tif"),
                        tile_names=lambda b: [f"T_30_{lon}" for lon in range(int(np.floor(b[0])), int(np.ceil(b[2])))],
                        horizontal_crs="EPSG:4326", vertical_datum="EGM2008 (test fixture)", units="metres",
                        pixel_convention="edge", reference="test")
        cfg = PipelineConfig(device="cpu", tile=256, overlap=32, dem_cache=str(cls.tmp / "cache"), mesh_max_side=64)
        cls.client = TestClient(create_app(cls.tmp / "jobs", cfg, {"predictor": FakePredictor(), "dem_source": dem}))
        cls.geotiff = write_utm_image(cls.tmp / "scene.tif", width=300, height=240)
        cls.png = cls.tmp / "photo.png"
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(cls.png)

    def submit(self, path: Path, **form) -> str:
        with open(path, "rb") as f:
            r = self.client.post("/jobs", files={"file": (path.name, f)}, data=form)
        self.assertEqual(r.status_code, 202, r.text)
        return r.json()["id"]

    def wait(self, jid: str, timeout: float = 60) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.client.get(f"/jobs/{jid}").json()
            if j["status"] in ("done", "failed"):
                return j
            time.sleep(0.1)
        self.fail("job timed out")

    def test_health(self):
        r = self.client.get("/health").json()
        self.assertTrue(r["ok"])
        self.assertIn("dem_plus_residual", r["calibrations"])

    def test_geotiff_round_trip(self):
        jid = self.submit(self.geotiff)
        j = self.wait(jid)
        self.assertEqual(j["status"], "done", j.get("error"))
        md = j["metadata"]
        self.assertTrue(md["input"]["georeferenced"])
        self.assertEqual(md["product"]["kind"], "metric DSM")
        self.assertEqual(md["product"]["vertical_datum"], "EGM2008")
        self.assertEqual(md["tiling"]["align"], "sequential")        # metric path: validated tiling (Phase 11 T3)
        r = self.client.get(f"/jobs/{jid}/files/dsm.tif")
        self.assertEqual(r.status_code, 200)
        out = self.tmp / f"{jid}_dsm.tif"
        out.write_bytes(r.content)
        with rasterio.open(out) as a, rasterio.open(self.geotiff) as b:
            self.assertEqual((a.crs, a.transform, a.shape), (b.crs, b.transform, b.shape))
            self.assertEqual(a.tags()["IS_METRIC"], "true")
        for name in ("mesh.glb", "texture.jpg", "calibration.json", "rdsm.tif", "ground.tif", "dem.tif"):
            self.assertEqual(self.client.get(f"/jobs/{jid}/files/{name}").status_code, 200, name)
        glb = self.client.get(f"/jobs/{jid}/files/mesh.glb").content
        self.assertEqual(glb[:4], b"glTF")
        p = self.client.get(f"/jobs/{jid}/point", params={"row": 100, "col": 150}).json()
        self.assertEqual(p["kind"], "absolute")
        self.assertIsInstance(p["value"], float)
        self.assertIn("derived_height_above_ground", p)
        conf = p["derived_height_confidence"]                      # every derived height carries a level
        self.assertIn(conf["level"], ("low", "medium"))
        self.assertIsInstance(conf["reasons"], list)
        self.assertIn("rmse_m", conf["measured_error"])             # the UI quotes the measured error per level
        self.assertIn("DERIVED", p["derived_note"])

    def test_stage_events_stream_and_previews(self):
        jid = self.submit(self.geotiff)
        self.assertEqual(self.wait(jid)["status"], "done")
        evs = self.client.get(f"/jobs/{jid}/events").json()["events"]
        self.assertEqual([e["seq"] for e in evs], list(range(len(evs))))           # numbered, gap-free
        done = {e["stage"] for e in evs if e["type"] == "stage_done"}
        self.assertTrue({"input", "tiling", "depth", "dem", "calibration", "dsm", "mesh"} <= done, done)
        self.assertEqual(evs[-1]["type"], "done")
        later = self.client.get(f"/jobs/{jid}/events", params={"after": evs[3]["seq"]}).json()["events"]
        self.assertEqual(later[0]["seq"], evs[4]["seq"])
        # every layer named by an event has a servable preview (what the processing view shows)
        layers = {x["id"]: x for x in self.client.get(f"/jobs/{jid}/files/layers.json").json()}
        for e in evs:
            for lid in e.get("layers", []):
                r = self.client.get(f"/jobs/{jid}/files/{layers[lid]['preview']}")
                self.assertEqual(r.status_code, 200, lid)
        self.assertIn("hillshade", layers)
        with self.client.stream("GET", f"/jobs/{jid}/stream") as r:
            body = "".join(r.iter_text())
        self.assertIn("event: end", body)
        self.assertEqual(body.count("\ndata: ") + body.startswith("data: "), len(evs) + 1)
        # previews are pattern-matched names, never paths
        self.assertEqual(self.client.get(f"/jobs/{jid}/files/layer_..%2Fjob.json").status_code, 404)

    def test_frontend_served_offline(self):
        r = self.client.get("/app/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("importmap", r.text)
        self.assertNotIn("https://", r.text)  # no CDN / external resources in the page
        for f in ("app.js", "style.css", "vendor/three/build/three.module.js", "vendor/fonts/barlow-latin-400-normal.woff2"):
            self.assertEqual(self.client.get(f"/app/{f}").status_code, 200, f)

    def test_reference_comparison(self):
        jid = self.submit(self.geotiff)
        self.assertEqual(self.wait(jid)["status"], "done")
        for f in ("slope.png", "mesh_lod1.glb"):
            self.assertEqual(self.client.get(f"/jobs/{jid}/files/{f}").status_code, 200, f)
        dsm = self.tmp / f"{jid}_ref_src.tif"
        dsm.write_bytes(self.client.get(f"/jobs/{jid}/files/dsm.tif").content)
        ref = self.tmp / f"{jid}_ref.tif"
        with rasterio.open(dsm) as s:
            prof = s.profile
            z = s.read(1)
        with rasterio.open(ref, "w", **prof) as d:
            d.write(z - 2.0, 1)            # reference 2 m lower -> bias +2 m on the same grid
            d.update_tags(VERTICAL_DATUM="EGM2008")
        with open(ref, "rb") as f:
            r = self.client.post(f"/jobs/{jid}/reference", files={"file": ("ref.tif", f)})
        self.assertEqual(r.status_code, 200, r.text)
        rep = r.json()
        self.assertAlmostEqual(rep["on_dsm_grid"]["bias"], 2.0, places=3)
        self.assertEqual(rep["datum_check"], "MATCH")
        self.assertEqual(self.client.get(f"/jobs/{jid}/files/reference_diff.png").status_code, 200)

    def test_png_gives_relative_only(self):
        jid = self.submit(self.png)
        j = self.wait(jid)
        self.assertEqual(j["status"], "done", j.get("error"))
        self.assertFalse(j["metadata"]["input"]["georeferenced"])
        self.assertEqual(j["metadata"]["product"]["kind"], "relative DSM (rDSM)")
        self.assertEqual(j["metadata"]["tiling"]["align"], "joint_plane")   # seam-free relative mosaic
        self.assertEqual(self.client.get(f"/jobs/{jid}/files/dsm.tif").status_code, 404)
        self.assertEqual(self.client.get(f"/jobs/{jid}/files/rdsm.tif").status_code, 200)
        p = self.client.get(f"/jobs/{jid}/point", params={"row": 10, "col": 10}).json()
        self.assertEqual(p["kind"], "relative")
        self.assertNotIn("derived_height_above_ground", p)
        self.assertNotEqual(p["units"], "metres")

    def test_rejections(self):
        bad = self.tmp / "x.exe"
        bad.write_bytes(b"MZ")
        with open(bad, "rb") as f:
            self.assertEqual(self.client.post("/jobs", files={"file": ("x.exe", f)}).status_code, 415)
        with open(self.png, "rb") as f:
            self.assertEqual(self.client.post("/jobs", files={"file": ("p.png", f)},
                                              data={"model": "dav2_large"}).status_code, 422)
        self.assertEqual(self.client.get("/jobs/" + "0" * 32).status_code, 404)
        self.assertEqual(self.client.get("/jobs/../../etc/passwd").status_code, 404)
        jid = self.submit(self.png)
        self.wait(jid)
        self.assertEqual(self.client.get(f"/jobs/{jid}/files/job.json").status_code, 404)
        self.assertEqual(self.client.get(f"/jobs/{jid}/point", params={"row": 99999, "col": 0}).status_code, 422)


if __name__ == "__main__":
    unittest.main()


class TestPipelineCalibrationChoice(unittest.TestCase):
    def test_every_calibration_option_is_honoured(self):
        from depthwizard.pipeline import CALIBRATIONS, run_pipeline
        tmp = Path(tempfile.mkdtemp())
        tiles = tmp / "tiles"; tiles.mkdir()
        for lat, lon in ((30, 77), (30, 78)):
            write_fake_tile(tiles / f"T_{lat}_{lon}.tif", lat, lon)
        dem = DEMSource(name="fake", url_template=str(tiles / "{tile}.tif"),
                        tile_names=lambda b: [f"T_30_{lon}" for lon in range(int(np.floor(b[0])), int(np.ceil(b[2])))],
                        horizontal_crs="EPSG:4326", vertical_datum="EGM2008 (test fixture)", units="metres",
                        pixel_convention="edge", reference="test")
        img = write_utm_image(tmp / "s.tif", width=300, height=240)
        want = {"dem_only": "M0_dem_only", "robust_affine": "M1_global_affine",
                "dem_plus_residual": "M2_dem_residual_cell", "dem_plus_smooth_residual": "M2_dem_residual_smooth"}
        for c in CALIBRATIONS:
            md = run_pipeline(img, tmp / c, PipelineConfig(device="cpu", tile=256, overlap=32, calibration=c,
                              dem_cache=str(tmp / "cache"), mesh_max_side=32), predictor=FakePredictor(), dem_source=dem)
            self.assertEqual(md["product"]["calibration"], want[c], c)
