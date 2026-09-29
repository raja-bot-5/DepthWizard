import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from depthwizard.export import RELATIVE_UNITS, export_dsm
from depthwizard.geo.dem import skadi_tile_names
from depthwizard.io import read_raster
from depthwizard.tiling import plan_tiles
from helpers import write_utm_image


class TestExport(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _, self.geo = read_raster(write_utm_image(self.tmp / "img.tif"))
        p = self.tmp / "img.png"
        Image.fromarray(np.zeros((150, 200, 3), np.uint8)).save(p)
        _, self.png = read_raster(p)

    def test_rdsm_tags_and_sidecar(self):
        out = export_dsm(self.tmp / "r.tif", np.ones((150, 200)), self.geo, model="da3_mono_large",
                         calibration="none", is_metric=False)
        with rasterio.open(out) as src:
            t = src.tags()
            self.assertEqual(src.transform, self.geo.affine)
            self.assertEqual(src.crs, self.geo.crs)
        self.assertEqual(t["DSM_KIND"], "relative")
        self.assertEqual(t["UNITS"], RELATIVE_UNITS)
        self.assertEqual(t["IS_METRIC"], "false")
        self.assertEqual(t["MODEL"], "da3_mono_large")
        side = json.loads(out.with_suffix(".json").read_text())
        self.assertEqual(side["kind"], "relative")

    def test_png_can_never_be_metric(self):
        with self.assertRaises(ValueError):
            export_dsm(self.tmp / "m.tif", np.ones((150, 200)), self.png, model="x", calibration="affine",
                       is_metric=True, vertical_datum="EGM2008")

    def test_png_rdsm_written_without_georef(self):
        out = export_dsm(self.tmp / "p.tif", np.ones((150, 200)), self.png, model="x", calibration="none",
                         is_metric=False)
        with rasterio.open(out) as src:
            self.assertIsNone(src.crs)

    def test_calibration_dem_tag(self):
        out = export_dsm(self.tmp / "d.tif", np.ones((150, 200)), self.geo, model="x", calibration="M2",
                         is_metric=True, vertical_datum="EGM2008", calibration_dem="NASADEM HGT v001")
        with rasterio.open(out) as src:
            self.assertEqual(src.tags()["CALIBRATION_DEM"], "NASADEM HGT v001")
        self.assertEqual(json.loads(out.with_suffix(".json").read_text())["CALIBRATION_DEM"], "NASADEM HGT v001")

    def test_metric_needs_datum(self):
        with self.assertRaises(ValueError):
            export_dsm(self.tmp / "m.tif", np.ones((150, 200)), self.geo, model="x", calibration="dem",
                       is_metric=True)


class TestTileTransform(unittest.TestCase):
    def test_tile_transform_matches_window(self):
        _, meta = read_raster(write_utm_image(Path(tempfile.mkdtemp()) / "img.tif", width=900, height=700))
        for t in plan_tiles(700, 900, 256, 48):
            tr = t.transform(meta.affine)
            x, y = tr * (0, 0)
            self.assertEqual((x, y), meta.pixel_to_geo(t.row0, t.col0))
            self.assertEqual((tr.a, tr.e), (meta.affine.a, meta.affine.e))  # same pixel size: crop, not resample


class TestSkadiNames(unittest.TestCase):
    def test_names(self):
        self.assertEqual(skadi_tile_names((-105.28, 40.00, -105.26, 40.02)), ["N40/N40W106"])
        self.assertEqual(skadi_tile_names((78.0, 30.2, 78.1, 30.4)), ["N30/N30E078"])
        self.assertEqual(skadi_tile_names((-71.5, -0.5, -71.2, -0.2)), ["S01/S01W072"])


if __name__ == "__main__":
    unittest.main()
