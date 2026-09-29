import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from depthwizard.io import read_raster, write_dsm
from helpers import UTM44N, write_utm_image


class TestReadRaster(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_geotiff_keeps_crs_and_transform(self):
        p = write_utm_image(self.tmp / "img.tif")
        data, meta = read_raster(p)
        self.assertEqual(data.shape, (3, 150, 200))
        self.assertTrue(meta.is_georeferenced)
        self.assertEqual(meta.crs, rasterio.crs.CRS.from_string(UTM44N))
        with rasterio.open(p) as src:
            self.assertEqual(meta.affine, src.transform)
            self.assertEqual(meta.bounds, tuple(src.bounds))

    def test_pixel_geo_roundtrip(self):
        _, meta = read_raster(write_utm_image(self.tmp / "img.tif"))
        x, y = meta.pixel_to_geo(10.5, 20.5)
        self.assertAlmostEqual(x, 214_400.0 + 20.5 * 0.6)
        self.assertAlmostEqual(y, 3_358_000.0 - 10.5 * 0.6)
        r, c = meta.geo_to_pixel(x, y)
        self.assertAlmostEqual(r, 10.5)
        self.assertAlmostEqual(c, 20.5)

    def test_png_is_not_georeferenced(self):
        p = self.tmp / "img.png"
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(p)
        _, meta = read_raster(p)
        self.assertFalse(meta.is_georeferenced)
        with self.assertRaises(ValueError):
            meta.affine


class TestWriteDsm(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _, self.meta = read_raster(write_utm_image(self.tmp / "img.tif"))

    def test_roundtrip_grid_and_tags(self):
        dsm = np.random.default_rng(1).normal(500, 10, (150, 200)).astype(np.float32)
        dsm[0, 0] = np.nan
        out = write_dsm(self.tmp / "dsm.tif", dsm, self.meta, kind="absolute", units="metres",
                        vertical_datum="EGM2008", provenance={"test": True})
        with rasterio.open(out) as src:
            self.assertEqual(src.transform, self.meta.affine)
            self.assertEqual(src.crs, self.meta.crs)
            self.assertEqual(src.dtypes[0], "float32")
            tags = src.tags()
            np.testing.assert_array_equal(src.read(1), dsm)
        self.assertEqual(tags["DSM_KIND"], "absolute")
        self.assertEqual(tags["VERTICAL_DATUM"], "EGM2008")

    def test_relative_dsm_cannot_be_metres(self):
        with self.assertRaises(ValueError):
            write_dsm(self.tmp / "r.tif", np.zeros((150, 200), np.float32), self.meta,
                      kind="relative", units="metres", vertical_datum="none")

    def test_refuses_shape_mismatch(self):
        # a resized DSM must never be written with the original transform
        with self.assertRaises(ValueError):
            write_dsm(self.tmp / "bad.tif", np.zeros((75, 100), np.float32), self.meta,
                      kind="absolute", units="metres", vertical_datum="EGM2008")


if __name__ == "__main__":
    unittest.main()
