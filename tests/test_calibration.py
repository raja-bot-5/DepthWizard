import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from depthwizard.calibration import (anchor_pixels, cell_mean, checkerboard, dem_cell_ids, dem_plus_residual,
                                     lowpass, robust_affine)
from depthwizard.io import read_raster
from helpers import DEHRADUN_UTM, UTM44N, write_utm_image


class TestCalibration(unittest.TestCase):
    def test_checkerboard_splits_disjoint_and_balanced(self):
        a = checkerboard(512, 768, 256)
        self.assertAlmostEqual(a.mean(), 0.5)
        self.assertTrue(a[0, 0] and not a[0, 256] and not a[256, 0] and a[256, 256])

    def test_lowpass_nan_aware(self):
        x = np.ones((50, 50), np.float32)
        x[10:20, 10:20] = np.nan
        y = lowpass(x, 7)
        self.assertTrue(np.allclose(y[np.isfinite(y)], 1.0))

    def test_robust_affine_recovers_scale_despite_outliers(self):
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:400, 0:400]
        terrain = 1500 + 0.2 * xx + 30 * np.sin(yy / 60.0)          # metres
        r = (terrain - 1500) / 4.0 + 7.0                              # relative: a=4, b=1500-28
        dem = terrain + rng.normal(0, 0.5, terrain.shape)
        dem[rng.random(dem.shape) < 0.05] += 80                       # 5% gross outliers (trees/buildings)
        anchors = checkerboard(400, 400, 50)
        cal = robust_affine(r.astype(np.float32), dem.astype(np.float32), anchors, lowpass_px=1)
        self.assertAlmostEqual(cal.params["a"], 4.0, delta=0.05)
        test = ~anchors
        self.assertLess(np.abs(cal.z - terrain)[test].mean(), 1.0)

    def test_residual_has_zero_mean_per_cell(self):
        r = np.random.default_rng(1).normal(size=(60, 60))
        cell = (np.arange(60)[:, None] // 20) * 3 + (np.arange(60)[None, :] // 20)
        dem = np.full((60, 60), 100.0)
        cal = dem_plus_residual(r, dem, cell, scale=2.0)
        m = cell_mean(cal.z - dem, cell)
        self.assertTrue(np.allclose(m, 0.0, atol=1e-5))

    def test_dem_cell_ids_on_image_grid(self):
        tmp = Path(tempfile.mkdtemp())
        _, meta = read_raster(write_utm_image(tmp / "img.tif", width=100, height=100, res=0.6))
        demp = tmp / "dem.tif"
        with rasterio.open(demp, "w", driver="GTiff", width=2, height=2, count=1, dtype="float32",
                           crs=UTM44N, transform=from_origin(*DEHRADUN_UTM, 30, 30)) as dst:
            dst.write(np.zeros((1, 2, 2), np.float32))
        ids = dem_cell_ids(demp, meta)
        self.assertEqual(ids[0, 0], 0)
        self.assertEqual(ids[0, 99], 1)
        self.assertEqual(ids[99, 0], 2)
        self.assertEqual(ids[99, 99], 3)
        self.assertEqual(ids[49, 49], 0)  # 49*0.6 = 29.4 m < 30 m
        self.assertEqual(ids[50, 50], 3)  # 30.0 m -> next cell

    def test_anchor_mask_excludes_steep_and_test_blocks(self):
        dem = np.tile(np.arange(100, dtype=np.float32) * 1.0, (100, 1))   # 45 deg at 1 m pixels
        r = np.zeros_like(dem)
        m = anchor_pixels(dem, r, checkerboard(100, 100, 50), (1.0, 1.0), max_slope_deg=15)
        self.assertFalse(m.any())
        flat = np.zeros_like(dem)
        m2 = anchor_pixels(flat, r, checkerboard(100, 100, 50), (1.0, 1.0), max_slope_deg=15)
        self.assertTrue(m2[:50, :50].all() and not m2[:50, 50:].any())


if __name__ == "__main__":
    unittest.main()


class TestSmoothResidualAndDownsample(unittest.TestCase):
    def test_smooth_residual_has_no_cell_edge_steps(self):
        from depthwizard.calibration import dem_plus_smooth_residual
        r = np.tile(np.linspace(0, 10, 120), (120, 1))          # smooth ramp: no real fine structure
        cell = (np.arange(120)[:, None] // 40) * 3 + (np.arange(120)[None, :] // 40)
        dem = np.zeros((120, 120))
        per_cell = dem_plus_residual(r, dem, cell, 1.0).z
        smooth = dem_plus_smooth_residual(r.astype(np.float32), dem, lowpass_px=41, scale=1.0).z
        step = lambda z: np.abs(np.diff(z[60, 30:50])).max()      # across the cell edge at col 40
        self.assertGreater(step(per_cell), 5 * step(smooth))
        self.assertLess(np.abs(smooth[:, 30:90]).max(), 0.05)    # interior: ramp removed entirely

    def test_downsample_rescales_transform(self):
        from depthwizard.io import downsample_geotiff, read_raster
        tmp = Path(tempfile.mkdtemp())
        src = write_utm_image(tmp / "img.tif", width=201, height=150, res=0.6)
        out = downsample_geotiff(src, 2, tmp / "img_x2.tif")
        a, ma = read_raster(src)
        b, mb = read_raster(out)
        self.assertEqual((mb.width, mb.height), (100, 75))
        self.assertAlmostEqual(mb.affine.a, 1.2)
        self.assertEqual(mb.affine.c, ma.affine.c)
        self.assertEqual(mb.affine.f, ma.affine.f)
        self.assertEqual(mb.crs, ma.crs)
        # value = rounded mean of the 2x2 block it covers
        self.assertEqual(int(b[0, 3, 5]), int(np.rint(a[0, 6:8, 10:12].astype(float).mean())))
