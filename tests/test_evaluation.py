import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from depthwizard.evaluation import (affine_fitted_metrics, evaluate_rasters, metric_res, metrics,
                                    slope_classes)
from helpers import DEHRADUN_UTM, UTM44N, plane


def write(path, arr, crs, transform, datum=None):
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan) as dst:
        dst.write(arr.astype(np.float32), 1)
        if datum:
            dst.update_tags(VERTICAL_DATUM=datum)
    return path


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.ref = np.random.default_rng(0).normal(600, 20, (60, 60))

    def test_constant_offset(self):
        m = metrics(self.ref + 2.0, self.ref)
        self.assertAlmostEqual(m["bias"], 2.0)
        self.assertAlmostEqual(m["mae"], 2.0)
        self.assertAlmostEqual(m["rmse"], 2.0)
        self.assertAlmostEqual(m["nmad"], 0.0)
        self.assertAlmostEqual(m["pearson_r"], 1.0)

    def test_symmetric_noise(self):
        noise = np.where(np.indices(self.ref.shape).sum(0) % 2 == 0, 1.0, -1.0)
        m = metrics(self.ref + noise, self.ref)
        self.assertAlmostEqual(m["bias"], 0.0)
        self.assertAlmostEqual(m["mae"], 1.0)
        self.assertAlmostEqual(m["rmse"], 1.0)

    def test_nodata_excluded_and_counted(self):
        pred = self.ref.copy()
        pred[:10] = np.nan
        ref = self.ref.copy()
        ref[:, :5] = np.nan
        m = metrics(pred + 1, ref)
        self.assertEqual(m["n"], 50 * 55)
        self.assertEqual(m["excluded_nodata"], 3600 - 50 * 55)
        self.assertAlmostEqual(m["rmse"], 1.0)

    def test_hand_computed_small_example(self):
        # pred [1,2,3,4] vs ref [1,3,2,5]: d = [0,-1,1,-1]
        m = metrics(np.array([1.0, 2, 3, 4]), np.array([1.0, 3, 2, 5]))
        self.assertAlmostEqual(m["bias"], -0.25, places=12)
        self.assertAlmostEqual(m["mae"], 0.75, places=12)
        self.assertAlmostEqual(m["rmse"], np.sqrt(3 / 4), places=12)
        # SS_res = 3, SS_tot = sum((ref - 2.75)^2) = 8.75
        self.assertAlmostEqual(m["r2"], 1 - 3 / 8.75, places=12)
        # cov sum 5.5, SS_pred 5, SS_ref 8.75
        self.assertAlmostEqual(m["pearson_r"], 5.5 / np.sqrt(5 * 8.75), places=12)

    def test_le90_and_nmad_hand_computed(self):
        e = np.arange(1, 11, dtype=float)              # |errors| 1..10
        m = metrics(e, np.zeros(10))
        self.assertAlmostEqual(m["le90"], np.percentile(e, 90))   # 9.1 with linear interpolation
        self.assertAlmostEqual(m["le90"], 9.1)
        # NMAD = 1.4826 * median(|e - median(e)|) = 1.4826 * 2.5
        self.assertAlmostEqual(m["nmad"], 1.4826 * 2.5)

    def test_r2_can_be_negative(self):
        ref = np.array([1.0, 2, 3, 4])
        self.assertLess(metrics(ref[::-1] + 10, ref)["r2"], 0)

    def test_all_nodata(self):
        self.assertEqual(metrics(np.full((3, 3), np.nan), np.zeros((3, 3)))["n"], 0)

    def test_affine_fit_is_flagged_optimistic(self):
        m = affine_fitted_metrics(0.5 * self.ref + 3.0, self.ref)
        self.assertLess(m["rmse"], 1e-6)
        self.assertAlmostEqual(m["fit_a"], 2.0, places=6)
        self.assertIn("OPTIMISTIC", m["warning"])


class TestSlope(unittest.TestCase):
    def test_known_slope_class(self):
        x = np.arange(50) * 2.0
        dem = np.tile(np.tan(np.radians(10)) * x, (40, 1))  # 10 deg -> moderate
        cls = slope_classes(dem, 2.0, 2.0)
        self.assertTrue((cls == 1).all())

    def test_geographic_res_in_metres(self):
        rx, ry = metric_res(from_origin(78.0, 30.5, 1 / 3600, 1 / 3600), rasterio.crs.CRS.from_epsg(4326), 100)
        self.assertAlmostEqual(rx, 26.7, delta=0.3)   # 1 arc-sec of longitude at ~30.5N
        self.assertAlmostEqual(ry, 30.8, delta=0.3)


class TestEvaluateRasters(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_native_refuses_grid_mismatch(self):
        a = write(self.tmp / "a.tif", np.zeros((10, 10)), UTM44N, from_origin(*DEHRADUN_UTM, 1, 1))
        b = write(self.tmp / "b.tif", np.zeros((10, 10)), UTM44N, from_origin(*DEHRADUN_UTM, 2, 2))
        with self.assertRaises(ValueError):
            evaluate_rasters(a, b, mode="native")

    def test_aggregate_disjoint_extents_raise(self):
        p = write(self.tmp / "p.tif", np.ones((100, 100)), UTM44N, from_origin(*DEHRADUN_UTM, 0.6, 0.6))
        x0, y0 = DEHRADUN_UTM
        r = write(self.tmp / "r.tif", np.ones((3, 3)), UTM44N, from_origin(x0 + 50_000, y0, 30, 30))
        with self.assertRaisesRegex(ValueError, "no reference cell lies inside"):
            evaluate_rasters(p, r, mode="aggregate")

    def test_missing_crs_raises(self):
        a = self.tmp / "nocrs.tif"
        with rasterio.open(a, "w", driver="GTiff", width=10, height=10, count=1, dtype="float32") as dst:
            dst.write(np.zeros((1, 10, 10), np.float32))
        b = write(self.tmp / "b.tif", np.zeros((10, 10)), UTM44N, from_origin(*DEHRADUN_UTM, 1, 1))
        with self.assertRaisesRegex(ValueError, "need a CRS"):
            evaluate_rasters(a, b)

    def test_native_and_datum_check(self):
        ref = np.random.default_rng(1).normal(600, 5, (40, 40))
        tr = from_origin(*DEHRADUN_UTM, 1, 1)
        r = write(self.tmp / "r.tif", ref, UTM44N, tr, datum="EGM2008")
        p1 = write(self.tmp / "p1.tif", ref + 1, UTM44N, tr, datum="EGM2008")
        p2 = write(self.tmp / "p2.tif", ref + 1, UTM44N, tr, datum="EGM96")
        p3 = write(self.tmp / "p3.tif", ref + 1, UTM44N, tr)
        rep = evaluate_rasters(p1, r)
        self.assertAlmostEqual(rep["overall"]["bias"], 1.0, places=5)
        self.assertEqual(rep["vertical_datum"]["check"], "MATCH")
        self.assertTrue(evaluate_rasters(p2, r)["vertical_datum"]["check"].startswith("MISMATCH"))
        self.assertTrue(evaluate_rasters(p3, r)["vertical_datum"]["check"].startswith("UNKNOWN"))
        from depthwizard.evaluation import datum_check
        self.assertEqual(datum_check("EGM2008", "EGM2008 (converted from NAVD88)"), "MATCH")   # datum, not string
        self.assertTrue(datum_check("EGM2008", "NAVD88 height").startswith("MISMATCH"))
        self.assertTrue(datum_check("EGM2008", "local mean sea level").startswith("UNKNOWN"))

    def test_aggregate_same_crs_block_means_and_edge_cells(self):
        # 0.6 m pred, 100x100 px = 60 m; 30 m ref cells = exactly 50x50 px blocks
        pred = np.zeros((100, 100))
        pred[:50, :50], pred[:50, 50:], pred[50:, :50], pred[50:, 50:] = 10, 20, 30, 40
        pred += np.random.default_rng(2).normal(0, 1, pred.shape)  # zero-mean texture
        p = write(self.tmp / "p.tif", pred, UTM44N, from_origin(*DEHRADUN_UTM, 0.6, 0.6))
        ref = np.array([[10.0, 20.0], [30.0, 40.0]])
        r = write(self.tmp / "r.tif", ref, UTM44N, from_origin(*DEHRADUN_UTM, 30, 30))
        rep = evaluate_rasters(p, r, mode="aggregate")
        self.assertEqual(rep["overall"]["n"], 4)
        self.assertLess(rep["overall"]["mae"], 0.1)
        # shift the ref grid by half a cell: only cells fully inside the 60 m footprint may count
        x0, y0 = DEHRADUN_UTM
        r2 = write(self.tmp / "r2.tif", np.full((3, 3), 25.0), UTM44N, from_origin(x0 - 15, y0 + 15, 30, 30))
        rep2 = evaluate_rasters(p, r2, mode="aggregate")
        self.assertEqual(rep2["overall"]["n"], 1)  # the centre cell only
        self.assertEqual(rep2["aggregation"]["cells_dropped_low_coverage"], 8)

    def test_aggregate_utm_pred_onto_geographic_ref(self):
        """Cartosat-like 0.6 m UTM prediction vs a Copernicus-like 1 arc-sec EPSG:4326 reference."""
        n, res = 2000, 0.6
        tr = from_origin(*DEHRADUN_UTM, res, res)
        rows, cols = np.mgrid[0:n, 0:n]
        x, y = tr * (cols + 0.5, rows + 0.5)
        lon, lat = Transformer.from_crs(UTM44N, "EPSG:4326", always_xy=True).transform(x, y)
        p = write(self.tmp / "p.tif", plane(lon, lat), UTM44N, tr, datum="EGM2008")
        d = 1 / 3600
        left, top = 78.02 - d / 2, 30.33 + d / 2
        rr, cc = np.mgrid[0:72, 0:72]
        rlon, rlat = left + (cc + 0.5) * d, top - (rr + 0.5) * d
        r = write(self.tmp / "r.tif", plane(rlon, rlat), "EPSG:4326", from_origin(left, top, d, d), datum="EGM2008")
        rep = evaluate_rasters(p, r, mode="aggregate")
        self.assertGreater(rep["overall"]["n"], 1000)
        self.assertLess(rep["overall"]["rmse"], 0.05)  # averaging a plane gives its centre value
        self.assertIn("flat (<5 deg)", rep["by_slope"])


if __name__ == "__main__":
    unittest.main()
