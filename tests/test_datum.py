import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from depthwizard.calibration import datum as D

GRID_DIR = Path.home() / ".local" / "share" / "proj"
HAVE_GRIDS = all((GRID_DIR / g).exists() for g in ("us_nga_egm08_25.tif", "us_nga_egm96_15.tif"))


@unittest.skipUnless(HAVE_GRIDS, "geoid grids not installed (see setup/geoid_grids.sha256)")
class TestDatum(unittest.TestCase):
    def test_ellipsoid_to_egm2008_matches_grid_file(self):
        # EGM2008 height = ellipsoidal height - N; read N straight from the grid at (0E, 0N)
        with rasterio.open(GRID_DIR / "us_nga_egm08_25.tif") as g:
            n = float(next(g.sample([(0.0, 0.0)]))[0])
        h = D.convert(np.array([0.0]), np.array([0.0]), np.array([0.0]), "WGS84_ELLIPSOID")[0]
        self.assertAlmostEqual(h, -n, delta=0.05)
        self.assertAlmostEqual(h, -17.23, delta=0.1)   # published EGM2008 undulation near (0, 0) is ~17.2 m

    def test_round_trip_egm96(self):
        lon, lat = np.array([78.03, -105.27, 10.0]), np.array([30.32, 40.01, -20.0])
        h = np.array([500.0, 1600.0, 0.0])
        there = D.convert(h, lon, lat, "EGM96", "EGM2008")
        back = D.convert(there, lon, lat, "EGM2008", "EGM96")
        np.testing.assert_allclose(back, h, atol=1e-3)
        self.assertTrue(np.all(np.abs(there - h) < 3.0))   # EGM96 vs EGM2008 differ by metres, not tens

    @unittest.skipUnless((GRID_DIR / "us_noaa_g2018u0.tif").exists(), "GEOID18 not installed")
    def test_navd88_to_egm2008_boulder(self):
        # NAVD88 height 0 at Boulder -> +0.17 m in EGM2008 (PROJ pipeline via GEOID18, checked by hand once)
        h = D.convert(np.array([0.0, 1600.0]), np.array([-105.272, -105.272]), np.array([40.012, 40.012]), "NAVD88")
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertAlmostEqual(float(h[0]), 0.17, delta=0.05)
        self.assertAlmostEqual(float(h[1] - h[0]), 1600.0, delta=1e-3)

    def test_identity(self):
        h = np.array([1.5, np.nan])
        out = D.convert(h, np.zeros(2), np.zeros(2), "EGM2008", "EGM2008")
        self.assertEqual(out[0], np.float32(1.5))
        self.assertTrue(np.isnan(out[1]))

    def test_parse_datum(self):
        self.assertEqual(D.parse_datum("EGM2008 geoid (per Copernicus DEM ...)"), "EGM2008")
        self.assertEqual(D.parse_datum('vertical: "WGS 84 Geoid EGM08"'), "EGM2008")
        self.assertEqual(D.parse_datum("EGM96 geoid (per tilezen/joerd ...)"), "EGM96")
        self.assertEqual(D.parse_datum("NAVD88 height (explicit in the file's compound CRS)"), "NAVD88")
        with self.assertRaises(ValueError):
            D.parse_datum("none (relative)")

    def test_ballpark_is_refused(self):
        # mean-sea-level height has no grid-based link to EGM2008: PROJ can only ballpark it
        D.CRS3D["TEST_MSL"] = "EPSG:4326+5714"
        try:
            D.transformer.cache_clear()
            with self.assertRaises(D.DatumGridMissing):
                D.transformer("TEST_MSL", "EGM2008")
        finally:
            del D.CRS3D["TEST_MSL"]
            D.transformer.cache_clear()

    def test_raster_shift_interpolation_is_exact_enough(self):
        tr = from_origin(470000.0, 4430000.0, 0.6, 0.6)   # UTM 13N, Boulder
        s = D.raster_shift(tr, "EPSG:26913", 1500, 1400, "EGM96", step_px=256)
        self.assertEqual(s.offset.shape, (1500, 1400))
        self.assertLess(s.info["max_interp_error_m"], 0.005)
        self.assertNotIn("ballpark", s.info["pipeline"].lower())


if __name__ == "__main__":
    unittest.main()


class TestRasterDatum(unittest.TestCase):
    """raster_datum_text / raster_file_to_egm2008: datum read from tag or vertical sub-CRS, never guessed."""

    def _write(self, path, crs, tags=None):
        with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=1, dtype="float32",
                           crs=crs, transform=from_origin(476000, 4430000, 2, 2)) as d:
            d.write(np.full((1, 4, 4), 1600.0, np.float32))
            if tags:
                d.update_tags(**tags)

    def test_tag_wins_and_compound_crs_is_read(self):
        from rasterio.crs import CRS
        self.assertEqual(D.raster_datum_text({"VERTICAL_DATUM": "EGM96 geoid"}, CRS.from_epsg(32613)), "EGM96 geoid")
        self.assertIn("NAVD88", D.raster_datum_text({}, "EPSG:26913+5703"))

    def test_no_vertical_info_is_refused(self):
        # WKT of a plain projected CRS contains ELLIPSOID/SPHEROID; it must NOT be read as an ellipsoidal datum
        from rasterio.crs import CRS
        with self.assertRaises(ValueError):
            D.raster_datum_text({}, CRS.from_epsg(32613))

    @unittest.skipUnless(HAVE_GRIDS and (GRID_DIR / "us_noaa_g2018u0.tif").exists(), "geoid grids not installed")
    def test_file_conversion_keeps_grid_and_drops_old_vertical_crs(self):
        import tempfile
        from rasterio.crs import CRS
        with tempfile.TemporaryDirectory() as t:
            src, out = Path(t) / "in.tif", Path(t) / "out.tif"
            self._write(src, CRS.from_user_input("EPSG:26913+5703"))
            _, info = D.raster_file_to_egm2008(src, out)
            with rasterio.open(src) as a, rasterio.open(out) as b:
                self.assertEqual((a.transform, a.shape), (b.transform, b.shape))
                self.assertEqual(b.crs.to_epsg(), 26913)                     # horizontal only
                self.assertTrue(b.tags()["VERTICAL_DATUM"].startswith("EGM2008"))
                shift = float(b.read(1).mean() - 1600.0)
            self.assertEqual(info["source_datum"], "NAVD88")
            self.assertAlmostEqual(shift, 0.17, delta=0.05)                   # Boulder, as in test_navd88_to_egm2008
