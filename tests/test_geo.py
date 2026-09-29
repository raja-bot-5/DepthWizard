import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

from depthwizard.geo import copernicus_tile_names, dem_to_image_grid, fetch_dem_window, footprint_wgs84
from depthwizard.geo.dem import COPERNICUS_GLO30, DEMSource, _snap_outward
from depthwizard.io import read_raster
from helpers import UTM44N, plane, write_fake_tile, write_utm_image


class TestTileNames(unittest.TestCase):
    def test_single_tile(self):
        self.assertEqual(copernicus_tile_names((77.2, 28.3, 77.4, 28.5)),
                         ["Copernicus_DSM_COG_10_N28_00_E077_00_DEM"])

    def test_corner_crossing_gives_four(self):
        self.assertEqual(len(copernicus_tile_names((77.9, 28.9, 78.1, 29.1))), 4)

    def test_exact_degree_edges_do_not_add_tiles(self):
        self.assertEqual(copernicus_tile_names((77.0, 28.0, 78.0, 29.0)),
                         ["Copernicus_DSM_COG_10_N28_00_E077_00_DEM"])

    def test_south_west_hemispheres(self):
        self.assertEqual(copernicus_tile_names((-71.5, -0.5, -71.2, -0.2)),
                         ["Copernicus_DSM_COG_10_S01_00_W072_00_DEM"])


class TestFootprint(unittest.TestCase):
    def test_contains_image_corners(self):
        tmp = Path(tempfile.mkdtemp())
        _, meta = read_raster(write_utm_image(tmp / "img.tif"))
        w, s, e, n = footprint_wgs84(meta)
        to_ll = Transformer.from_crs(UTM44N, "EPSG:4326", always_xy=True)
        l, b, r, t = meta.bounds
        for x, y in ((l, b), (l, t), (r, b), (r, t)):
            lon, lat = to_ll.transform(x, y)
            self.assertTrue(w <= lon <= e and s <= lat <= n)
        self.assertTrue(77.5 < w < 78.5 and 30 < s < 31)  # Dehradun sanity


class TestSnap(unittest.TestCase):
    def test_snaps_to_half_offset_grid(self):
        res = 1 / 3600
        origin = 77 - res / 2
        lo, hi = _snap_outward(77.10001, 77.20001, origin, res, pad_px=0)
        for v in (lo, hi):
            k = (v - origin) / res
            self.assertAlmostEqual(k, round(k), places=6)
        self.assertLessEqual(lo, 77.10001)
        self.assertGreaterEqual(hi, 77.20001)


class TestFetchAndReproject(unittest.TestCase):
    """Local fake tiles with Copernicus' half-pixel grid offset stand in for the network."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        tiles_dir = self.tmp / "tiles"
        tiles_dir.mkdir()
        for lat, lon in ((30, 77), (30, 78)):
            write_fake_tile(tiles_dir / f"T_{lat}_{lon}.tif", lat, lon)
        self.source = DEMSource(
            name="fake", url_template=str(tiles_dir / "{tile}.tif"),
            tile_names=lambda b: [f"T_30_{lon}" for lon in range(int(np.floor(b[0])), int(np.ceil(b[2])))],
            horizontal_crs="EPSG:4326", vertical_datum="fake-geoid", units="metres",
            pixel_convention="edge", reference="test")

    def test_window_is_exact_copy_across_tiles(self):
        win = fetch_dem_window((77.8, 30.2, 78.3, 30.6), self.tmp / "cache", self.source, pad_px=1)
        with rasterio.open(win.path) as src:
            band, tr = src.read(1), src.transform
            self.assertEqual(src.tags()["VERTICAL_DATUM"], "fake-geoid")
        self.assertEqual(len(win.provenance["tiles_used"]), 2)
        self.assertFalse(np.isnan(band).any())
        rows, cols = np.mgrid[0:band.shape[0], 0:band.shape[1]]
        lon, lat = tr * (cols + 0.5, rows + 0.5)
        # values are copied, not resampled: equal to the plane at the pixel centres
        np.testing.assert_allclose(band, plane(lon, lat), atol=1e-3)

    def test_cache_hit(self):
        a = fetch_dem_window((77.8, 30.2, 78.3, 30.6), self.tmp / "cache", self.source)
        b = fetch_dem_window((77.8, 30.2, 78.3, 30.6), self.tmp / "cache", self.source)
        self.assertEqual(a.path, b.path)
        self.assertEqual(a.provenance["fetched_utc"], b.provenance["fetched_utc"])

    def test_cached_window_needs_no_network(self):
        b = (77.8, 30.2, 78.3, 30.6)
        first = fetch_dem_window(b, self.tmp / "cache3", self.source)
        from dataclasses import replace
        offline = replace(self.source, url_template="/vsicurl/http://127.0.0.1:9/unreachable/{tile}.tif",
                          tile_names=lambda _b: (_ for _ in ()).throw(AssertionError("network/tile lookup touched")))
        again = fetch_dem_window(b, self.tmp / "cache3", offline)
        self.assertEqual(first.path, again.path)

    def test_dem_on_utm_image_grid(self):
        _, meta = read_raster(write_utm_image(self.tmp / "img.tif"))
        win = fetch_dem_window(footprint_wgs84(meta, margin_deg=0.01), self.tmp / "cache", self.source)
        dem, info = dem_to_image_grid(win.path, meta)
        self.assertEqual(dem.shape, (meta.height, meta.width))
        self.assertEqual(info["coverage_fraction"], 1.0)
        rows, cols = np.mgrid[0:meta.height, 0:meta.width]
        x, y = meta.affine * (cols + 0.5, rows + 0.5)
        lon, lat = Transformer.from_crs(UTM44N, "EPSG:4326", always_xy=True).transform(x, y)
        # bilinear reproduces a plane; a half-pixel DEM shift would show up as ~7 m here
        np.testing.assert_allclose(dem, plane(lon, lat), atol=0.05)

    def test_padding_across_degree_line_opens_neighbour_tile(self):
        # regression: requested south edge just above 30.0; 2 px of padding reaches the lat-29 tile
        tiles_dir = self.tmp / "tiles"
        write_fake_tile(tiles_dir / "T_29_77.tif", 29, 77)
        src = DEMSource(
            name="fake2", url_template=str(tiles_dir / "{tile}.tif"),
            tile_names=lambda b: [f"T_{lat}_77" for lat in range(int(np.floor(b[1])), int(np.ceil(b[3])))],
            horizontal_crs="EPSG:4326", vertical_datum="fake-geoid", units="metres",
            pixel_convention="edge", reference="test")
        win = fetch_dem_window((77.3, 30.001, 77.5, 30.2), self.tmp / "cache2", src, pad_px=2)
        with rasterio.open(win.path) as s:
            band = s.read(1)
        self.assertEqual(len(win.provenance["tiles_used"]), 2)
        self.assertFalse(np.isnan(band).any(), "a padded row fell into an unopened neighbour tile")

    def test_copernicus_source_declares_datum(self):
        self.assertIn("EGM2008", COPERNICUS_GLO30.vertical_datum)
        self.assertIn("verified", COPERNICUS_GLO30.vertical_datum)


if __name__ == "__main__":
    unittest.main()
