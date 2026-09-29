import unittest

import numpy as np

from depthwizard.calibration.ndsm import LOW, MEDIUM, building_heights, describe_reasons, estimate_ground, height_confidence


def town(n=400, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    ground = 1600 + 0.02 * xx + 0.01 * yy                      # gentle slope
    dsm = ground + rng.normal(0, 0.03, (n, n))                 # smooth open ground
    fp = np.zeros((n, n), np.int32)
    for k, (r0, c0, hgt) in enumerate(((50, 50, 12.0), (200, 250, 20.0), (300, 80, 6.0)), start=1):
        dsm[r0:r0 + 30, c0:c0 + 30] += hgt
        fp[r0:r0 + 30, c0:c0 + 30] = k
    return dsm.astype(np.float32), ground, fp


class TestGround(unittest.TestCase):
    def test_ground_and_ndsm_recover_building_heights(self):
        dsm, ground, fp = town()
        g = estimate_ground(dsm, (0.6, 0.6))
        open_ground = fp == 0
        self.assertLess(np.abs(g.ground - ground)[open_ground].mean(), 0.3)
        self.assertAlmostEqual(float(np.median(g.ndsm[fp == 2])), 20.0, delta=1.0)
        self.assertTrue(np.all(g.ndsm[np.isfinite(g.ndsm)] >= 0))
        self.assertLess(g.low_confidence.mean(), 0.05)
        self.assertIn("derived", g.notes[0])

    def test_continuous_canopy_is_flagged_not_trusted(self):
        rng = np.random.default_rng(1)
        canopy = (1800 + 15 + 3 * rng.standard_normal((300, 300))).astype(np.float32)   # rough everywhere
        g = estimate_ground(canopy, (0.6, 0.6))
        self.assertGreater(g.low_confidence.mean(), 0.9)

    def test_water_is_low_confidence(self):
        dsm, _, _ = town()
        water = np.zeros(dsm.shape, bool)
        water[:, :40] = True
        g = estimate_ground(dsm, (0.6, 0.6), water=water)
        self.assertTrue(g.low_confidence[water].all())

    def test_building_heights_from_footprints(self):
        dsm, _, fp = town()
        g = estimate_ground(dsm, (0.6, 0.6))
        b = building_heights(dsm, g.ground, fp)
        self.assertEqual(set(b), {1, 2, 3})
        for k, want in ((1, 12.0), (2, 20.0), (3, 6.0)):
            self.assertAlmostEqual(b[k]["height_m"], want, delta=0.3)
            self.assertEqual(b[k]["label"], "derived")


if __name__ == "__main__":
    unittest.main()


class TestHeightConfidence(unittest.TestCase):
    def test_reasons_and_levels(self):
        z = np.full((120, 120), 100.0, np.float32)
        z[40:60, 40:60] = 115.0                                  # a 15 m building on open flat ground
        g = estimate_ground(z, (1.0, 1.0))
        lc = np.full(z.shape, 30, np.uint8)                     # grassland
        lc[:, 100:] = 10                                         # tree cover strip
        water = np.zeros(z.shape, bool)
        water[:5] = True
        sigma = np.full(z.shape, 1.0, np.float32)
        level, why = height_confidence(g, sigma, lc, water)
        self.assertEqual(level[50, 50], MEDIUM)                  # building roof: ground visible, 15 m >> 2 sigma
        self.assertEqual(level[50, 110], LOW)
        self.assertTrue(why[50, 110] & 2)                        # tree cover
        self.assertTrue(why[2, 50] & 4)                          # water
        self.assertTrue(why[80, 20] & 8)                         # 0 m on open ground: below 2 sigma
        self.assertIn("water", " ".join(describe_reasons(int(why[2, 50]))))
        self.assertFalse((level == 3).any())                     # no HIGH level exists

    def test_without_landcover_trees_are_not_flagged(self):
        z = np.full((60, 60), 10.0, np.float32)
        level, why = height_confidence(estimate_ground(z, (1.0, 1.0)))
        self.assertFalse((why & 2).any())
