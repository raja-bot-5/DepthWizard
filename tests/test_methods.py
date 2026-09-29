import unittest

import numpy as np

from depthwizard.calibration import checkerboard
from depthwizard.calibration.core import cell_mean
from depthwizard.calibration.methods import (CalibrationInputs, DemOnly, DemResidual, GateConfig, GlobalAffine,
                                             anchor_masks)
from depthwizard.calibration.signal import condition


def world(n=300, a=4.0, b=1500.0, noise=0.0, seed=0):
    """Terrain + buildings; R is an exact affine image of the surface (optionally noisy)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    terrain = 1500 + 0.05 * xx + 5 * np.sin(yy / 50.0)
    surface = terrain.copy()
    for r0, c0 in ((40, 40), (60, 200), (180, 120), (220, 230)):
        surface[r0:r0 + 15, c0:c0 + 15] += 12.0
    r = (surface - b) / a + rng.normal(0, noise, surface.shape)
    cell = (np.arange(n)[:, None] // 50) * (n // 50) + (np.arange(n)[None, :] // 50)   # 30 m cells at 0.6 m
    dem = cell_mean(terrain + 0.0 * xx, cell)        # a coarse DEM: one value per 30 m cell
    return surface, terrain, r, dem.astype(np.float32), cell


def inputs(r, dem, cell, water=None, dem_error=None):
    blocks = checkerboard(*dem.shape, 60)
    return CalibrationInputs(signal=condition(r, +1, gsd_m=0.6), dem=dem, res=(0.6, 0.6), fit_mask=blocks,
                             eval_mask=~blocks, cell_ids=cell, water=water, dem_error=dem_error, edge_px=8)


class TestMethods(unittest.TestCase):
    def test_m2_cell_block_average_returns_the_dem_exactly(self):
        _, _, r, dem, cell = world()
        res = DemResidual("cell").calibrate(inputs(r, dem, cell), GateConfig(max_scale_rel_std=5))
        self.assertTrue(res.is_metric, res.report["gate"])
        # the property from the spec: block-averaging Z over each DEM cell gives back the DEM exactly
        np.testing.assert_allclose(cell_mean(res.z, cell), cell_mean(dem, cell), atol=1e-3)

    def test_m2_recovers_building_heights(self):
        surface, terrain, r, dem, cell = world()
        res = DemResidual("cell").calibrate(inputs(r, dem, cell), GateConfig(max_scale_rel_std=5))
        roof = res.z[45:52, 45:52].mean()
        ground = res.z[45:52, 70:77].mean()
        self.assertAlmostEqual(roof - ground, 12.0, delta=2.5)

    def test_m0_is_the_dem_and_reports(self):
        _, _, r, dem, cell = world()
        res = DemOnly().calibrate(inputs(r, dem, cell))
        self.assertTrue(res.is_metric)
        np.testing.assert_array_equal(res.z, dem)
        self.assertEqual(res.report["output_datum"], "EGM2008")
        self.assertTrue(res.report["gate"]["passed"])

    def test_m1_recovers_affine_on_ideal_data(self):
        from depthwizard.calibration import lowpass
        surface, _, r, dem, cell = world()
        # an ideal DEM is the true surface seen at the DEM's own (30 m) resolution, like the low-pass of R
        inp = inputs(r, lowpass(np.asarray(surface, np.float32), 50), cell)
        res = GlobalAffine().calibrate(inp)
        self.assertTrue(res.is_metric, res.report["gate"])
        self.assertAlmostEqual(res.report["a"], 4.0, delta=0.05)
        self.assertLess(res.report["residual_heldout_m"]["nmad"], 0.5)

    def test_m1_gate_fails_when_low_frequency_does_not_follow_terrain(self):
        surface, terrain, r, dem, cell = world()
        rng = np.random.default_rng(1)
        r_bad = np.cumsum(rng.normal(0, 1, r.shape), axis=1) / 5.0     # structured nonsense, unrelated to terrain
        res = GlobalAffine().calibrate(inputs(r_bad, dem, cell))
        self.assertFalse(res.is_metric)
        self.assertIsNone(res.z)
        self.assertIn("relative DSM only", res.report["fallback"])
        self.assertTrue(res.report["gate"]["reasons"])

    def test_negative_scale_fails_gate(self):
        _, _, r, dem, cell = world()
        res = DemResidual("smooth").calibrate(inputs(-r, dem, cell))
        self.assertFalse(res.is_metric)
        self.assertTrue(any("<= 0" in x for x in res.report["gate"]["reasons"]))

    def test_water_excluded_and_flagged(self):
        _, _, r, dem, cell = world()
        water = np.zeros(dem.shape, bool)
        water[100:160, :] = True
        inp = inputs(r, dem, cell, water=water, dem_error=np.full(dem.shape, 1.0, np.float32))
        fit, held, excl = anchor_masks(inp)
        self.assertFalse((fit | held)[water].any())
        self.assertGreater(excl["water"], 0)
        res = DemResidual("smooth").calibrate(inp, GateConfig(max_scale_rel_std=5))
        self.assertTrue(res.low_confidence[water].all())
        self.assertTrue(np.isnan(res.uncertainty[water]).all())
        self.assertTrue(np.isfinite(res.uncertainty[~water]).all())

    def test_split_never_overlaps(self):
        _, _, r, dem, cell = world()
        inp = inputs(r, dem, cell)
        inp.eval_mask = np.ones_like(inp.fit_mask)
        with self.assertRaises(ValueError):
            anchor_masks(inp)

    def test_too_few_anchors_fails(self):
        _, _, r, dem, cell = world()
        res = DemResidual("smooth").calibrate(inputs(r, dem, cell), GateConfig(min_anchors=10 ** 7))
        self.assertFalse(res.is_metric)


if __name__ == "__main__":
    unittest.main()
