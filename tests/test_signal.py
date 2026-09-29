"""Polarity and signal-split tests.

Synthetic scene: known surface heights h (a building block on a slope). Fake backends mimic the two
output conventions a nadir camera at height 500 m would produce: depth = 500 - h (DA3-Mono style) and
disparity = 1 / depth (DA-V2 style). After conditioning with each backend's height_sign, R must increase
with h for both. The same height_sign constants are the ones the real predictors declare.
Real-model check on real overhead tiles with LiDAR heights: see test_polarity_real_models (skipped if
data or GPU are missing).
"""
import json
import unittest
from pathlib import Path

import numpy as np

from depthwizard.calibration.signal import condition, lowpass_pixels
from depthwizard.depth import DA2Small, DA3MonoLarge

ROOT = Path(__file__).resolve().parents[1]


def scene(n=240):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    h = 0.05 * xx + 3 * np.sin(yy / 40.0)          # terrain
    h[100:120, 120:140] += 15.0                    # a 15 m building, 12 x 12 m (smaller than the 30 m DEM cell)
    return h


class TestPolarity(unittest.TestCase):
    def test_both_conventions_become_height_like(self):
        h = scene()
        depth = 500.0 - h                            # DA3-Mono style
        disparity = 1.0 / depth                      # DA-V2 style
        for out, sign in ((depth, DA3MonoLarge.height_sign), (disparity, DA2Small.height_sign)):
            s = condition(out, sign, gsd_m=0.6)
            self.assertGreater(np.corrcoef(s.r.ravel(), h.ravel())[0, 1], 0.99)
            self.assertGreater(s.r[110, 130], s.r[110, 100])  # the roof is higher than the ground next to it

    def test_split_reconstructs_and_isolates_fine_structure(self):
        h = scene()
        s = condition(h, +1, gsd_m=0.6)
        np.testing.assert_allclose(s.low + s.high, s.r, atol=1e-5)
        self.assertEqual(s.lowpass_px, 50)                    # 30 m / 0.6 m
        self.assertGreater(s.high[110, 130], 5.0)             # building survives in the high-pass
        self.assertLess(abs(float(s.high[200, 40])), 1.0)     # smooth terrain mostly goes to the low-pass

    def test_structures_wider_than_the_dem_cell_leak_into_the_low_pass(self):
        # KNOWN LIMITATION (documented, not a bug in the split): a 48 x 48 m block is wider than the
        # 30 m window, so its height is mostly in low(R), which M2/M3 replace with the DEM.
        h = np.zeros((300, 300))
        h[70:150, 70:150] += 15.0
        s = condition(h, +1, gsd_m=0.6)
        self.assertLess(float(s.high[110, 110]), 3.0)       # centre of the big block: little left in the high-pass
        small = np.zeros((300, 300))
        small[100:120, 100:120] += 15.0
        self.assertGreater(float(condition(small, +1, 0.6).high[110, 110]), 10.0)

    def test_lowpass_pixels(self):
        self.assertEqual(lowpass_pixels(0.6), 50)
        self.assertEqual(lowpass_pixels(2.4), 12)
        with self.assertRaises(ValueError):
            lowpass_pixels(0)

    def test_bad_sign(self):
        with self.assertRaises(ValueError):
            condition(np.zeros((4, 4)), 0, 0.6)


@unittest.skipUnless((ROOT / "data/raw/gamus/manifest_test.json").exists(), "GAMUS subset not downloaded")
class TestPolarityRealModels(unittest.TestCase):
    """Real models on real overhead tiles: after height_sign, R must correlate POSITIVELY with LiDAR AGL."""

    def test_polarity_real_models(self):
        import torch
        import h5py
        if not torch.cuda.is_available():
            self.skipTest("no CUDA")
        man = json.loads((ROOT / "data/raw/gamus/manifest_test.json").read_text())
        tiles = [t for t in man["tiles"] if t["city"] == "DC"][:3]
        load = lambda rel: h5py.File(ROOT / "data/raw/gamus" / rel, "r")["image"][()]   # noqa: E731
        for cls in (DA2Small, DA3MonoLarge):
            pred = cls(device="cuda", process_res=518) if cls is DA3MonoLarge else cls(device="cuda", fp16=True)
            if cls is DA3MonoLarge:
                pred.max_sky_fraction = 1.0
            rs = []
            for t in tiles:
                img, agl = load(t["image"]), load(t["height"]).astype(np.float64)
                r = condition(pred(img), cls.height_sign, 0.5).r
                ok = np.isfinite(agl) & (agl > -5)
                rs.append(np.corrcoef(r[ok], agl[ok])[0, 1])
            self.assertGreater(float(np.median(rs)), 0.0, f"{cls.__name__}: median r {np.median(rs):.3f} <= 0")
            del pred
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
