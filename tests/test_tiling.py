import unittest

import numpy as np

from depthwizard.tiling import blend_weights, fit_affine, plan_tiles, run_tiled


class TestPlanTiles(unittest.TestCase):
    def test_covers_every_pixel_with_overlap(self):
        H, W, tile, ov = 1000, 1300, 518, 64
        cover = np.zeros((H, W), int)
        tiles = plan_tiles(H, W, tile, ov)
        for t in tiles:
            cover[t.slices] += 1
            self.assertLessEqual(t.row0 + t.height, H)
            self.assertLessEqual(t.col0 + t.width, W)
        self.assertTrue((cover >= 1).all())
        # neighbours overlap by at least `ov`
        rows = sorted({t.row0 for t in tiles})
        self.assertTrue(all(b - a <= tile - ov for a, b in zip(rows, rows[1:])))

    def test_small_image_single_tile(self):
        self.assertEqual(len(plan_tiles(300, 200, 518, 64)), 1)


class TestFitAffine(unittest.TestCase):
    def test_recovers_scale_shift(self):
        x = np.random.default_rng(0).normal(size=(50, 50))
        a, b = fit_affine(x, 3.0 * x - 7.0)
        self.assertAlmostEqual(a, 3.0, places=6)
        self.assertAlmostEqual(b, -7.0, places=6)


class TestBlendWeights(unittest.TestCase):
    def test_contract(self):
        w = blend_weights(518, 400, 64)
        self.assertEqual(w.shape, (518, 400))
        self.assertTrue(np.all(w > 0), "weights must be > 0 or image-border pixels divide by 0")
        self.assertTrue(np.all(np.isfinite(w)))


class TestRunTiled(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[0:700, 0:900]
        self.truth = (np.sin(xx / 60.0) + np.cos(yy / 45.0) + xx / 300.0).astype(np.float32)
        self.image = np.repeat(self.truth[..., None], 3, axis=2)

    def test_identity_predictor_is_exact(self):
        mosaic, info = run_tiled(self.image, lambda crop: crop[..., 0], tile=256, overlap=48)
        np.testing.assert_allclose(mosaic, self.truth, atol=1e-5)
        self.assertGreater(info["n_tiles"], 1)

    def test_per_tile_affine_ambiguity_is_removed(self):
        # every tile answers in its own random scale/shift, like a monocular model
        rng = np.random.default_rng(3)
        predict = lambda crop: rng.uniform(0.5, 2.0) * crop[..., 0] + rng.uniform(-5, 5)  # noqa: E731
        mosaic, _ = run_tiled(self.image, predict, tile=256, overlap=48)
        a, b = fit_affine(mosaic, self.truth)
        # after one global affine the mosaic matches truth: no seams left
        np.testing.assert_allclose(a * mosaic + b, self.truth, atol=1e-3)

    def test_rejects_wrong_output_shape(self):
        with self.assertRaises(ValueError):
            run_tiled(self.image, lambda crop: crop[::2, ::2, 0], tile=256, overlap=48)


if __name__ == "__main__":
    unittest.main()


class TestJointAlign(unittest.TestCase):
    """Each tile sees the true field through its own scale, offset and tilt (what a monocular model does per
    inference). Joint alignment must undo that up to one global affine, without shrinking the detail."""

    def setUp(self):
        rng = np.random.default_rng(0)
        H, W = 700, 900
        yy, xx = np.mgrid[0:H, 0:W]
        from scipy.ndimage import gaussian_filter
        self.field = gaussian_filter(rng.normal(size=(H, W)), 3) * 10 + 0.002 * xx   # detail + gentle slope
        self.img = np.zeros((H, W, 3), np.uint8)
        self.distort = {}

        def predict(crop):
            r0, c0 = self._where(crop)
            k = len(self.distort)
            a, b, tx, ty = 1 + 0.3 * np.sin(k + 1), 5.0 * k, 0.004 * (k % 3 - 1), -0.003 * (k % 2)
            self.distort[(r0, c0)] = a
            h, w = crop.shape[:2]
            yy_, xx_ = np.mgrid[0:h, 0:w]
            return a * self.field[r0:r0 + h, c0:c0 + w] + b + tx * xx_ + ty * yy_
        self.predict = predict

    def _where(self, crop):                 # crops are views of self.img: recover their offset
        off = crop.__array_interface__["data"][0] - self.img.__array_interface__["data"][0]
        row_bytes = self.img.strides[0]
        return off // row_bytes, (off % row_bytes) // self.img.strides[1]

    def _error(self, mosaic):
        # remove what no tiling can observe: one global scale, offset and plane (the DEM supplies those later)
        H, W = mosaic.shape
        yy, xx = np.mgrid[0:H, 0:W]
        A = np.stack([mosaic.ravel(), np.ones(mosaic.size), xx.ravel(), yy.ravel()], 1).astype(np.float64)
        c = np.linalg.lstsq(A, self.field.ravel(), rcond=None)[0]
        return float(np.std(A @ c - self.field.ravel())), c[0]

    def test_joint_beats_sequential_and_keeps_scale(self):
        m_seq, _ = run_tiled(self.img, self.predict, tile=400, overlap=100, align="sequential")
        self.distort.clear()
        m_joint, info = run_tiled(self.img, self.predict, tile=400, overlap=100, align="joint_plane")
        e_seq, _ = self._error(m_seq)
        e_joint, a_joint = self._error(m_joint)
        self.assertEqual(info["align"], "joint_plane")
        self.assertGreater(e_seq, 0.1 * float(np.std(self.field)))      # sequential leaves tilt steps
        self.assertLess(e_joint, 1e-3 * float(np.std(self.field)))
        # tile scales: a_i * distortion_i must be the same for every tile (no shrink toward 0)
        prod = [r["a"] * self.distort[(r["row0"], r["col0"])] for r in info["tiles"]]
        self.assertLess(np.ptp(prod) / np.mean(prod), 0.02)

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            run_tiled(self.img, self.predict, tile=400, overlap=100, align="diagonal")
