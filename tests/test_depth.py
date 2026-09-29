import unittest

import numpy as np

from depthwizard.depth import DA2Small, DA3MonoLarge, _check_crop, _to_crop_size


class TestDepthHelpers(unittest.TestCase):
    def test_resize_back_to_crop(self):
        out = _to_crop_size(np.random.rand(518, 518), 300, 700)
        self.assertEqual(out.shape, (300, 700))
        self.assertEqual(out.dtype, np.float32)

    def test_same_size_is_untouched(self):
        a = np.random.rand(40, 50).astype(np.float32)
        np.testing.assert_array_equal(_to_crop_size(a, 40, 50), a)

    def test_rejects_non_rgb_uint8(self):
        with self.assertRaises(ValueError):
            _check_crop(np.zeros((10, 10, 3), np.float32))
        with self.assertRaises(ValueError):
            _check_crop(np.zeros((10, 10), np.uint8))

    def test_polarity_signs(self):
        # class-level defaults; no weights are loaded
        self.assertEqual(DA2Small.height_sign, +1)
        self.assertEqual(DA3MonoLarge.height_sign, -1)


if __name__ == "__main__":
    unittest.main()
