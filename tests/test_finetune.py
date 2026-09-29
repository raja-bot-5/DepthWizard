import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from finetune import ssi_l1  # noqa: E402


class TestSSILoss(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.y = torch.rand(2, 64, 64, generator=g) * 20
        self.m = torch.ones(2, 64, 64, dtype=torch.bool)

    def test_invariant_to_positive_scale_and_shift(self):
        self.assertLess(ssi_l1(3.7 * self.y + 12.0, self.y, self.m).item(), 1e-5)

    def test_sign_flip_is_penalised(self):
        self.assertGreater(ssi_l1(-self.y, self.y, self.m).item(), 0.5)

    def test_mask_excludes_pixels(self):
        pred = self.y.clone()
        pred[:, :10] = 1e6          # garbage only where masked out
        m = self.m.clone()
        m[:, :10] = False
        self.assertLess(ssi_l1(pred, self.y, m).item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
