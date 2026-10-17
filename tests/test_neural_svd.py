import math
import unittest

import torch
from torch import nn

from utils import NestedLoRASVDLoss, nesting_masks, neural_svd_spectrum


class NeuralSVDTests(unittest.TestCase):
    def test_masks_weight_columns_by_prefixes_containing_them(self) -> None:
        vector_mask, matrix_mask = nesting_masks(3)
        torch.testing.assert_close(vector_mask, torch.tensor([1, 2 / 3, 1 / 3]))
        torch.testing.assert_close(
            matrix_mask,
            torch.tensor(
                [[1, 2 / 3, 1 / 3], [2 / 3, 2 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]
            ),
        )
        torch.testing.assert_close(
            nesting_masks(5, step=2)[0],
            torch.tensor([1, 1, 2 / 3, 2 / 3, 1 / 3]),
        )
        vector_mask, matrix_mask = nesting_masks(3, sequential=True)
        torch.testing.assert_close(vector_mask, torch.ones(3))
        torch.testing.assert_close(matrix_mask, torch.triu(torch.ones(3, 3)))

    def test_recovers_top_singular_triplets(self) -> None:
        for sequential in (False, True):
            with self.subTest(sequential=sequential):
                gen = torch.Generator().manual_seed(1)
                u, _ = torch.linalg.qr(
                    torch.randn(6, 6, generator=gen, dtype=torch.float64)
                )
                v, _ = torch.linalg.qr(
                    torch.randn(5, 5, generator=gen, dtype=torch.float64)
                )
                a = (
                    u[:, :5]
                    @ torch.diag(torch.tensor([3, 2, 1, 0.3, 0.1], dtype=torch.float64))
                    @ v.T
                )
                f = nn.Parameter(
                    0.1 * torch.randn(6, 3, generator=gen, dtype=torch.float64)
                )
                g = nn.Parameter(
                    0.1 * torch.randn(5, 3, generator=gen, dtype=torch.float64)
                )
                loss_fn = NestedLoRASVDLoss(3, sequential=sequential)
                optimizer = torch.optim.Adam([f, g], lr=1e-2)
                for _ in range(2000):
                    optimizer.zero_grad()
                    loss = loss_fn(f, g, a)
                    loss.backward()
                    optimizer.step()

                spectrum = neural_svd_spectrum(f.detach(), g.detach())
                torch.testing.assert_close(
                    spectrum.singular_values * math.sqrt(30),
                    torch.tensor([3, 2, 1], dtype=torch.float64),
                    atol=1e-4,
                    rtol=0,
                )
                torch.testing.assert_close(
                    ((spectrum.left / math.sqrt(6)).T @ u[:, :3]).abs().diagonal(),
                    torch.ones(3, dtype=torch.float64),
                    atol=1e-4,
                    rtol=0,
                )
                torch.testing.assert_close(
                    ((spectrum.right / math.sqrt(5)).T @ v[:, :3]).abs().diagonal(),
                    torch.ones(3, dtype=torch.float64),
                    atol=1e-4,
                    rtol=0,
                )

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank must be positive"):
            nesting_masks(0)
        for step in (0, 4):
            with self.subTest(step=step):
                with self.assertRaisesRegex(ValueError, "step must be in"):
                    nesting_masks(3, step=step)
        with self.assertRaisesRegex(ValueError, "joint nesting"):
            nesting_masks(3, step=2, sequential=True)
        with self.assertRaisesRegex(ValueError, "f and g must have shape"):
            NestedLoRASVDLoss(3)(
                torch.zeros(4, 2), torch.zeros(5, 3), torch.zeros(4, 5)
            )
        with self.assertRaisesRegex(ValueError, "a_block must have shape"):
            NestedLoRASVDLoss(3)(
                torch.zeros(4, 3), torch.zeros(5, 3), torch.zeros(5, 4)
            )
        with self.assertRaisesRegex(ValueError, "nonzero norm"):
            neural_svd_spectrum(torch.zeros(4, 2), torch.ones(5, 2))


if __name__ == "__main__":
    unittest.main()
