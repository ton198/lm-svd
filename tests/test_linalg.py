import unittest

import numpy as np
import torch

from utils.linalg import matrix_svd


class MatrixSVDTests(unittest.TestCase):
    def test_rank_and_factors_reconstruct_matrix(self) -> None:
        a = (
            np.outer([1, 2, 3, 4], [1, 0, 2, 0, 1, 3])
            + np.outer([0, 1, 0, -1], [2, 1, 0, 1, 0, 1])
        ).astype(np.float32)
        result = matrix_svd(a, device="cpu")
        self.assertEqual(result.rank, 2)
        self.assertEqual(result.U.shape, (4, 4))
        self.assertEqual(result.S.shape, (4,))
        self.assertEqual(result.Vt.shape, (4, 6))
        for factor in (result.U, result.S, result.Vt):
            self.assertEqual(factor.dtype, np.float64)
        self.assertTrue(np.all(result.S >= 0))
        self.assertTrue(np.all(result.S[:-1] >= result.S[1:]))
        np.testing.assert_allclose((result.U * result.S) @ result.Vt, a, atol=1e-5)
        np.testing.assert_allclose(result.U.T @ result.U, np.eye(4), atol=1e-10)
        self.assertEqual(result.rank, np.linalg.matrix_rank(a.astype(np.float64)))

    def test_zero_matrix_has_rank_zero(self) -> None:
        result = matrix_svd(np.zeros((3, 5), dtype=np.float32), device="cpu")
        self.assertEqual(result.rank, 0)
        self.assertEqual(result.rank_tolerance, 0.0)

    def test_float32_matrix_rank_uses_float64_tolerance(self) -> None:
        matrix = np.diag(np.array([1e12, 1e5, 1e-5, 0], dtype=np.float32))
        result = matrix_svd(matrix, device="cpu")
        tolerance = float(result.S.max()) * 4 * float(np.finfo(np.float64).eps)
        self.assertEqual(result.rank, 2)
        self.assertEqual(result.rank_tolerance, tolerance)
        self.assertGreater(result.S[1], tolerance)
        self.assertLess(result.S[2], tolerance)
        self.assertEqual(result.S[3], 0.0)

    def test_invalid_matrices_are_rejected(self) -> None:
        for matrix, message in (
            (np.zeros(3), "non-empty 2-D"),
            (np.zeros((0, 3)), "non-empty 2-D"),
            (np.array([[1.0, np.nan]]), "non-finite"),
        ):
            with self.subTest(matrix=matrix):
                with self.assertRaisesRegex(ValueError, message):
                    matrix_svd(matrix, device="cpu")

    def test_invalid_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid device"):
            matrix_svd(np.eye(2), device="not-a-device")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_cpu(self) -> None:
        matrix = np.random.default_rng(0).standard_normal((40, 300))
        matrix *= np.logspace(0, -10, 40)[:, None]
        cpu = matrix_svd(matrix, device="cpu")
        cuda = matrix_svd(matrix, device="cuda")
        np.testing.assert_allclose(
            cuda.S, cpu.S, rtol=0, atol=1e-13 * cpu.S.max()
        )
        self.assertEqual(cuda.rank, cpu.rank)
        np.testing.assert_allclose(
            (cuda.U * cuda.S) @ cuda.Vt,
            matrix,
            atol=1e-12 * cpu.S.max(),
        )


if __name__ == "__main__":
    unittest.main()
