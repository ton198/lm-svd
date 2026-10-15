import unittest

import numpy as np

from utils import matrix_svd


class MatrixSVDTests(unittest.TestCase):
    def test_rank_and_factors_reconstruct_matrix(self) -> None:
        a = (
            np.outer([1, 2, 3, 4], [1, 0, 2, 0, 1, 3])
            + np.outer([0, 1, 0, -1], [2, 1, 0, 1, 0, 1])
        ).astype(np.float32)
        result = matrix_svd(a)
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
        self.assertEqual(result.rank, np.linalg.matrix_rank(a))

    def test_zero_matrix_has_rank_zero(self) -> None:
        result = matrix_svd(np.zeros((3, 5), dtype=np.float32))
        self.assertEqual(result.rank, 0)
        self.assertEqual(result.rank_tolerance, 0.0)

    def test_invalid_matrices_are_rejected(self) -> None:
        for matrix, message in (
            (np.zeros(3), "non-empty 2-D"),
            (np.zeros((0, 3)), "non-empty 2-D"),
            (np.array([[1.0, np.nan]]), "non-finite"),
        ):
            with self.subTest(matrix=matrix):
                with self.assertRaisesRegex(ValueError, message):
                    matrix_svd(matrix)


if __name__ == "__main__":
    unittest.main()
