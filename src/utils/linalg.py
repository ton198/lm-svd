from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatrixSVD:
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    rank: int
    rank_tolerance: float


def numerical_rank(
    singular_values: np.ndarray, shape: tuple[int, int], dtype: np.dtype
) -> tuple[int, float]:
    eps_dtype = dtype if np.issubdtype(dtype, np.floating) else np.float64
    tolerance = (
        float(singular_values.max()) * max(shape) * float(np.finfo(eps_dtype).eps)
        if singular_values.size
        else 0.0
    )
    return int((singular_values > tolerance).sum()), tolerance


def matrix_svd(matrix: np.ndarray) -> MatrixSVD:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(
            f"matrix must be a non-empty 2-D array, got shape {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    U, S, Vt = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
    rank, tolerance = numerical_rank(S, matrix.shape, matrix.dtype)
    return MatrixSVD(U, S, Vt, rank, tolerance)
