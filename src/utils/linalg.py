from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MatrixSVD:
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    rank: int
    rank_tolerance: float


def numerical_rank(
    singular_values: np.ndarray, shape: tuple[int, int]
) -> tuple[int, float]:
    tolerance = (
        float(singular_values.max()) * max(shape) * float(np.finfo(np.float64).eps)
        if singular_values.size
        else 0.0
    )
    return int((singular_values > tolerance).sum()), tolerance


def matrix_svd(matrix: np.ndarray, *, device: str | torch.device) -> MatrixSVD:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(
            f"matrix must be a non-empty 2-D array, got shape {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    try:
        device = torch.device(device)
    except RuntimeError as error:
        raise ValueError(f"invalid device: {device}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use device='cpu'")
    data = torch.as_tensor(np.ascontiguousarray(matrix, dtype=np.float64), device=device)
    U, S, Vh = torch.linalg.svd(
        data, full_matrices=False, driver="gesvd" if device.type == "cuda" else None
    )
    U, S, Vt = (factor.cpu().numpy() for factor in (U, S, Vh))
    rank, tolerance = numerical_rank(S, matrix.shape)
    return MatrixSVD(U, S, Vt, rank, tolerance)
