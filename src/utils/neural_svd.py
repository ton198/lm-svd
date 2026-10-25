"""NestedLoRA singular value decomposition (NeuralSVD).

Ported from neural-svd ``methods/nestedlora.py`` (MIT License, Copyright (c) 2024
Jongha (Jon) Ryu): Ryu et al., "Operator SVD with Neural Networks via Nested
Low-Rank Approximation", ICML 2024.

Train ``f: rows -> R^rank`` and ``g: columns -> R^rank`` on sampled blocks of a
matrix or kernel ``A(row, column)``; column ``l`` of ``f g^T`` converges to the
``l``-th singular triplet of the operator ``(T g)(row) = E_column[A(row, column) g(column)]``.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn


def nesting_masks(
    rank: int, *, step: int = 1, sequential: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    if sequential:
        if step != 1:
            raise ValueError("step applies only to joint nesting")
        return torch.ones(rank), torch.triu(torch.ones(rank, rank))
    if not 1 <= step <= rank:
        raise ValueError(f"step must be in [1, {rank}], got {step}")
    end_indices = list(range(step, rank + 1, step))
    if end_indices[-1] != rank:
        end_indices.append(rank)
    weights = torch.zeros(rank, dtype=torch.float64)
    weights[torch.tensor(end_indices) - 1] = 1.0
    weights /= weights.sum()
    vector_mask = weights.flip(0).cumsum(0).flip(0).float()
    matrix_mask = torch.minimum(vector_mask[:, None], vector_mask[None, :])
    return vector_mask, matrix_mask


def cosine_lr(
    step: int,
    *,
    base_lr: float,
    min_lr: float,
    decay_start_step: int,
    decay_steps: int,
) -> float:
    progress = min(max(step - decay_start_step, 0), decay_steps) / decay_steps
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class _NestedLoRASVDFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, f, tg, g, tadj_f, vector_mask, matrix_mask):
        lam_f = f.T @ f / f.shape[0]
        lam_g = g.T @ g / g.shape[0]
        ctx.save_for_backward(f, tg, g, tadj_f, lam_f, lam_g, vector_mask, matrix_mask)
        operator_term = -2 * (vector_mask * f * tg).sum(dim=1).mean()
        metric_term = (matrix_mask * lam_f * lam_g).sum()
        return operator_term + metric_term

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        f, tg, g, tadj_f, lam_f, lam_g, vector_mask, matrix_mask = ctx.saved_tensors
        grad_f = (2 / f.shape[0]) * (f @ (matrix_mask * lam_g) - vector_mask * tg)
        grad_g = (2 / g.shape[0]) * (g @ (matrix_mask * lam_f) - vector_mask * tadj_f)
        return grad_output * grad_f, None, grad_output * grad_g, None, None, None


class NestedLoRASVDLoss(nn.Module):
    """Nested low-rank approximation loss on one sampled block.

    ``f``: (B_rows, rank) features of independently sampled rows; ``g``: (B_cols, rank)
    features of independently sampled columns; ``a_block``: (B_rows, B_cols) entries
    ``A(row_i, column_j)``. ``a_block`` receives no gradient.
    """

    def __init__(self, rank: int, *, step: int = 1, sequential: bool = False) -> None:
        super().__init__()
        vector_mask, matrix_mask = nesting_masks(rank, step=step, sequential=sequential)
        self.rank = rank
        self.register_buffer("vector_mask", vector_mask)
        self.register_buffer("matrix_mask", matrix_mask)

    def forward(
        self, f: torch.Tensor, g: torch.Tensor, a_block: torch.Tensor
    ) -> torch.Tensor:
        if f.ndim != 2 or g.ndim != 2 or f.shape[1] != self.rank or g.shape[1] != self.rank:
            raise ValueError(
                f"f and g must have shape (batch, {self.rank}), "
                f"got {tuple(f.shape)} and {tuple(g.shape)}"
            )
        if a_block.shape != (f.shape[0], g.shape[0]):
            raise ValueError(
                f"a_block must have shape {(f.shape[0], g.shape[0])}, "
                f"got {tuple(a_block.shape)}"
            )
        with torch.no_grad():
            a = a_block.to(device=f.device, dtype=f.dtype)
            tg = a @ g / g.shape[0]
            tadj_f = a.T @ f / f.shape[0]
        return _NestedLoRASVDFunction.apply(
            f, tg, g, tadj_f, self.vector_mask.to(f.dtype), self.matrix_mask.to(f.dtype)
        )


@dataclass(frozen=True)
class NeuralSVDSpectrum:
    """Operator singular values and functions under the sampling measures.

    ``left``/``right`` columns have unit mean square. For a finite m x n matrix
    evaluated on all rows and columns with uniform sampling: matrix singular values
    are ``singular_values * sqrt(m * n)``, unit-norm vectors are ``left / sqrt(m)``
    and ``right / sqrt(n)``.
    """

    singular_values: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor


@torch.no_grad()
def neural_svd_spectrum(f: torch.Tensor, g: torch.Tensor) -> NeuralSVDSpectrum:
    if (
        f.ndim != 2
        or g.ndim != 2
        or f.shape[1] != g.shape[1]
        or 0 in f.shape
        or 0 in g.shape
    ):
        raise ValueError(
            "f and g must be non-empty (n, rank) matrices with equal rank, "
            f"got {tuple(f.shape)} and {tuple(g.shape)}"
        )
    f_norm = f.pow(2).mean(dim=0).sqrt()
    g_norm = g.pow(2).mean(dim=0).sqrt()
    if (f_norm == 0).any() or (g_norm == 0).any():
        raise ValueError("every feature column must have nonzero norm")
    return NeuralSVDSpectrum(f_norm * g_norm, f / f_norm, g / g_norm)
