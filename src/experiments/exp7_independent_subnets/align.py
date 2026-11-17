"""The alignment of the learned functions of exp7.

The L learned pairs (f_l on histories, g_l on future and token columns) are not
orthogonal and have no common scale. Three moment matrices over the sampled
pairs

    lam_f = E[f fᵀ],   lam_g = E[g gᵀ],   cross[i, j] = E[f_i * mean_k(target_k g_kj)]

give a small L x L problem. Its SVD returns two transforms T_f, T_g such that
F T_f and G T_g are orthogonal, ordered by size, and their product is the
least-squares fit of the target inside the span of the learned functions.
"""

from __future__ import annotations

import torch


class Moments:
    """Running means of the three moment matrices, accumulated on the CPU in float64."""

    def __init__(self, modes: int) -> None:
        self.modes = modes
        self.pairs = 0
        self.columns = 0
        self._f = torch.zeros(modes, modes, dtype=torch.float64)
        self._g = torch.zeros(modes, modes, dtype=torch.float64)
        self._cross = torch.zeros(modes, modes, dtype=torch.float64)

    def add(self, left: torch.Tensor, right: torch.Tensor, target: torch.Tensor) -> None:
        """left (P, L), right (P, K, L), target (P, K)."""
        left, right, target = (
            x.detach().to("cpu", torch.float64) for x in (left, right, target))
        P, K, _ = right.shape
        self._f += left.T @ left
        self._g += torch.einsum("pki,pkj->ij", right, right)
        self._cross += left.T @ torch.einsum("pk,pkj->pj", target, right) / K
        self.pairs += P
        self.columns += P * K

    @property
    def lam_f(self) -> torch.Tensor:
        return self._f / self.pairs

    @property
    def lam_g(self) -> torch.Tensor:
        return self._g / self.columns

    @property
    def cross(self) -> torch.Tensor:
        return self._cross / self.pairs


def inverse_sqrt(matrix: torch.Tensor, floor: float) -> torch.Tensor:
    """Symmetric inverse square root; eigenvalues below floor * largest give 0."""
    values, vectors = torch.linalg.eigh(matrix)
    keep = values > floor * values.max().clamp_min(0.0)
    scale = torch.where(keep, values.clamp_min(1e-300).rsqrt(), torch.zeros_like(values))
    return (vectors * scale) @ vectors.T


def alignment(
    lam_f: torch.Tensor, lam_g: torch.Tensor, cross: torch.Tensor, floor: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (T_f, T_g, S) with F T_f, G T_g orthogonal and S the sizes, descending."""
    root_f = inverse_sqrt(lam_f, floor)
    root_g = inverse_sqrt(lam_g, floor)
    P, S, Qt = torch.linalg.svd(root_f @ cross @ root_g)
    half = S.sqrt()
    return root_f @ (P * half), root_g @ (Qt.T * half), S


def max_correlation(gram: torch.Tensor) -> float:
    """Largest absolute off-diagonal entry of the gram matrix as a correlation.

    A mode of zero scale (a zero diagonal entry) has correlation 0 with every other.
    """
    if gram.shape[0] < 2:
        return 0.0
    scale = gram.diagonal().sqrt()
    outer = torch.outer(scale, scale)
    corr = torch.where(outer > 0, gram.abs() / outer.where(outer > 0, 1.0), 0.0)
    return corr.fill_diagonal_(0.0).max().item()
