"""One stage of exp7: the residual, the loss and the stopping rule.

Stage l trains one mode on the target minus the output of the l frozen modes.
joint_loss is the loss of a run that trains all modes at once instead (joint.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def residual_target(
    target: torch.Tensor, frozen_left: torch.Tensor, frozen_right: torch.Tensor
) -> torch.Tensor:
    """target (B, K) minus the frozen modes: left (B, k) and right (B, K, k)."""
    return target - torch.einsum("bi,bji->bj", frozen_left, frozen_right)


def stage_loss(left: torch.Tensor, right: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Rank-one loss of left (B,) and right (B, K) against the residual (B, K).

    -2 mean_b[left_b mean_j(residual_bj right_bj)] + mean_b(left_b^2) mean_bj(right_bj^2)
    """
    operator = -2 * (left * (residual * right).mean(dim=1)).mean()
    metric = left.square().mean() * right.square().mean()
    return operator + metric


def nesting_weights(modes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Weights of joint nesting with every rank 1..L counted equally: (L,) and (L, L).

    Mode l (1-based) is part of the ranks l..L, so its weight is (L - l + 1) / L;
    a pair of modes is part of the ranks from the later one on.
    """
    weights = torch.arange(modes, 0, -1, dtype=torch.float32) / modes
    return weights, torch.minimum(weights[:, None], weights[None, :])


def joint_loss(
    left: torch.Tensor, right: torch.Tensor, target: torch.Tensor,
    weights: torch.Tensor, pair_weights: torch.Tensor,
) -> torch.Tensor:
    """Mean over the ranks k = 1..L of the rank-k loss of the first k modes.

    left (B, L), right (B, K, L), target (B, K); weights (L,) and pair_weights
    (L, L) from nesting_weights. The rank-k loss is stage_loss's with k modes:
    -2 sum_{l<=k} mean_b[left_bl mean_j(target_bj right_bjl)]
    + sum_{l,m<=k} mean_b(left_bl left_bm) mean_bj(right_bjl right_bjm).
    No mode is detached: every mode gets the gradient of every rank it is part of.
    """
    batch, top_k = target.shape
    operator = -2 * (weights * (left * torch.einsum("bj,bjl->bl", target, right)).mean(dim=0)
                     ).sum() / top_k
    left_moments = left.T @ left / batch
    right_moments = torch.einsum("bjl,bjm->lm", right, right) / (batch * top_k)
    return operator + (pair_weights * left_moments * right_moments).sum()


@dataclass
class Selection:
    """Early stopping of a stage on a held-out metric (higher is better)."""

    max_steps: int
    min_steps: int
    patience: int
    best: float | None = None
    best_step: int = 0
    stale: int = 0

    @property
    def keeps_best(self) -> bool:
        return self.patience > 0

    def record(self, metric: float, stage_step: int) -> bool:
        """True when metric is a new best; otherwise counts a stale evaluation."""
        if self.patience == 0:
            return False
        if self.best is None or metric > self.best:
            self.best, self.best_step, self.stale = metric, stage_step, 0
            return True
        self.stale += 1
        return False

    def reason(self, stage_step: int) -> str | None:
        """Why the stage stops at this step, or None to go on."""
        if stage_step >= self.max_steps:
            return "steps"
        if self.patience > 0 and stage_step >= self.min_steps and self.stale >= self.patience:
            return "patience"
        return None
