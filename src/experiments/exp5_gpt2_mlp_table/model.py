"""The trainable part of exp5: one MLP per side on frozen GPT-2 hidden states.

The right MLP reads a future's hidden state and returns the feature vector of
every header slot of that future at once. It is not told which token a slot
holds.
"""

from __future__ import annotations

import torch
from torch import nn

from utils.nestedlora import NestedLoRA


def _mlp(hidden_size: int, mlp_width: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        # GPT-2 hidden states are far from unit scale.
        nn.LayerNorm(hidden_size, elementwise_affine=False),
        nn.Linear(hidden_size, mlp_width),
        nn.GELU(),
        nn.Linear(mlp_width, out_features),
    )


class Exp5Network(nn.Module):
    def __init__(self, hidden_size: int, rank: int, top_k: int, mlp_width: int) -> None:
        super().__init__()
        self.rank = rank
        self.top_k = top_k
        self.left = _mlp(hidden_size, mlp_width, rank)
        self.right = _mlp(hidden_size, mlp_width, top_k * rank)

    def left_features(self, history_hidden: torch.Tensor) -> torch.Tensor:
        """(B, hidden) -> (B, rank)."""
        return self.left(history_hidden)

    def right_features(self, future_hidden: torch.Tensor) -> torch.Tensor:
        """(F, hidden) -> (F * top_k, rank); row f * top_k + j is slot j of future f."""
        return self.right(future_hidden).reshape(-1, self.rank)

    def forward(
        self, history_hidden: torch.Tensor, future_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.left_features(history_hidden), self.right_features(future_hidden)


def build_model(hidden_size: int, rank: int, top_k: int, mlp_width: int) -> NestedLoRA:
    """The network under the jointly nested NestedLoRA loss."""
    return NestedLoRA(model=Exp5Network(hidden_size, rank, top_k, mlp_width), neigs=rank)
