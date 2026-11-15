"""The trainable part of exp5: one MLP per side on frozen GPT-2 hidden states.

The right MLP reads a future's hidden state and returns the feature vector of
every header slot of that future at once. It is not told which token a slot
holds. With a right encoder the hidden state of a future comes from a trainable
copy of the language model's transformer instead of from the frozen model.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from utils.inference import CausalLMInference
from utils.nestedlora import NestedLoRA


def _mlp(hidden_size: int, mlp_width: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        # GPT-2 hidden states are far from unit scale.
        nn.LayerNorm(hidden_size, elementwise_affine=False),
        nn.Linear(hidden_size, mlp_width),
        nn.GELU(),
        nn.Linear(mlp_width, out_features),
    )


def trainable_encoder(inference: CausalLMInference) -> nn.Module:
    """A trainable copy of the language model's transformer, with dropout off.

    The model itself stays frozen: it still defines the header tokens, g and
    the history hidden states.
    """
    encoder = copy.deepcopy(inference.model.base_model)
    for module in encoder.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0  # training pairs are not reused, so there is nothing to regularise
    return encoder.requires_grad_(True)


class Exp5Network(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int,
        top_k: int,
        mlp_width: int,
        right_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.top_k = top_k
        self.left = _mlp(hidden_size, mlp_width, rank)
        self.right = _mlp(hidden_size, mlp_width, top_k * rank)
        self.register_module("right_encoder", right_encoder)

    def left_features(self, history_hidden: torch.Tensor) -> torch.Tensor:
        """(B, hidden) -> (B, rank)."""
        return self.left(history_hidden)

    def right_features(self, future_input: torch.Tensor) -> torch.Tensor:
        """Features (F * top_k, rank); row f * top_k + j is slot j of future f.

        future_input is the frozen hidden state (F, hidden), or with a right
        encoder the future's token ids (F, T).
        """
        if self.right_encoder is not None:
            future_input = self.right_encoder(
                input_ids=future_input, use_cache=False
            ).last_hidden_state[:, -1].float()
        return self.right(future_input).reshape(-1, self.rank)

    def forward(
        self, history_hidden: torch.Tensor, future_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.left_features(history_hidden), self.right_features(future_input)


def build_model(
    hidden_size: int,
    rank: int,
    top_k: int,
    mlp_width: int,
    right_encoder: nn.Module | None = None,
) -> NestedLoRA:
    """The network under the jointly nested NestedLoRA loss."""
    return NestedLoRA(
        model=Exp5Network(hidden_size, rank, top_k, mlp_width, right_encoder), neigs=rank
    )
