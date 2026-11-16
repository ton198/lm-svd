"""The trainable part of exp5.

Exp5Network: one MLP per side on frozen GPT-2 hidden states. The right MLP
reads a future's hidden state and returns the feature vector of every header
slot of that future at once. It is not told which token a slot holds. With a
right encoder the hidden state of a future comes from a trainable copy of the
language model's transformer instead of from the frozen model.

Exp5FinetuneNetwork: a trainable copy of the transformer per side, each with a
linear head on its last hidden state. The left side reads the history. The
right side reads a column as its future followed by the header token, so the
token is part of the input; the header tokens of one future share one
sequence, each seeing the future and itself only.
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


def linear_head(hidden_size: int, rank: int) -> nn.Sequential:
    """LayerNorm without weights, then one linear layer to `rank` outputs."""
    return nn.Sequential(
        nn.LayerNorm(hidden_size, elementwise_affine=False), nn.Linear(hidden_size, rank)
    )


def header_attention(
    future_tokens: int,
    top_k: int,
    batch: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Additive mask (batch, 1, L, L) and position ids (batch, L) of f z_1 ... z_K.

    f is causal; every z_j sees f and itself and sits at position |f|, so its
    final hidden state is that of the plain sequence f z_j.
    """
    length = future_tokens + top_k
    allowed = torch.zeros(length, length, dtype=torch.bool, device=device)
    allowed[:future_tokens, :future_tokens] = torch.ones(
        future_tokens, future_tokens, dtype=torch.bool, device=device).tril()
    allowed[future_tokens:, :future_tokens] = True
    allowed[future_tokens:, future_tokens:] = torch.eye(top_k, dtype=torch.bool, device=device)
    mask = torch.zeros(length, length, dtype=dtype, device=device).masked_fill(
        ~allowed, torch.finfo(dtype).min)
    position = torch.cat([
        torch.arange(future_tokens, device=device),
        torch.full((top_k,), future_tokens, device=device),
    ])
    return mask.expand(batch, 1, -1, -1), position.expand(batch, -1)


class Exp5FinetuneNetwork(nn.Module):
    def __init__(
        self, left_encoder: nn.Module, right_encoder: nn.Module, hidden_size: int, rank: int
    ) -> None:
        super().__init__()
        self.rank = rank
        self.left_encoder = left_encoder
        self.right_encoder = right_encoder
        self.left_head = linear_head(hidden_size, rank)
        self.right_head = linear_head(hidden_size, rank)

    def left_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        """(B, H) token ids -> (B, rank)."""
        hidden = self.left_encoder(input_ids=history_ids, use_cache=False).last_hidden_state
        return self.left_head(hidden[:, -1].float())

    def right_features(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        """Features (F * top_k, rank); row f * top_k + j is f followed by header_ids[f, j]."""
        count, future_tokens = future_ids.shape
        top_k = header_ids.shape[1]
        mask, position = header_attention(
            future_tokens, top_k, count, self.right_head[1].weight.dtype, future_ids.device)
        hidden = self.right_encoder(
            input_ids=torch.cat([future_ids, header_ids.to(future_ids.device)], dim=1),
            attention_mask=mask, position_ids=position, use_cache=False,
        ).last_hidden_state
        return self.right_head(hidden[:, future_tokens:].float()).reshape(-1, self.rank)

    def forward(
        self, history_ids: torch.Tensor, future_ids: torch.Tensor, header_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.left_features(history_ids), self.right_features(future_ids, header_ids)


def build_finetune_model(inference: CausalLMInference, rank: int) -> NestedLoRA:
    """Exp5FinetuneNetwork on two trainable copies of the model, under NestedLoRA."""
    left_encoder, right_encoder = trainable_encoder(inference), trainable_encoder(inference)
    hidden_size = inference.model.config.hidden_size
    return NestedLoRA(
        model=Exp5FinetuneNetwork(left_encoder, right_encoder, hidden_size, rank), neigs=rank
    )
