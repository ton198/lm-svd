"""The sub-networks of one mode of exp7.

Every singular function of the table gets its own sub-network, a Mode. Its left
side reads the history: an encoder and a head give one number per history. Its
right side reads the future: an encoder and a head give a vector w(future) of
the unembedding width E, which meets the header token z only in the inner product

    right(future, z) = <w(future), v_z> / sqrt(E)

with v_z the unembedding row of z minus the mean row, at unit scale. No
parameter is shared between the two sides or between two modes.

An encoder maps token ids (B, T) to one pooled vector per sequence (B, width).
Sequences have a fixed length and no padding, so no attention mask is needed.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import BertConfig, BertModel

from experiments.exp7_independent_subnets import SUBNETS
from experiments.exp5_gpt2_mlp_table.model import linear_head, trainable_encoder
from utils.inference import CausalLMInference


def token_vectors(unembedding: torch.Tensor) -> torch.Tensor:
    """(V, E) unembedding -> (V, E): rows minus the mean row, over one root mean square."""
    centered = unembedding.detach().float()
    centered = centered - centered.mean(dim=0)
    # Divided by one number, so the features stay linear in the rows.
    return centered / centered.square().mean().sqrt()


class LastToken(nn.Module):
    """A causal transformer read at its last position."""

    def __init__(self, transformer: nn.Module, width: int) -> None:
        super().__init__()
        self.transformer = transformer
        self.width = width

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids -> (B, width)."""
        return self.transformer(input_ids=ids, use_cache=False).last_hidden_state[:, -1].float()


class MeanPool(nn.Module):
    """A transformer's last hidden state averaged over all positions."""

    def __init__(self, transformer: nn.Module, width: int) -> None:
        super().__init__()
        self.transformer = transformer
        self.width = width

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids -> (B, width)."""
        return self.transformer(input_ids=ids).last_hidden_state.mean(dim=1).float()


def scratch_encoder(
    vocab_size: int, positions: int, layers: int, width: int, heads: int
) -> MeanPool:
    """A randomly initialised bidirectional BERT transformer, dropout off, mean-pooled.

    It reads GPT-2 token ids directly; BERT's own pooler is left out.
    """
    return MeanPool(BertModel(BertConfig(
        vocab_size=vocab_size, hidden_size=width, num_hidden_layers=layers,
        num_attention_heads=heads, intermediate_size=4 * width,
        max_position_embeddings=positions, type_vocab_size=1,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    ), add_pooling_layer=False), width)


class Mode(nn.Module):
    def __init__(
        self,
        left_encoder: nn.Module,
        right_encoder: nn.Module,
        left_head: nn.Module,
        right_head: nn.Module,
    ) -> None:
        super().__init__()
        self.left_encoder = left_encoder
        self.right_encoder = right_encoder
        self.left_head = left_head
        self.right_head = right_head

    def left_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        """(B, H) token ids -> (B,)."""
        return self.left_head(self.left_encoder(history_ids)).squeeze(-1)

    def right_vectors(self, future_ids: torch.Tensor) -> torch.Tensor:
        """w(future): (F, T) token ids -> (F, E)."""
        return self.right_head(self.right_encoder(future_ids))

    def right_features(
        self, future_ids: torch.Tensor, header_ids: torch.Tensor, vectors: torch.Tensor
    ) -> torch.Tensor:
        """(F, T) futures, (F, K) header tokens, (V, E) token vectors -> (F, K)."""
        w = self.right_vectors(future_ids)
        tokens = vectors[header_ids.to(vectors.device)].to(w.device)
        return torch.einsum("fe,fke->fk", w, tokens) / w.shape[1] ** 0.5


def build_mode(
    inference: CausalLMInference,
    subnet: str,
    *,
    scratch_layers: int,
    scratch_width: int,
    scratch_heads: int,
) -> Mode:
    """One mode with two encoders of kind `subnet` and their two heads.

    `scratch` starts each side at a random BERT transformer, `gpt2` at a copy of
    the teacher's transformer.
    """
    if subnet not in SUBNETS:
        raise ValueError(f"subnet must be one of {SUBNETS}, got {subnet!r}")
    if subnet == "scratch":
        left_encoder, right_encoder = (
            scratch_encoder(
                inference.vocab_size, inference.max_length,
                scratch_layers, scratch_width, scratch_heads)
            for _ in range(2)
        )
    elif subnet == "gpt2":
        width = inference.model.config.hidden_size
        left_encoder, right_encoder = (
            LastToken(trainable_encoder(inference), width) for _ in range(2))
    else:
        raise ValueError(f"subnet {subnet!r} cannot be built yet")
    embedding = inference.model.get_output_embeddings().weight.shape[1]
    return Mode(
        left_encoder, right_encoder,
        linear_head(left_encoder.width, 1), linear_head(right_encoder.width, embedding))
