"""Networks of exp8.

Study B (the table minus its column means) uses two layouts with one interface,
left_features (N, L) and right_features (F, K, L):

- SharedNetwork: one encoder per side, mode l is output l of each head (exp6's
  layout with exp7's encoders).
- IndependentNetwork: L exp7 Modes, each with its own two encoders.

Study C (the column means themselves) uses MeanStudent, a trainable copy of the
frozen model whose output keeps the model's own unembedding.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from experiments.exp5_gpt2_mlp_table.model import linear_head, trainable_encoder
from experiments.exp7_independent_subnets.model import (
    LastToken,
    Mode,
    build_mode,
    roberta_encoder,
    scratch_encoder,
    token_vectors,
)
from utils.inference import CausalLMInference

ENCODERS = ("gpt2", "roberta", "scratch")
LAYOUTS = ("shared", "independent")
SCRATCH = {"scratch_layers": 6, "scratch_width": 384, "scratch_heads": 6}


def build_encoder(
    inference: CausalLMInference, kind: str, roberta_path: Path | None
) -> nn.Module:
    """An exp7 encoder: token ids (B, T) -> (B, width)."""
    if kind == "gpt2":
        return LastToken(trainable_encoder(inference), inference.model.config.hidden_size)
    if kind == "roberta":
        if roberta_path is None:
            raise ValueError("encoder 'roberta' needs roberta_path")
        return roberta_encoder(roberta_path, inference.model_path, inference.vocab_size)
    if kind == "scratch":
        return scratch_encoder(
            inference.vocab_size, inference.max_length, SCRATCH["scratch_layers"],
            SCRATCH["scratch_width"], SCRATCH["scratch_heads"])
    raise ValueError(f"encoder must be one of {ENCODERS}, got {kind!r}")


class SharedNetwork(nn.Module):
    def __init__(
        self, left_encoder: nn.Module, right_encoder: nn.Module, modes: int,
        vectors: torch.Tensor,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.left_encoder = left_encoder
        self.right_encoder = right_encoder
        self.left_head = linear_head(left_encoder.width, modes)
        self.right_head = linear_head(right_encoder.width, modes * vectors.shape[1])
        self.register_buffer("vectors", vectors, persistent=False)

    def encoders(self) -> list[nn.Module]:
        return [self.left_encoder, self.right_encoder]

    def left_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        """(N, H) -> (N, L)."""
        return self.left_head(self.left_encoder(history_ids))

    def right_features(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        """(F, T) futures and (F, K) header tokens -> (F, K, L)."""
        width = self.vectors.shape[1]
        w = self.right_head(self.right_encoder(future_ids)).unflatten(1, (self.modes, width))
        return torch.einsum("fle,fke->fkl", w, self.vectors[header_ids]) / width ** 0.5


class IndependentNetwork(nn.Module):
    def __init__(self, modes: list[Mode], vectors: torch.Tensor) -> None:
        super().__init__()
        self.mode_list = nn.ModuleList(modes)
        self.register_buffer("vectors", vectors, persistent=False)

    def encoders(self) -> list[nn.Module]:
        return [encoder for mode in self.mode_list
                for encoder in (mode.left_encoder, mode.right_encoder)]

    def left_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([mode.left_features(history_ids) for mode in self.mode_list], dim=-1)

    def right_features(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            mode.right_features(future_ids, header_ids, self.vectors) for mode in self.mode_list
        ], dim=-1)


def build_network(
    inference: CausalLMInference, *, layout: str, encoder: str, modes: int,
    roberta_path: Path | None = None,
) -> nn.Module:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    vectors = token_vectors(inference.model.get_output_embeddings().weight)
    if layout == "shared":
        return SharedNetwork(
            build_encoder(inference, encoder, roberta_path),
            build_encoder(inference, encoder, roberta_path), modes, vectors)
    return IndependentNetwork(
        [build_mode(inference, encoder, roberta_path=roberta_path, **SCRATCH)
         for _ in range(modes)],
        vectors)


def parameter_groups(network: nn.Module, encoder: str, lr: float, encoder_lr: float) -> list[dict]:
    """Pretrained encoders at encoder_lr, everything else at lr; scratch has one group."""
    if encoder == "scratch":
        return [{"params": list(network.parameters()), "lr": lr}]
    in_encoders = {
        id(parameter) for module in network.encoders() for parameter in module.parameters()}
    return [
        {"params": [p for p in network.parameters() if id(p) in in_encoders], "lr": encoder_lr},
        {"params": [p for p in network.parameters() if id(p) not in in_encoders], "lr": lr},
    ]


class MeanStudent(nn.Module):
    """Predicts a future's entries averaged over the histories, reading the future alone.

    It is a trainable copy of the frozen model's transformer and it keeps the
    frozen unembedding: its prediction for header token z is <s'(future), W_z -
    mean row>, where s' is the copy's last hidden state. Before any training it
    therefore equals the frozen model's own entries.

    With prefix_ids, the copy reads those tokens' embeddings in front of the
    future, as trainable vectors. The future then sits at the positions it has
    behind a real history, and the prefix can turn into an "average history".
    At the start it is one real history, i.e. the prediction is that history's
    row of the table.
    """

    def __init__(self, inference: CausalLMInference, prefix_ids: torch.Tensor | None) -> None:
        super().__init__()
        self.transformer = trainable_encoder(inference)
        weight = inference.model.get_output_embeddings().weight.detach().float()
        self.register_buffer("rows", weight - weight.mean(dim=0), persistent=False)
        embedding = self.transformer.get_input_embeddings()
        self.prefix = None
        if prefix_ids is not None and len(prefix_ids):
            self.prefix = nn.Parameter(
                embedding(prefix_ids.to(embedding.weight.device)).detach().clone())

    def forward(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        """(F, T) futures and (F, K) header tokens -> predicted mean entries (F, K)."""
        embedded = self.transformer.get_input_embeddings()(future_ids)
        if self.prefix is not None:
            embedded = torch.cat(
                (self.prefix[None].expand(len(embedded), -1, -1), embedded), dim=1)
        hidden = self.transformer(inputs_embeds=embedded, use_cache=False).last_hidden_state
        return torch.einsum("fe,fke->fk", hidden[:, -1].float(), self.rows[header_ids])
