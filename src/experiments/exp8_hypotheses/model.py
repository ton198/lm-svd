"""Networks of exp8.

Study B (the table minus its column means) uses two layouts with one interface,
left_features (N, L) and right_features (F, K, L):

- SharedNetwork: one encoder per side, mode l is output l of each head (exp6's
  layout with exp7's encoders).
- IndependentNetwork: L exp7 Modes, each with its own two encoders.
- LandmarkNetwork (study D): only the left side is a network. A right function
  is a fixed mix of the frozen model's own entries behind M landmark histories.

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
    MeanPool,
    Mode,
    build_mode,
    roberta_encoder,
    scratch_encoder,
    token_vectors,
)
from utils.inference import CausalLMInference

ENCODERS = ("gpt2", "gpt2-mean", "roberta", "scratch")
# Kinds that start from pretrained weights, trained at encoder_lr.
PRETRAINED = ("gpt2", "gpt2-mean", "roberta")
LAYOUTS = ("shared", "independent", "landmark", "prototype")
SCRATCH = {"scratch_layers": 6, "scratch_width": 384, "scratch_heads": 6}


def build_encoder(
    inference: CausalLMInference, kind: str, roberta_path: Path | None
) -> nn.Module:
    """An exp7 encoder: token ids (B, T) -> (B, width)."""
    if kind == "gpt2":
        return LastToken(trainable_encoder(inference), inference.model.config.hidden_size)
    if kind == "gpt2-mean":
        # The same copy, averaged over its positions instead of read at the last one.
        return MeanPool(trainable_encoder(inference), inference.model.config.hidden_size)
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


class LandmarkNetwork(nn.Module):
    """Right functions made of the frozen model's entries behind landmark histories.

    For M fixed landmark histories h_1..h_M, the response of a column (future, z)
    is the vector of its entries behind each of them, minus their mean:

        r_m(future, z) = entry(h_m, future, z) - mean over m' of entry(h_m', future, z)

    and right function l is sum_m mix[l, m] r_m. The frozen model computes the
    responses for any future, so the right side has nothing to generalise; it
    costs M passes of the model per future. Only `mix` (L x M) and the left
    encoder and head are trained. The mean over the landmarks is itself an
    estimate of the column's mean over all histories (see landmark_mean).
    """

    def __init__(
        self, left_encoder: nn.Module, modes: int, teacher, landmark_ids: torch.Tensor
    ) -> None:
        super().__init__()
        if modes > len(landmark_ids):
            raise ValueError(
                f"{modes} modes need at least as many landmarks, got {len(landmark_ids)}")
        self.modes = modes
        self.left_encoder = left_encoder
        self.left_head = linear_head(left_encoder.width, modes)
        self.mix = nn.Parameter(torch.eye(modes, len(landmark_ids)))
        self.register_buffer("landmark_ids", landmark_ids)
        # Not a submodule: the frozen model is neither trained nor saved here.
        object.__setattr__(self, "teacher", teacher)
        self._table_responses: dict[tuple[int, int], torch.Tensor] = {}

    def encoders(self) -> list[nn.Module]:
        return [self.left_encoder]

    def left_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        return self.left_head(self.left_encoder(history_ids))

    @torch.no_grad()
    def entries(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        """Entries (M, F, K) of every landmark history with every future."""
        return self.teacher.grid_entries(self.landmark_ids, future_ids, header_ids)

    def responses(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        """(M, F, K), centered over the landmarks.

        In evaluation mode the result is kept for the same slice of a table's
        futures: the landmarks are fixed, so a table's responses never change.
        """
        key = (future_ids.data_ptr(), len(future_ids))
        if not self.training and key in self._table_responses:
            return self._table_responses[key]
        entries = self.entries(future_ids, header_ids)
        responses = entries - entries.mean(dim=0, keepdim=True)
        if not self.training:
            self._table_responses[key] = responses
        return responses

    def right_features(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        return torch.einsum("lm,mfk->fkl", self.mix, self.responses(future_ids, header_ids))


class PrototypeNetwork(LandmarkNetwork):
    """LandmarkNetwork whose landmarks are trained.

    Each landmark history is replaced by its token embeddings as free vectors (a
    "prototype history"), read in front of the future by a copy of the model.
    At the start the two networks are the same function. Training moves the
    prototypes towards histories whose rows span the table best; with
    train_copy the copy's weights are trained as well. The right side is then
    a network of its own: it never calls the frozen model.

    The copy runs in bfloat16 on a GPU, which changes the centered entries'
    energy by about 0.1%.
    """

    def __init__(
        self, left_encoder: nn.Module, modes: int, inference: CausalLMInference,
        landmark_ids: torch.Tensor, train_copy: bool, chunk: int = 512,
    ) -> None:
        super().__init__(left_encoder, modes, None, landmark_ids)
        self.transformer = trainable_encoder(inference).requires_grad_(train_copy)
        embedding = self.transformer.get_input_embeddings()
        self.prototypes = nn.Parameter(
            embedding(landmark_ids.to(embedding.weight.device)).detach().clone())
        weight = inference.model.get_output_embeddings().weight.detach().float()
        self.register_buffer("rows", weight - weight.mean(dim=0), persistent=False)
        self.train_copy = train_copy
        self.chunk = chunk

    def encoders(self) -> list[nn.Module]:
        return [self.left_encoder, self.transformer]

    def _entries(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        prototypes, futures = len(self.prototypes), len(future_ids)
        on_gpu = self.prototypes.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_gpu):
            cache = self.transformer(inputs_embeds=self.prototypes, use_cache=True).past_key_values
            cache.batch_repeat_interleave(futures)
            hidden = self.transformer(
                input_ids=future_ids.repeat(prototypes, 1), past_key_values=cache, use_cache=True,
            ).last_hidden_state[:, -1]
        return torch.einsum(
            "mfe,fke->mfk", hidden.float().unflatten(0, (prototypes, futures)),
            self.rows[header_ids])

    def entries(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        """(M, F, K): the copy's entries behind every prototype, with gradients in training."""
        if self.training:
            return self._entries(future_ids, header_ids)
        group = max(self.chunk // len(self.prototypes), 1)
        with torch.no_grad():
            return torch.cat([
                self._entries(future_ids[start:start + group], header_ids[start:start + group])
                for start in range(0, len(future_ids), group)
            ], dim=1)

    def responses(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        # The prototypes change, so nothing is kept between evaluations.
        entries = self.entries(future_ids, header_ids)
        return entries - entries.mean(dim=0, keepdim=True)


def build_network(
    inference: CausalLMInference, *, layout: str, encoder: str, modes: int,
    roberta_path: Path | None = None, teacher=None, landmark_ids: torch.Tensor | None = None,
    train_copy: bool = False, right_encoder: str | None = None,
) -> nn.Module:
    """right_encoder gives the shared layout another encoder kind on the right side."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if right_encoder is not None:
        if layout != "shared":
            raise ValueError("right_encoder is for the shared layout")
        if (right_encoder == "scratch") != (encoder == "scratch"):
            raise ValueError("a scratch encoder cannot be mixed with a pretrained one")
    if layout == "prototype":
        if landmark_ids is None:
            raise ValueError("layout 'prototype' needs landmark_ids")
        return PrototypeNetwork(
            build_encoder(inference, encoder, roberta_path), modes, inference, landmark_ids,
            train_copy)
    if layout == "landmark":
        if teacher is None or landmark_ids is None:
            raise ValueError("layout 'landmark' needs the teacher and landmark_ids")
        return LandmarkNetwork(
            build_encoder(inference, encoder, roberta_path), modes, teacher, landmark_ids)
    if encoder == "gpt2-mean" and layout == "independent":
        raise ValueError("encoder 'gpt2-mean' has no independent layout")
    vectors = token_vectors(inference.model.get_output_embeddings().weight)
    if layout == "shared":
        return SharedNetwork(
            build_encoder(inference, encoder, roberta_path),
            build_encoder(inference, right_encoder or encoder, roberta_path), modes, vectors)
    return IndependentNetwork(
        [build_mode(inference, encoder, roberta_path=roberta_path, **SCRATCH)
         for _ in range(modes)],
        vectors)


def parameter_groups(network: nn.Module, encoder: str, lr: float, encoder_lr: float) -> list[dict]:
    """Pretrained encoders at encoder_lr, everything else at lr; scratch has one group."""
    if encoder not in ENCODERS:
        raise ValueError(f"encoder must be one of {ENCODERS}, got {encoder!r}")
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


class GenericMeanStudent(nn.Module):
    """exp7's right sub-network on study C's regression, as the control for MeanStudent.

    encoder -> LayerNorm -> Linear gives w(future), and the predicted entry minus g
    is <w(future), v_z> / sqrt(E), exactly as in an exp7 Mode. Nothing of the
    frozen model's output layer is reused and the prediction does not start at
    the frozen model's own. The forward pass adds g, so both students predict the
    same quantity.
    """

    def __init__(self, encoder: nn.Module, vectors: torch.Tensor, teacher) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = linear_head(encoder.width, vectors.shape[1])
        self.register_buffer("vectors", vectors, persistent=False)
        object.__setattr__(self, "teacher", teacher)

    def forward(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        w = self.head(self.encoder(future_ids))
        change = torch.einsum("fe,fke->fk", w, self.vectors[header_ids]) / w.shape[1] ** 0.5
        return change + self.teacher.future_only(future_ids, header_ids)
