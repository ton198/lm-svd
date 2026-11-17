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
All token ids are the frozen model's (GPT-2's); a roberta encoder maps them to
RoBERTa's ids by token string.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertModel,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    RobertaModel,
)

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


def _tokenizer(path: Path) -> PreTrainedTokenizerBase:
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except OSError as error:
        raise ValueError(f"cannot load the local tokenizer of {path}: {error}") from error


def token_id_map(model_path: Path, roberta_path: Path, vocab_size: int) -> torch.Tensor:
    """(vocab_size,) int64: entry i is RoBERTa's id of the string of token i of model_path.

    Rejects a missing RoBERTa directory and model tokens missing from RoBERTa's
    vocabulary.
    """
    if not Path(roberta_path).is_dir():
        raise ValueError(
            f"RoBERTa directory does not exist: {roberta_path}; download it with "
            "`uv run python scripts/download_models.py --models roberta` or pass --roberta-path")
    strings = {index: token for token, index in _tokenizer(model_path).get_vocab().items()}
    absent = [i for i in range(vocab_size) if i not in strings]
    if absent:
        raise ValueError(
            f"the tokenizer of {model_path} has no token of id {absent[0]} "
            f"(the model has {vocab_size} ids)")
    vocab = _tokenizer(roberta_path).get_vocab()
    missing = [strings[i] for i in range(vocab_size) if strings[i] not in vocab]
    if missing:
        raise ValueError(
            f"{len(missing)} of the {vocab_size} tokens of {model_path} are not in the "
            f"vocabulary of {roberta_path}, for example {missing[0]!r}")
    return torch.tensor([vocab[strings[i]] for i in range(vocab_size)], dtype=torch.int64)


class _PretrainedRoberta(RobertaModel):
    """RobertaModel loaded from a RobertaForMaskedLM checkpoint, as RoBERTa-base is stored.

    The checkpoint's masked-LM head (lm_head.*) is not part of the encoder, so it
    is not reported; any other unexpected weight still is.
    """

    _keys_to_ignore_on_load_unexpected = [r"^lm_head\."]


class RobertaInput(nn.Module):
    """A RoBERTa transformer fed the model's token ids, mapped and wrapped in <s> .. </s>."""

    def __init__(
        self, transformer: nn.Module, id_map: torch.Tensor, bos_id: int, eos_id: int
    ) -> None:
        super().__init__()
        self.transformer = transformer
        # Derived from the tokenizer files, so it is saved in no mode file.
        self.register_buffer("id_map", id_map, persistent=False)
        self.bos_id = bos_id
        self.eos_id = eos_id

    @property
    def config(self) -> PretrainedConfig:
        return self.transformer.config

    @property
    def max_tokens(self) -> int:
        """The longest sequence of model tokens it reads, <s> and </s> not counted.

        RoBERTa's positions start after its pad id: T tokens without padding take
        positions pad_token_id + 1 .. pad_token_id + T.
        """
        config = self.transformer.config
        return config.max_position_embeddings - config.pad_token_id - 1 - 2

    def forward(self, input_ids: torch.Tensor):
        """(B, T) model token ids -> the transformer's output on (B, T + 2) RoBERTa ids."""
        ids = self.id_map[input_ids]
        bos = ids.new_full((len(ids), 1), self.bos_id)
        eos = ids.new_full((len(ids), 1), self.eos_id)
        return self.transformer(input_ids=torch.cat([bos, ids, eos], dim=1))


def roberta_encoder(roberta_path: Path, model_path: Path, vocab_size: int) -> MeanPool:
    """A trainable copy of the pretrained RoBERTa at roberta_path, dropout off, mean-pooled.

    It reads the token ids of the model at model_path; the mean is over all
    positions, <s> and </s> included. RoBERTa's own pooler is left out. Rejects
    a checkpoint that lacks an encoder weight and a model token mapped to the
    pad id of RoBERTa's config.
    """
    id_map = token_id_map(model_path, roberta_path, vocab_size)
    try:
        transformer, info = _PretrainedRoberta.from_pretrained(
            roberta_path, add_pooling_layer=False, hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0, local_files_only=True, output_loading_info=True)
    except OSError as error:
        raise ValueError(f"cannot load the local RoBERTa from {roberta_path}: {error}") from error
    if info["missing_keys"]:
        # from_pretrained would start them at random weights.
        raise ValueError(
            f"the RoBERTa checkpoint in {roberta_path} lacks encoder weights: "
            f"{', '.join(sorted(info['missing_keys']))}")
    # A checkpoint stored in half precision (MiniLMv2) loads as float16, and Adam on
    # float16 parameters gives NaN: every kind is trained in float32.
    transformer.float()
    config = transformer.config
    # Its embeddings read the config's pad id as padding (padding_idx): the positions
    # of the tokens after it would shift.
    padded = (id_map == config.pad_token_id).nonzero().flatten().tolist()
    if padded:
        raise ValueError(
            f"token id {padded[0]} of {model_path} maps to {config.pad_token_id}, the pad "
            f"id of the RoBERTa in {roberta_path}")
    return MeanPool(
        RobertaInput(transformer, id_map, config.bos_token_id, config.eos_token_id),
        config.hidden_size,
    ).train()


def max_tokens(encoder: nn.Module) -> int:
    """The largest T of the token ids (B, T) a LastToken or MeanPool encoder reads."""
    transformer = encoder.transformer
    if isinstance(transformer, RobertaInput):
        return transformer.max_tokens
    return transformer.config.max_position_embeddings


def check_lengths(mode: Mode, history_tokens: int, future_tokens: int) -> None:
    """Reject sequences that do not fit the positions of the encoder of their side."""
    for side, name, encoder, tokens in (
        ("history", "left", mode.left_encoder, history_tokens),
        ("future", "right", mode.right_encoder, future_tokens),
    ):
        longest = max_tokens(encoder)
        if tokens > longest:
            raise ValueError(
                f"{side} sequences of {tokens} tokens do not fit the {name} encoder, which "
                f"reads at most {longest} tokens besides the ones it adds itself (history "
                f"{history_tokens}, future {future_tokens} tokens)")


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
    roberta_path: Path | None = None,
) -> Mode:
    """One mode with two encoders of kind `subnet` and their two heads.

    `scratch` starts each side at a random BERT transformer, `roberta` at the
    pretrained RoBERTa in roberta_path, `gpt2` at a copy of the teacher's
    transformer.
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
    elif subnet == "roberta":
        if roberta_path is None:
            raise ValueError("subnet 'roberta' needs roberta_path")
        left_encoder, right_encoder = (
            roberta_encoder(roberta_path, inference.model_path, inference.vocab_size)
            for _ in range(2))
    else:
        width = inference.model.config.hidden_size
        left_encoder, right_encoder = (
            LastToken(trainable_encoder(inference), width) for _ in range(2))
    embedding = inference.model.get_output_embeddings().weight.shape[1]
    return Mode(
        left_encoder, right_encoder,
        linear_head(left_encoder.width, 1), linear_head(right_encoder.width, embedding))
