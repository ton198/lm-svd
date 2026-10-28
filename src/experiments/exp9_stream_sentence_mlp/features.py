import math
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from utils.inference import CausalLMInference
from utils.stream_neural_svd import centered_unembedding


class FrozenSentenceEncoder:
    def __init__(
        self,
        encoder_path: Path,
        teacher_path: Path,
        device: torch.device,
        max_seq_length: int,
    ) -> None:
        if not encoder_path.is_dir():
            raise ValueError(f"sentence encoder directory does not exist: {encoder_path}")

        try:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                str(teacher_path), use_fast=True, local_files_only=True
            )
        except OSError as error:
            raise ValueError(
                f"cannot load local tokenizer from {teacher_path}: {error}"
            ) from error
        try:
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(
                str(encoder_path), use_fast=True, local_files_only=True
            )
        except OSError as error:
            raise ValueError(
                f"cannot load local tokenizer from {encoder_path}: {error}"
            ) from error
        try:
            model = AutoModel.from_pretrained(
                str(encoder_path), local_files_only=True, dtype=torch.float32
            )
        except OSError as error:
            raise ValueError(
                f"cannot load local sentence encoder from {encoder_path}: {error}"
            ) from error

        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.max_seq_length = max_seq_length
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        texts = self.teacher_tokenizer.batch_decode(
            ids.cpu().tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        batch = self.encoder_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=-1)


class FrozenSentenceMLPFeatures(nn.Module):
    def __init__(
        self,
        inference: CausalLMInference,
        *,
        encoder_path: Path,
        max_seq_length: int,
        rank: int,
        mlp_width: int,
        token_rms: float,
    ) -> None:
        super().__init__()
        object.__setattr__(
            self,
            "_encoder",
            FrozenSentenceEncoder(
                encoder_path, inference.model_path, inference.device, max_seq_length
            ),
        )
        hidden_size = inference.model.get_output_embeddings().weight.shape[1]
        sentence_dim = self._encoder.dim
        self.config = {
            "kind": "frozen_sentence_mlp",
            "encoder_path": str(encoder_path),
            "max_seq_length": max_seq_length,
            "sentence_dim": sentence_dim,
            "mlp_width": mlp_width,
            "hidden_size": hidden_size,
            "rank": rank,
        }
        self.left = nn.Sequential(
            nn.LayerNorm(sentence_dim, elementwise_affine=False),
            nn.Linear(sentence_dim, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, rank),
        )
        self.right = nn.Sequential(
            nn.LayerNorm(sentence_dim, elementwise_affine=False),
            nn.Linear(sentence_dim, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, rank * hidden_size),
        )
        self.register_buffer(
            "right_scale", torch.tensor(1 / (math.sqrt(hidden_size) * token_rms))
        )

    def row_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        return self.left(self._encoder.encode(history_ids))

    def column_weights(self, future_ids: torch.Tensor) -> torch.Tensor:
        return self.right(self._encoder.encode(future_ids)).view(
            future_ids.shape[0], self.config["rank"], self.config["hidden_size"]
        ) * self.right_scale


def build_features(config: dict, inference: CausalLMInference) -> nn.Module:
    token_rms = centered_unembedding(inference).pow(2).mean().sqrt().item()
    return FrozenSentenceMLPFeatures(
        inference,
        encoder_path=Path(config["encoder_path"]),
        max_seq_length=config["max_seq_length"],
        rank=config["rank"],
        mlp_width=config["mlp_width"],
        token_rms=token_rms,
    )
