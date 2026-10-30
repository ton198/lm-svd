import math

import torch
from torch import nn

from utils.inference import CausalLMInference
from utils.stream_neural_svd import centered_unembedding


class StudentLM(nn.Module):
    def __init__(
        self,
        token_matrix: torch.Tensor,
        *,
        model_dim: int,
        layers: int,
        heads: int,
        max_positions: int,
    ) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError(
                f"model_dim {model_dim} must be divisible by heads {heads}"
            )
        object.__setattr__(self, "_token_matrix", token_matrix)
        self._token_rms = token_matrix.pow(2).mean().sqrt().item()
        self.max_positions = max_positions
        hidden = token_matrix.shape[1]
        self.in_proj = nn.Linear(hidden, model_dim)
        self.position_embedding = nn.Embedding(max_positions, model_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                model_dim,
                heads,
                4 * model_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, hidden)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        if length > self.max_positions:
            raise ValueError(
                f"window of {length} tokens exceeds student max_positions {self.max_positions}"
            )
        positions = torch.arange(length, device=ids.device)
        x = self.in_proj(self._token_matrix[ids] / self._token_rms)
        x = x + self.position_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(length, device=ids.device)
        return self.norm(self.transformer(x, mask=mask, is_causal=True))

    def predicted_hidden(self, states: torch.Tensor) -> torch.Tensor:
        return self.out_proj(states)


def build_student(config: dict, token_matrix: torch.Tensor) -> StudentLM:
    return StudentLM(
        token_matrix,
        model_dim=config["model_dim"],
        layers=config["layers"],
        heads=config["heads"],
        max_positions=config["max_positions"],
    )


class StudentSVDFeatures(nn.Module):
    def __init__(
        self,
        token_matrix: torch.Tensor,
        *,
        student: dict,
        rank: int,
        column_dim: int,
        skip: bool,
        init: str,
        student_checkpoint: str | None,
    ) -> None:
        super().__init__()
        if skip and rank < 2:
            raise ValueError("skip requires rank >= 2")
        object.__setattr__(self, "_token_matrix", token_matrix)
        self.trunk = build_student(student, token_matrix)
        hidden = token_matrix.shape[1]
        model_dim = student["model_dim"]
        n = rank - 1 if skip else rank
        self.row_head = nn.Linear(2 * model_dim, n)
        self.column_head = nn.Linear(2 * model_dim, n * column_dim)
        self.column_basis = nn.Linear(column_dim, hidden, bias=False)
        if skip:
            self.skip_scale = nn.Parameter(torch.ones(()))
        self.register_buffer(
            "right_scale", torch.tensor(1 / (math.sqrt(hidden) * self.trunk._token_rms))
        )
        self.config = {
            "kind": "distilled_student_svd",
            "student": student,
            "rank": rank,
            "column_dim": column_dim,
            "skip": skip,
            "init": init,
            "student_checkpoint": student_checkpoint,
            "hidden_size": hidden,
        }

    def _summary(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.trunk(ids)
        return torch.cat([states[:, -1], states.mean(dim=1)], dim=-1), states

    def row_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        summary, _ = self._summary(history_ids)
        rows = self.row_head(summary)
        if self.config["skip"]:
            rows = torch.cat([self.skip_scale.expand(history_ids.shape[0], 1), rows], dim=1)
        return rows

    def column_weights(self, future_ids: torch.Tensor) -> torch.Tensor:
        summary, states = self._summary(future_ids)
        n = self.config["rank"] - 1 if self.config["skip"] else self.config["rank"]
        weights = self.column_head(summary).view(
            future_ids.shape[0], n, self.config["column_dim"]
        )
        weights = self.column_basis(weights) * self.right_scale
        if self.config["skip"]:
            skip_weights = self.trunk.predicted_hidden(states[:, -1])[:, None, :]
            weights = torch.cat([skip_weights, weights], dim=1)
        return weights


def build_features(config: dict, inference: CausalLMInference) -> nn.Module:
    token_matrix = centered_unembedding(inference)
    return StudentSVDFeatures(
        token_matrix,
        student=config["student"],
        rank=config["rank"],
        column_dim=config["column_dim"],
        skip=config["skip"],
        init=config["init"],
        student_checkpoint=config["student_checkpoint"],
    )
