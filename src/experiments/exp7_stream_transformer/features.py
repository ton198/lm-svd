import math

import torch
from torch import nn

from utils.inference import CausalLMInference
from utils.stream_neural_svd import centered_unembedding


class TokenEncoder(nn.Module):
    def __init__(
        self, vocab_size: int, max_length: int, model_dim: int, layers: int, heads: int
    ) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError(
                f"model_dim {model_dim} must be divisible by heads {heads}"
            )
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_length, model_dim)
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

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        if length != self.max_length:
            raise ValueError(
                f"expected windows of {self.max_length} tokens, got {length}"
            )
        positions = torch.arange(length, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        x = self.norm(self.transformer(x))
        return torch.cat([x[:, -1], x.mean(dim=1)], dim=-1)


class SmallTransformerFeatures(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        model_dim: int,
        layers: int,
        heads: int,
        history_tokens: int,
        future_tokens: int,
        token_rms: float,
    ) -> None:
        super().__init__()
        self.history_encoder = TokenEncoder(
            vocab_size, history_tokens, model_dim, layers, heads
        )
        self.future_encoder = TokenEncoder(
            vocab_size, future_tokens, model_dim, layers, heads
        )
        self.row_head = nn.Linear(2 * model_dim, rank)
        self.column_head = nn.Linear(2 * model_dim, rank * hidden_size)
        self.register_buffer(
            "right_scale", torch.tensor(1 / (math.sqrt(hidden_size) * token_rms))
        )
        self.config = {
            "kind": "small_transformer",
            "model_dim": model_dim,
            "layers": layers,
            "heads": heads,
            "history_tokens": history_tokens,
            "future_tokens": future_tokens,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "rank": rank,
        }

    def row_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        return self.row_head(self.history_encoder(history_ids))

    def column_weights(self, future_ids: torch.Tensor) -> torch.Tensor:
        weights = self.column_head(self.future_encoder(future_ids))
        return weights.view(
            future_ids.shape[0], self.config["rank"], self.config["hidden_size"]
        ) * self.right_scale


def build_features(config: dict, inference: CausalLMInference) -> nn.Module:
    token_matrix = centered_unembedding(inference)
    token_rms = token_matrix.pow(2).mean().sqrt().item()
    return SmallTransformerFeatures(
        vocab_size=inference.vocab_size,
        hidden_size=token_matrix.shape[1],
        rank=config["rank"],
        model_dim=config["model_dim"],
        layers=config["layers"],
        heads=config["heads"],
        history_tokens=config["history_tokens"],
        future_tokens=config["future_tokens"],
        token_rms=token_rms,
    )
