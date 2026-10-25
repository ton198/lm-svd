import math

import torch
from torch import nn

from utils.inference import CausalLMInference
from utils.stream_neural_svd import centered_unembedding


class ConcatTokenEncoder(nn.Module):
    def __init__(
        self, vocab_size: int, window_tokens: int, embedding_dim: int
    ) -> None:
        super().__init__()
        self.window_tokens = window_tokens
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        if length != self.window_tokens:
            raise ValueError(
                f"expected windows of {self.window_tokens} tokens, got {length}"
            )
        return self.token_embedding(ids).reshape(
            ids.shape[0], length * self.embedding_dim
        )


class TokenMLPFeatures(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        embedding_dim: int,
        mlp_width: int,
        history_tokens: int,
        future_tokens: int,
        token_rms: float,
    ) -> None:
        super().__init__()
        self.history_encoder = ConcatTokenEncoder(
            vocab_size, history_tokens, embedding_dim
        )
        self.future_encoder = ConcatTokenEncoder(
            vocab_size, future_tokens, embedding_dim
        )
        self.left = nn.Sequential(
            nn.LayerNorm(history_tokens * embedding_dim, elementwise_affine=False),
            nn.Linear(history_tokens * embedding_dim, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, rank),
        )
        self.right = nn.Sequential(
            nn.LayerNorm(future_tokens * embedding_dim, elementwise_affine=False),
            nn.Linear(future_tokens * embedding_dim, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, rank * hidden_size),
        )
        self.register_buffer(
            "right_scale", torch.tensor(1 / (math.sqrt(hidden_size) * token_rms))
        )
        self.config = {
            "kind": "token_mlp",
            "embedding_dim": embedding_dim,
            "mlp_width": mlp_width,
            "history_tokens": history_tokens,
            "future_tokens": future_tokens,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "rank": rank,
        }

    def row_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        return self.left(self.history_encoder(history_ids))

    def column_weights(self, future_ids: torch.Tensor) -> torch.Tensor:
        return self.right(self.future_encoder(future_ids)).view(
            future_ids.shape[0], self.config["rank"], self.config["hidden_size"]
        ) * self.right_scale


def build_features(config: dict, inference: CausalLMInference) -> nn.Module:
    token_matrix = centered_unembedding(inference)
    token_rms = token_matrix.pow(2).mean().sqrt().item()
    return TokenMLPFeatures(
        vocab_size=inference.vocab_size,
        hidden_size=token_matrix.shape[1],
        rank=config["rank"],
        embedding_dim=config["embedding_dim"],
        mlp_width=config["mlp_width"],
        history_tokens=config["history_tokens"],
        future_tokens=config["future_tokens"],
        token_rms=token_rms,
    )
