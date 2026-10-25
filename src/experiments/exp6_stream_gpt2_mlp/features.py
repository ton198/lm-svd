import math

import torch
from torch import nn

from utils.inference import CausalLMInference
from utils.stream_neural_svd import centered_unembedding


class FrozenGPT2MLPFeatures(nn.Module):
    def __init__(
        self,
        inference: CausalLMInference,
        *,
        rank: int,
        mlp_width: int,
        token_rms: float,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inference", inference)
        hidden_size = inference.model.get_output_embeddings().weight.shape[1]
        self.config = {
            "kind": "frozen_gpt2_mlp",
            "mlp_width": mlp_width,
            "hidden_size": hidden_size,
            "rank": rank,
        }
        self.left = nn.Sequential(
            nn.LayerNorm(2 * hidden_size, elementwise_affine=False),
            nn.Linear(2 * hidden_size, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, rank),
        )
        self.right = nn.Sequential(
            nn.LayerNorm(2 * hidden_size, elementwise_affine=False),
            nn.Linear(2 * hidden_size, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, rank * hidden_size),
        )
        self.register_buffer(
            "right_scale", torch.tensor(1 / (math.sqrt(hidden_size) * token_rms))
        )

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self._inference.final_hidden_states(ids)
        return torch.cat([hidden[:, -1], hidden.mean(dim=1)], dim=-1)

    def row_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        return self.left(self.encode(history_ids))

    def column_weights(self, future_ids: torch.Tensor) -> torch.Tensor:
        return self.right(self.encode(future_ids)).view(
            future_ids.shape[0], self.config["rank"], self.config["hidden_size"]
        ) * self.right_scale


def build_features(config: dict, inference: CausalLMInference) -> nn.Module:
    token_rms = centered_unembedding(inference).pow(2).mean().sqrt().item()
    return FrozenGPT2MLPFeatures(
        inference,
        rank=config["rank"],
        mlp_width=config["mlp_width"],
        token_rms=token_rms,
    )
