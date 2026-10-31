import importlib

import torch
from torch import nn

from experiments.exp11_stream_student.features import build_student


SOURCE_MODULES = {
    "frozen_gpt2_mlp": "experiments.exp6_stream_gpt2_mlp.features",
    "frozen_sentence_mlp": "experiments.exp9_stream_sentence_mlp.features",
    "distilled_student_svd": "experiments.exp11_stream_student.features",
}


class WideRight(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, *, width: int, blocks: int,
        block_hidden: int, bottleneck: int,
    ) -> None:
        super().__init__()
        self.inp = nn.Sequential(
            nn.LayerNorm(in_dim, elementwise_affine=False),
            nn.Linear(in_dim, width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, block_hidden),
                nn.GELU(),
                nn.Linear(block_hidden, width),
            )
            for _ in range(blocks)
        )
        self.out = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp(x)
        for block in self.blocks:
            x = x + block(x)
        return self.out(x)


class RightRefitFeatures(nn.Module):
    def __init__(self, inference, config: dict) -> None:
        super().__init__()
        self.config = config
        source = config["source_features"]
        kind = source["kind"]
        if kind not in SOURCE_MODULES:
            raise ValueError(f"unsupported source features kind: {kind}")
        self.base = importlib.import_module(SOURCE_MODULES[kind]).build_features(
            source, inference
        )
        self.base.requires_grad_(False)
        if config["variant"] == "continue":
            if kind == "distilled_student_svd":
                self.base.column_head.requires_grad_(True)
                self.base.column_basis.requires_grad_(True)
                self.base.trunk.out_proj.requires_grad_(True)
            else:
                self.base.right.requires_grad_(True)
        elif kind == "distilled_student_svd":
            self.base.column_head = None
            self.base.column_basis = None
            self.column_trunk = build_student(source["student"], self.base._token_matrix)
            model_dim = source["student"]["model_dim"]
            head_width = config["right"]["head_width"]
            n = source["rank"] - 1 if source["skip"] else source["rank"]
            self.column_head = nn.Sequential(
                nn.LayerNorm(2 * model_dim),
                nn.Linear(2 * model_dim, head_width),
                nn.GELU(),
                nn.Linear(head_width, head_width),
                nn.GELU(),
                nn.Linear(head_width, n * source["hidden_size"]),
            )
        else:
            self.base.right = None
            in_dim = (
                2 * source["hidden_size"]
                if kind == "frozen_gpt2_mlp"
                else source["sentence_dim"]
            )
            self.right = WideRight(
                in_dim, source["rank"] * source["hidden_size"], **config["right"]
            )

    def row_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        return self.base.row_features(history_ids)

    def column_weights(self, future_ids: torch.Tensor) -> torch.Tensor:
        if self.config["variant"] == "continue":
            return self.base.column_weights(future_ids)
        source = self.config["source_features"]
        kind = source["kind"]
        if kind == "distilled_student_svd":
            states = self.column_trunk(future_ids)
            summary = torch.cat([states[:, -1], states.mean(dim=1)], dim=-1)
            n = source["rank"] - 1 if source["skip"] else source["rank"]
            weights = self.column_head(summary).view(
                future_ids.shape[0], n, source["hidden_size"]
            ) * self.base.right_scale
            if source["skip"]:
                skip = self.column_trunk.predicted_hidden(states[:, -1])[:, None, :]
                weights = torch.cat([skip, weights], dim=1)
            return weights
        encoded = (
            self.base.encode(future_ids)
            if kind == "frozen_gpt2_mlp"
            else self.base._encoder.encode(future_ids)
        )
        return self.right(encoded).view(
            future_ids.shape[0], source["rank"], source["hidden_size"]
        ) * self.base.right_scale

    def load_source_state(self, state: dict[str, torch.Tensor]) -> None:
        if self.config["variant"] == "continue":
            self.base.load_state_dict(state, strict=True)
            return
        if self.config["source_features"]["kind"] == "distilled_student_svd":
            self.base.load_state_dict(
                {k: v for k, v in state.items() if not k.startswith(("column_head.", "column_basis."))},
                strict=True,
            )
            self.column_trunk.load_state_dict(
                {k.removeprefix("trunk."): v for k, v in state.items() if k.startswith("trunk.")},
                strict=True,
            )
        else:
            self.base.load_state_dict(
                {k: v for k, v in state.items() if not k.startswith("right.")},
                strict=True,
            )


def build_features(config: dict, inference) -> nn.Module:
    return RightRefitFeatures(inference, config)
