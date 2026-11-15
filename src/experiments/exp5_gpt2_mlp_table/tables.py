"""exp4's stored tables as exp5 evaluates on them.

exp5 fixes the first mode to 1 x g, where g is the centered logit of each
header token after its future alone, and trains on what is left. The stored
tables therefore lose g from every row and get a fresh exact SVD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.exp4_table_neural_svd import SVD_RANK
from experiments.exp4_table_neural_svd.train import Table, table_columns
from utils.inference import CausalLMInference
from utils.linalg import matrix_svd
from utils.nestedlora import NestedLoRA

from . import HEADER_TOLERANCE
from .features import future_only_logits, last_hidden


@dataclass(frozen=True)
class EvalTable:
    table: Table  # the stored matrix minus g, with the exact SVD of that matrix
    history_hidden: torch.Tensor  # (N, hidden) float32
    future_hidden: torch.Tensor  # (F, hidden) float32, from the frozen model
    future_ids: torch.Tensor  # (F, T) int64, for a right side with its own encoder
    residual_energy: float  # ||matrix - g||^2 / ||matrix||^2
    residual_top_mode: float  # leading squared singular value over ||matrix - g||^2


@torch.no_grad()
def load_eval_table(
    table_dir: Path,
    inference: CausalLMInference,
    device: str | torch.device,
    batch_size: int = 1024,
) -> EvalTable:
    """Load an exp4 table, remove g and take the exact SVD of the result.

    g is read at the stored header tokens. exp4 selected them one future at a
    time, and a batched pass can order near-tied tokens differently, so they
    are not selected again; they only have to be among the model's top-k.
    """
    tokens = np.load(table_dir / "tokens.npz")
    matrix = np.load(table_dir / "matrix.npy")
    histories = torch.from_numpy(tokens["histories"])
    futures = torch.from_numpy(tokens["futures"])
    header_ids = torch.from_numpy(tokens["header_token_ids"]).long()
    top_k = header_ids.shape[1]

    history_hidden = last_hidden(inference, histories, batch_size)
    future_hidden = last_hidden(inference, futures, batch_size)
    centered = future_only_logits(inference, future_hidden)
    g = centered.gather(1, header_ids.to(centered.device))
    boundary = centered.topk(top_k, dim=1).values[:, -1]
    outside = int((g.min(dim=1).values < boundary - HEADER_TOLERANCE).sum())
    if outside:
        raise ValueError(
            f"the header tokens of {outside} future(s) in {table_dir} are not the top "
            f"{top_k} next tokens of {inference.model_path}; the table belongs to "
            "another model or top-k"
        )

    residual = matrix - g.reshape(-1).cpu().numpy()
    svd = matrix_svd(residual, device=device)
    residual_norm_sq = float(np.square(residual, dtype=np.float64).sum())
    return EvalTable(
        table=Table(
            histories=histories.to(device),
            columns=table_columns(futures, header_ids).to(device),
            matrix=torch.from_numpy(residual).to(device),
            singular_values=torch.from_numpy(svd.S).to(device),
            left_vectors=torch.from_numpy(svd.U[:, :SVD_RANK]).to(device),
            right_vectors=torch.from_numpy(svd.Vt[:SVD_RANK].T.copy()).to(device),
        ),
        history_hidden=history_hidden.to(device),
        future_hidden=future_hidden.to(device),
        future_ids=futures.to(device),
        residual_energy=residual_norm_sq / float(np.square(matrix, dtype=np.float64).sum()),
        residual_top_mode=float(svd.S[0]) ** 2 / residual_norm_sq,
    )


@torch.no_grad()
def eval_features(
    model: NestedLoRA, table: EvalTable, chunk: int = 2048
) -> tuple[torch.Tensor, torch.Tensor]:
    """Learned features of every row (N, L) and every column (C, L), in float64."""
    model.eval()
    network = model.model
    left = torch.cat([
        network.left_features(table.history_hidden[start:start + chunk]).double()
        for start in range(0, len(table.history_hidden), chunk)
    ])
    future_input = table.future_hidden if network.right_encoder is None else table.future_ids
    right = torch.cat([
        network.right_features(future_input[start:start + chunk]).double()
        for start in range(0, len(future_input), chunk)
    ])
    return left, right
