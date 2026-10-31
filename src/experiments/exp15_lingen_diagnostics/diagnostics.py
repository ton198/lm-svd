from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.common import save_json, write_atomic
from utils.lingen import compare_distributions


TOPK_KEYS = ("1", "5", "10", "50")


@torch.no_grad()
def teacher_hidden_grid(
    inference, history_ids: torch.Tensor, future_ids: torch.Tensor, *, pair_batch: int
) -> torch.Tensor:
    history_ids = history_ids.to(inference.device)
    future_ids = future_ids.to(inference.device)
    history_count, history_length = history_ids.shape
    future_count, future_length = future_ids.shape
    hidden_size = inference.model.get_output_embeddings().weight.shape[1]
    if future_length == 0:
        result = torch.empty((1, history_count, hidden_size), dtype=torch.float32, device=inference.device)
        for start in range(0, history_count, pair_batch):
            result[0, start : start + pair_batch] = inference.final_hidden_states(
                history_ids[start : start + pair_batch]
            )[:, -1]
        return result

    result = torch.empty(
        (future_count, history_count, hidden_size), dtype=torch.float32, device=inference.device
    )
    future_batch = max(1, pair_batch // history_count)
    for start in range(0, future_count, future_batch):
        futures = future_ids[start : start + future_batch]
        count = len(futures)
        pairs = torch.cat(
            (
                history_ids[None, :, :].expand(count, -1, -1),
                futures[:, None, :].expand(-1, history_count, -1),
            ),
            dim=-1,
        ).reshape(count * history_count, history_length + future_length)
        result[start : start + count] = inference.final_hidden_states(pairs)[:, -1].reshape(
            count, history_count, hidden_size
        )
    return result


def history_gram(states: torch.Tensor, token_gram: torch.Tensor) -> torch.Tensor:
    gram = torch.zeros((states.shape[1], states.shape[1]), dtype=torch.float64, device=states.device)
    token_gram = token_gram.double()
    for state in states:
        state = state.double()
        gram += state @ token_gram @ state.T
    return gram


def new_metric_totals() -> dict[str, float]:
    return dict.fromkeys(
        ("pairs", "kl_pq_sum", "kl_qp_sum", "top1_sum", "top5_sum", "top10_sum",
         "top50_sum", "sq_error_sum", "sq_norm_sum", "relative_error_sum"),
        0.0,
    )


@torch.no_grad()
def accumulate_metrics(
    totals: dict[str, float], teacher_hidden: torch.Tensor, approx_hidden: torch.Tensor,
    token_matrix: torch.Tensor, *, temperature: float = 1.0,
) -> None:
    dtype = torch.promote_types(
        torch.promote_types(teacher_hidden.dtype, approx_hidden.dtype), token_matrix.dtype
    )
    tokens = token_matrix.to(dtype=dtype)
    teacher_logits = teacher_hidden.to(dtype=dtype) @ tokens.T
    approx_logits = temperature * (approx_hidden.to(dtype=dtype) @ tokens.T)
    metrics = compare_distributions(teacher_logits, approx_logits)
    totals["pairs"] += teacher_logits.shape[0]
    for key in ("kl_pq", "kl_qp"):
        totals[f"{key}_sum"] += metrics[key].sum().item()
    for index, key in enumerate(TOPK_KEYS):
        totals[f"top{key}_sum"] += metrics["topk_overlap"][:, index].sum().item()
    error, norm = metrics["logit_sq_error"], metrics["logit_sq_norm"]
    totals["sq_error_sum"] += error.sum().item()
    totals["sq_norm_sum"] += norm.sum().item()
    totals["relative_error_sum"] += (error / norm).sum().item()


def finalize_metrics(totals: dict[str, float]) -> dict:
    pairs = totals["pairs"]
    return {
        "pairs": int(pairs),
        "kl_pq": totals["kl_pq_sum"] / pairs,
        "kl_qp": totals["kl_qp_sum"] / pairs,
        "topk_overlap": {key: totals[f"top{key}_sum"] / pairs for key in TOPK_KEYS},
        "logit_relative_error_mean": totals["relative_error_sum"] / pairs,
        "logit_relative_error_aggregate": totals["sq_error_sum"] / totals["sq_norm_sum"],
    }


def orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    if s.size == 0 or s[0] == 0:
        raise ValueError("matrix has zero rank")
    return u[:, s > s[0] * max(matrix.shape) * np.finfo(np.float64).eps]


def optimal_left(weights: torch.Tensor, states: torch.Tensor, token_gram: torch.Tensor) -> np.ndarray:
    token_gram = token_gram.double()
    rank, histories = weights.shape[1], states.shape[1]
    b = torch.zeros((rank, rank), dtype=torch.float64, device=weights.device)
    c = torch.zeros((rank, histories), dtype=torch.float64, device=weights.device)
    for weight, state in zip(weights, states):
        weight, state = weight.double(), state.double()
        weighted = weight @ token_gram
        b += weighted @ weight.T
        c += weighted @ state.T
    return np.linalg.lstsq(b.cpu().numpy(), c.cpu().numpy(), rcond=None)[0].T


def fit_diagonal_scale(
    rows: torch.Tensor, weights: torch.Tensor, states: torch.Tensor, token_gram: torch.Tensor
) -> np.ndarray:
    rows, token_gram = rows.double(), token_gram.double()
    rank = rows.shape[1]
    weight_gram = torch.zeros((rank, rank), dtype=torch.float64, device=rows.device)
    b = torch.zeros(rank, dtype=torch.float64, device=rows.device)
    for weight, state in zip(weights, states):
        weight, state = weight.double(), state.double()
        weighted = weight @ token_gram
        weight_gram += weighted @ weight.T
        b += (rows * (state @ weighted.T)).sum(dim=0)
    m = (rows.T @ rows) * weight_gram
    return np.linalg.lstsq(m.cpu().numpy(), b.cpu().numpy(), rcond=None)[0]


def golden_section_minimize(
    objective: Callable[[float], float], lower: float, upper: float, *, iterations: int = 40
) -> float:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    f_left, f_right = objective(left), objective(right)
    for _ in range(iterations):
        if f_left < f_right:
            upper, right, f_right = right, left, f_left
            left = upper - ratio * (upper - lower)
            f_left = objective(left)
        else:
            lower, left, f_left = left, right, f_right
            right = lower + ratio * (upper - lower)
            f_right = objective(right)
    return (lower + upper) / 2.0


def fit_temperature(
    teacher_logits_blocks: Callable[[], Iterable[torch.Tensor]],
    approx_logits_blocks: Callable[[], Iterable[torch.Tensor]],
) -> float:
    def objective(log_temperature: float) -> float:
        total, count = 0.0, 0
        for teacher, approx in zip(teacher_logits_blocks(), approx_logits_blocks()):
            kl = compare_distributions(teacher, math.exp(log_temperature) * approx)["kl_pq"]
            total += kl.sum().item()
            count += kl.numel()
        return total / count

    return math.exp(golden_section_minimize(objective, math.log(0.25), math.log(4.0)))


def write_outputs(output_dir: Path, summary: dict, figure, metadata: dict, *, plot_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(output_dir / "summary.json", lambda tmp: save_json(tmp, summary))
    write_atomic(output_dir / plot_name, lambda tmp: figure.savefig(tmp, format="png", dpi=150))
    plt.close(figure)
    write_atomic(output_dir / "metadata.json", lambda tmp: save_json(tmp, metadata))


def prepare_output_dir(output_dir: Path, names: tuple[str, ...], overwrite: bool) -> None:
    path = output_dir / "metadata.json"
    if not overwrite and path.exists():
        raise FileExistsError(f"output already exists: {path}; pass --overwrite to replace it")
    if overwrite:
        for name in names:
            (output_dir / name).unlink(missing_ok=True)
