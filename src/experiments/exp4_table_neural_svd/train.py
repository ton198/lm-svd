"""Train NeuralSVD on the exp4 training table and compare it with the exact SVDs.

Every step draws random rows and random columns of the training table and
feeds the full sub-table between them to the NestedLoRA loss. At each
evaluation the learned features of every row and column of both tables are
compared with that table's exact SVD.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from utils.common import save_json, write_atomic
from utils.linalg import matrix_svd
from utils.nestedlora import NestedLoRA

from . import CENTERING_CHOICES, SVD_RANK, TABLE_NAMES

RANKS = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class Table:
    histories: torch.Tensor  # (N, H) int64
    columns: torch.Tensor  # (C, F + 1) int64: a future followed by one header token
    matrix: torch.Tensor  # (N, C) float32
    singular_values: torch.Tensor  # (min(N, C),) float64, descending
    left_vectors: torch.Tensor  # (N, R) float64
    right_vectors: torch.Tensor  # (C, R) float64


def table_columns(futures: torch.Tensor, header_token_ids: torch.Tensor) -> torch.Tensor:
    """Token sequence of every matrix column, in the matrix's future-major order."""
    top_k = header_token_ids.shape[1]
    return torch.cat(
        (futures[:, None, :].expand(-1, top_k, -1), header_token_ids[:, :, None]), dim=-1
    ).flatten(0, 1)


def load_table(
    table_dir: Path, device: str | torch.device, centering: str = "none"
) -> Table:
    """Load a table and its exact SVD.

    With centering "column-mean", every column loses its mean over the table's
    own histories. That removes the leading mode, which is close to a constant
    over histories times the column means. The saved SVD belongs to the stored
    matrix, so the centered matrix gets a fresh exact SVD here.
    """
    if centering not in CENTERING_CHOICES:
        raise ValueError(f"centering must be one of {CENTERING_CHOICES}, got {centering!r}")
    tokens = np.load(table_dir / "tokens.npz")
    matrix = np.load(table_dir / "matrix.npy")
    if centering == "none":
        saved = np.load(table_dir / "svd.npz")
        singular_values, left, right_t = saved["S"], saved["U"], saved["Vt"]
    else:
        matrix = (matrix - matrix.mean(axis=0, dtype=np.float64)).astype(np.float32)
        svd = matrix_svd(matrix, device=device)
        singular_values, left, right_t = svd.S, svd.U[:, :SVD_RANK], svd.Vt[:SVD_RANK]
    return Table(
        histories=torch.from_numpy(tokens["histories"]).to(device),
        columns=table_columns(
            torch.from_numpy(tokens["futures"]),
            torch.from_numpy(tokens["header_token_ids"]).long(),
        ).to(device),
        matrix=torch.from_numpy(matrix).to(device),
        singular_values=torch.from_numpy(singular_values).to(device),
        left_vectors=torch.from_numpy(left).to(device),
        right_vectors=torch.from_numpy(right_t.T.copy()).to(device),
    )


def sample_subtable(
    table: Table, history_batch: int, column_batch: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw distinct rows and distinct columns; return h, columns and K (rows, columns)."""
    device = table.matrix.device
    rows = torch.randperm(table.matrix.shape[0], generator=generator)[:history_batch].to(device)
    columns = torch.randperm(table.matrix.shape[1], generator=generator)[:column_batch].to(device)
    return table.histories[rows], table.columns[columns], table.matrix[rows][:, columns]


@torch.no_grad()
def table_features(
    model: NestedLoRA, table: Table, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Learned features of every row (N, L) and every column (C, L), in float64."""
    model.eval()
    # The model maps both sides at once, so the other side gets a one-row placeholder.
    left = torch.cat([
        model(table.histories[start:start + chunk], table.columns[:1])[0].double()
        for start in range(0, len(table.histories), chunk)
    ])
    right = torch.cat([
        model(table.histories[:1], table.columns[start:start + chunk])[1].double()
        for start in range(0, len(table.columns), chunk)
    ])
    return left, right


@torch.no_grad()
def compare_with_exact_svd(
    left: torch.Tensor, right: torch.Tensor, table: Table,
    ranks: tuple[int, ...] = RANKS, chunk: int = 256,
) -> dict[str, list]:
    """Compare learned features left (N, L), right (C, L) with the table's exact SVD.

    Mode l's learned singular value is ||left[:, l]|| * ||right[:, l]||. For
    each rank k the first k learned modes are compared with the leading k exact
    modes: the subspace overlap is ||U_k^T Q_k||_F^2 / k with Q_k an
    orthonormal basis of the learned features, and the captured energy is
    1 - ||A - left_k right_k^T||^2 / ||A||^2, whose upper bound is the exact
    SVD's sum of the leading k squared singular values over ||A||^2.
    """
    left, right = left.double(), right.double()
    exact = table.singular_values
    modes = min(left.shape[1], exact.numel())
    learned = left.norm(dim=0) * right.norm(dim=0)

    # ||A - L_k R_k^T||^2 = ||A||^2 - 2 tr(L_k^T A R_k) + <L_k^T L_k, R_k^T R_k>.
    norm_sq = left.new_zeros(())
    projection = left.new_zeros((left.shape[1], left.shape[1]))
    for start in range(0, table.matrix.shape[0], chunk):
        block = table.matrix[start:start + chunk].double()
        norm_sq += block.square().sum()
        projection += left[start:start + chunk].T @ (block @ right)
    left_gram, right_gram = left.T @ left, right.T @ right
    # The leading k columns of Q span the leading k feature columns.
    left_basis, right_basis = torch.linalg.qr(left).Q, torch.linalg.qr(right).Q

    kept = min(left.shape[1], table.left_vectors.shape[1])
    result: dict[str, list] = {
        "singular_values": learned[:modes].tolist(),
        "singular_value_relative_error": (learned[:modes] / exact[:modes] - 1).tolist(),
        "ranks": [rank for rank in ranks if rank <= kept],
        "left_subspace_overlap": [],
        "right_subspace_overlap": [],
        "captured_energy": [],
        "exact_captured_energy": [],
    }
    for rank in result["ranks"]:
        for name, exact_vectors, basis in (
            ("left_subspace_overlap", table.left_vectors, left_basis),
            ("right_subspace_overlap", table.right_vectors, right_basis),
        ):
            overlap = (exact_vectors[:, :rank].T @ basis[:, :rank]).square().sum() / rank
            result[name].append(overlap.item())
        residual = (
            norm_sq - 2 * projection[:rank, :rank].trace()
            + (left_gram[:rank, :rank] * right_gram[:rank, :rank]).sum()
        )
        result["captured_energy"].append((1 - residual / norm_sq).item())
        result["exact_captured_energy"].append(
            (exact[:rank].square().sum() / norm_sq).item()
        )
    return result


def format_comparison(name: str, comparison: dict[str, list]) -> str:
    lines = [
        f"  {name:<5} rank  sv_rel_err  left_overlap  right_overlap    energy  exact_energy"
    ]
    for index, rank in enumerate(comparison["ranks"]):
        lines.append(
            f"        {rank:4d}  {comparison['singular_value_relative_error'][rank - 1]:+10.4f}"
            f"  {comparison['left_subspace_overlap'][index]:12.4f}"
            f"  {comparison['right_subspace_overlap'][index]:13.4f}"
            f"  {comparison['captured_energy'][index]:8.5f}"
            f"  {comparison['exact_captured_energy'][index]:12.5f}"
        )
    return "\n".join(lines)


def _write_history(path: Path, history: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in history), encoding="utf-8")


def train_exp4(
    *,
    inference_dir: Path,
    output_dir: Path,
    device: str,
    neigs: int,
    history_batch: int,
    column_batch: int,
    steps: int,
    eval_every: int,
    lr: float,
    seed: int,
    overwrite: bool,
    centering: str = "none",
    eval_chunk: int = 2048,
    model: NestedLoRA | None = None,
) -> dict:
    """Train on the training table; every eval_every steps compare both tables.

    Writes checkpoint.pt, eval_history.jsonl (one line per evaluation) and
    metadata.json to output_dir. An existing checkpoint with the same settings
    is continued up to `steps`; pass overwrite to start again. `centering`
    is applied to both tables before training and comparison, see load_table.
    `model` replaces the default svd_inference network.
    """
    if not (inference_dir / "metadata.json").is_file():
        raise ValueError(f"no finished tables in {inference_dir}; run exp4 inference first")
    tables = {
        name: load_table(inference_dir / name, device, centering) for name in TABLE_NAMES
    }
    train = tables["train"]
    rows, columns = train.matrix.shape
    if history_batch > rows or column_batch > columns:
        raise ValueError(
            f"batch {history_batch} x {column_batch} exceeds the training table {rows} x {columns}"
        )

    config = {
        "inference_dir": str(inference_dir), "neigs": neigs, "history_batch": history_batch,
        "column_batch": column_batch, "lr": lr, "seed": seed, "optimizer": "Adam",
        "centering": centering,
    }
    checkpoint_path = output_dir / "checkpoint.pt"
    history_path = output_dir / "eval_history.jsonl"
    if overwrite:
        for path in (checkpoint_path, history_path):
            path.unlink(missing_ok=True)

    torch.manual_seed(seed)
    if model is None:
        from experiments.svd_inference.model import get_nested_lora_network

        model = get_nested_lora_network(neigs=neigs)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    step = 0
    history: list[dict] = []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError(
                f"checkpoint settings {checkpoint['config']} differ from {config}; "
                "pass --overwrite to start over"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        generator.set_state(checkpoint["generator_state"])
        step, history = checkpoint["step"], checkpoint["history"]
        print(f"Resuming from step {step}", flush=True)
    elif history_path.exists():
        raise FileExistsError(
            f"output already exists: {history_path}; pass --overwrite to replace it"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(output_dir / "metadata.json", lambda tmp: save_json(tmp, {
        **config,
        "device": device,
        "sampling": "random distinct rows x random distinct columns of the training table",
        "ranks": list(RANKS),
        "table_shapes": {name: list(table.matrix.shape) for name, table in tables.items()},
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }))
    print(
        f"Training on {device}: {rows} x {columns} table, "
        f"{history_batch} x {column_batch} sub-table per step",
        flush=True,
    )

    loss_sum = energy_sum = 0.0
    window = 0
    started = time.perf_counter()
    while step < steps:
        model.train()
        h, column_tokens, kernel = sample_subtable(train, history_batch, column_batch, generator)
        optimizer.zero_grad()
        left, right = model(h, column_tokens)
        loss = model.compute_loss_kernel(left, right, kernel)[0]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        step += 1
        loss_sum += loss.item()
        energy_sum += kernel.square().mean().item()
        window += 1

        if step % eval_every == 0 or step == steps:
            # A perfect fit of every mode brings the loss to about -E[K^2].
            record = {
                "step": step,
                "train_loss": loss_sum / window,
                "kernel_energy": energy_sum / window,
            }
            lines = [
                f"step {step}: loss {record['train_loss']:.4f}  "
                f"E[K^2] {record['kernel_energy']:.4f}  "
                f"elapsed {time.perf_counter() - started:.0f}s"
            ]
            for name, table in tables.items():
                record[name] = compare_with_exact_svd(
                    *table_features(model, table, eval_chunk), table
                )
                lines.append(format_comparison(name, record[name]))
            print("\n".join(lines), flush=True)
            history.append(record)
            loss_sum = energy_sum = 0.0
            window = 0
            write_atomic(checkpoint_path, lambda tmp: torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "step": step,
                "history": history,
                "config": config,
            }, tmp))
            write_atomic(history_path, lambda tmp: _write_history(tmp, history))
    return {"output_dir": str(output_dir), "step": step, "evaluations": len(history)}
