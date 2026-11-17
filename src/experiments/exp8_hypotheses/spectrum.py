"""Study A of exp8: how much of a table's spectrum carries over to new samples.

A table's exact SVD always fits that table. Whether its modes are properties of
histories and futures in general, which is what a network has to learn, shows in
three split tests on one table:

- left transfer: the leading left vectors of one half of the futures, applied to
  the other half. It asks whether a function of the history found on some futures
  is still a singular function on other futures.
- right transfer: the leading right vectors of one half of the histories, applied
  to the other half.
- block prediction (bi-cross-validation, Owen and Perry 2009): the rank-k SVD of
  one block of histories and futures predicts the opposite block through the two
  blocks they share with it. No network is involved, so this is what an ideal
  estimator gets from that many samples.

Each test reports the captured share of the held-out energy next to the best
that rank can do on the held-out part itself.
"""

from __future__ import annotations

import torch

RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def thin_svd(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """U, S, V of a float64 matrix through the eigenvectors of its smaller Gram matrix.

    Singular values come out in descending order. Values far below the largest
    one are not accurate, which does not matter for the leading modes used here.
    """
    if matrix.shape[0] > matrix.shape[1]:
        right, values, left = thin_svd(matrix.T)
        return left, values, right
    eigenvalues, left = torch.linalg.eigh(matrix @ matrix.T)
    order = torch.argsort(eigenvalues, descending=True)
    values = eigenvalues[order].clamp_min(0).sqrt()
    left = left[:, order]
    floor = values[0] * 1e-7
    right = (matrix.T @ left) / values.clamp_min(floor)
    return left, values, right


def center_rows_out(matrix: torch.Tensor) -> torch.Tensor:
    """Subtract from every column its mean over the rows given."""
    return matrix - matrix.mean(dim=0, keepdim=True)


def _energy(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.square().sum()


def _kept(ranks: tuple[int, ...], limit: int) -> list[int]:
    return [rank for rank in ranks if rank <= limit]


def spectrum(matrix: torch.Tensor, ranks: tuple[int, ...] = RANKS) -> dict:
    """Share of the energy in the leading modes of the whole matrix."""
    values = thin_svd(matrix)[1].square()
    total = values.sum()
    kept = _kept(ranks, len(values))
    return {
        "ranks": kept,
        "energy": [(values[:rank].sum() / total).item() for rank in kept],
        # exp(entropy) of the normalised spectrum: 1 for one mode, n for a flat one.
        "effective_rank": torch.exp(
            -(values / total * (values / total).clamp_min(1e-300).log()).sum()
        ).item(),
    }


def subspace_transfer(
    fit: torch.Tensor, test: torch.Tensor, ranks: tuple[int, ...] = RANKS
) -> dict:
    """Leading left vectors of `fit` applied to `test`, which has the same rows.

    captured: ||U_k^T test||^2 / ||test||^2. optimum: the same for test's own
    leading left vectors. For a right transfer pass the transposes.
    """
    left = thin_svd(fit)[0]
    own = thin_svd(test)[1].square()
    total = _energy(test)
    projected = (left.T @ test).square().sum(dim=1)
    kept = _kept(ranks, min(len(own), left.shape[1]))
    return {
        "ranks": kept,
        "captured": [(projected[:rank].sum() / total).item() for rank in kept],
        "optimum": [(own[:rank].sum() / total).item() for rank in kept],
    }


def block_prediction(
    matrix: torch.Tensor,
    fit_rows: torch.Tensor,
    test_rows: torch.Tensor,
    fit_columns: torch.Tensor,
    test_columns: torch.Tensor,
    ranks: tuple[int, ...] = RANKS,
) -> dict:
    """Predict the (test rows, test columns) block from the rank-k SVD of the fit block.

    prediction_k = A[test, fit] V_k S_k^-1 U_k^T A[fit, test], with U, S, V the SVD
    of A[fit, fit]. captured is 1 - ||block - prediction_k||^2 / ||block||^2, which
    turns negative once the extra modes are noise.
    """
    left, values, right = thin_svd(matrix[fit_rows][:, fit_columns])
    held_out = matrix[test_rows][:, test_columns]
    own = thin_svd(held_out)[1].square()
    total = _energy(held_out)
    # Coordinates of the held-out rows and columns in the fit block's modes.
    row_part = matrix[test_rows][:, fit_columns] @ right  # (test rows, modes)
    column_part = left.T @ matrix[fit_rows][:, test_columns]  # (modes, test columns)
    kept = _kept(ranks, min(len(own), len(values)))
    captured = []
    for rank in kept:
        prediction = (row_part[:, :rank] / values[:rank]) @ column_part[:rank]
        captured.append((1 - _energy(held_out - prediction) / total).item())
    return {
        "ranks": kept,
        "captured": captured,
        "optimum": [(own[:rank].sum() / total).item() for rank in kept],
    }


def split_report(
    matrix: torch.Tensor,
    top_k: int,
    *,
    center: bool,
    seed: int = 0,
    ranks: tuple[int, ...] = RANKS,
) -> dict:
    """All three split tests on one table whose columns are future-major.

    Histories and futures are each split into two random halves; a future keeps
    its top_k columns together. With center, every block loses its column means
    over its own histories, so no half sees the other half's means.
    """
    rows, columns = matrix.shape
    if columns % top_k:
        raise ValueError(f"{columns} columns are not a multiple of top_k {top_k}")
    futures = columns // top_k
    generator = torch.Generator().manual_seed(seed)
    row_order = torch.randperm(rows, generator=generator).to(matrix.device)
    future_order = torch.randperm(futures, generator=generator).to(matrix.device)
    slots = torch.arange(top_k, device=matrix.device)

    def columns_of(chosen: torch.Tensor) -> torch.Tensor:
        return (chosen[:, None] * top_k + slots).reshape(-1)

    rows_a, rows_b = row_order[: rows // 2], row_order[rows // 2:]
    columns_a = columns_of(future_order[: futures // 2])
    columns_b = columns_of(future_order[futures // 2:])
    prepare = center_rows_out if center else (lambda block: block)

    whole = prepare(matrix)
    # Row halves are centered apart, so the block test rebuilds its four blocks.
    stacked = torch.empty_like(matrix)
    stacked[rows_a] = prepare(matrix[rows_a])
    stacked[rows_b] = prepare(matrix[rows_b])
    return {
        "shape": [rows, columns],
        "centered": center,
        "spectrum": spectrum(whole, ranks),
        "left_transfer": subspace_transfer(whole[:, columns_a], whole[:, columns_b], ranks),
        "right_transfer": subspace_transfer(stacked[rows_a].T, stacked[rows_b].T, ranks),
        "block_prediction": block_prediction(
            stacked, rows_a, rows_b, columns_a, columns_b, ranks
        ),
    }


def format_report(name: str, report: dict) -> str:
    lines = [f"{name}  shape {report['shape']}  "
             f"effective rank {report['spectrum']['effective_rank']:.1f}"]
    lines.append("  rank  spectrum | left: captured optimum | right: captured optimum "
                 "| block: captured optimum")
    tests = [report[key] for key in ("left_transfer", "right_transfer", "block_prediction")]
    for index, rank in enumerate(report["spectrum"]["ranks"]):
        cells = [f"  {rank:4d}  {report['spectrum']['energy'][index]:8.4f}"]
        for test in tests:
            if index < len(test["ranks"]):
                cells.append(f"{test['captured'][index]:9.4f} {test['optimum'][index]:7.4f}")
            else:
                cells.append(" " * 17)
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def run_spectrum(*, inference_dir, model_path, device: str, output_dir, seeds: int = 2) -> dict:
    """Split reports of both exp4 tables: stored, minus g, and minus the column means."""
    import json
    from pathlib import Path

    import numpy as np

    from experiments.exp4_table_neural_svd import TABLE_NAMES
    from utils.common import write_atomic
    from utils.inference import CausalLMInference

    from .grid import Teacher

    inference_dir, output_dir = Path(inference_dir), Path(output_dir)
    if not (inference_dir / "metadata.json").is_file():
        raise ValueError(f"no finished tables in {inference_dir}; run exp4 inference first")
    teacher = Teacher(CausalLMInference(model_path, device))
    reports: dict = {}
    for name in TABLE_NAMES:
        tokens = np.load(inference_dir / name / "tokens.npz")
        header_ids = torch.from_numpy(tokens["header_token_ids"]).long().to(device)
        g = teacher.future_only(torch.from_numpy(tokens["futures"]).to(device), header_ids)
        stored = torch.from_numpy(np.load(inference_dir / name / "matrix.npy")).to(device).double()
        minus_g = stored - g.reshape(-1).double()
        for label, matrix, center in (
            ("stored", stored, False), ("minus-g", minus_g, False),
            ("minus-column-mean", stored, True),
        ):
            for seed in range(seeds):
                report = split_report(matrix, header_ids.shape[1], center=center, seed=seed)
                reports[f"{name}/{label}/seed{seed}"] = report
                print(format_report(f"{name} table, {label}, split {seed}", report), flush=True)
        del stored, minus_g
    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(
        output_dir / "spectrum.json",
        lambda tmp: tmp.write_text(json.dumps(reports, indent=1), encoding="utf-8"))
    return {"output_dir": str(output_dir), "reports": len(reports)}
