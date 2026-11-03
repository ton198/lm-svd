import json
from pathlib import Path

import numpy as np

from utils.analysis import (
    CENTERING_FORMULAS,
    RANK_TOLERANCE_RULE,
    compute_svd,
    format_log_log_fit,
    visualize_svd,
)
from utils.common import save_json, save_numpy, write_atomic
from utils.experiment_paths import ANALYSIS_DIR, INFERENCE_DIR


MATRIX_NAME = "pmi.npy"
FACTOR_NAMES = {"U": "svd_U.npy", "S": "svd_S.npy", "Vt": "svd_Vt.npy"}
METADATA_NAME = "metadata.json"
PLOT_NAME = "svd_spectrum.png"
TRUNCATION_PLOT_NAME = "pairing_truncation.png"
TRUNCATION_THRESHOLD = 0.9


def pairing_diagnostics(matrix: np.ndarray) -> dict:
    n = matrix.shape[0]
    diagonal = np.diag(matrix)
    off = matrix[~np.eye(n, dtype=bool)]
    row_rank = 1 + (matrix > diagonal[:, None]).sum(axis=1)
    column_rank = 1 + (matrix > diagonal[None, :]).sum(axis=0)
    diagonal_mean = float(np.mean(diagonal))
    off_mean = float(np.mean(off)) if n >= 2 else None
    return {
        "diagonal": "matched pairs: future i is the continuation of history i",
        "matrix": "stored uncentered PMI matrix; identical for every --centering",
        "diagonal_mean": diagonal_mean,
        "diagonal_std": float(np.std(diagonal)),
        "off_diagonal_mean": off_mean,
        "off_diagonal_std": float(np.std(off)) if n >= 2 else None,
        "mean_gap": diagonal_mean - off_mean if off_mean is not None else None,
        "row_top1": float(np.mean(row_rank == 1)),
        "column_top1": float(np.mean(column_rank == 1)),
        "row_mean_rank": float(np.mean(row_rank)),
        "column_mean_rank": float(np.mean(column_rank)),
        "rank_rule": "rank = 1 + number of entries in the row (column) strictly greater than the diagonal entry",
    }


def truncation_pairing(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> dict:
    n = U.shape[0]
    K = S.size
    reconstruction = np.zeros((n, n), dtype=np.float64)
    metrics = {
        "mean_gap": [],
        "row_top1": [],
        "column_top1": [],
        "row_mean_rank": [],
        "column_mean_rank": [],
    }
    for k in range(K):
        reconstruction += S[k] * np.outer(U[:, k], Vt[k])
        diagnostics = pairing_diagnostics(reconstruction)
        for key in metrics:
            metrics[key].append(diagnostics[key])

    energy = np.cumsum(S**2) / np.sum(S**2)
    error = np.sqrt(np.clip(1.0 - energy, 0.0, None))

    def first_rank(values, target):
        return next((r for r, value in enumerate(values, 1) if value >= target), K)

    return {
        "matrix": "rank-r truncation of the centered matrix: U[:, :r] @ diag(S[:r]) @ Vt[:r], r = 1..len(S)",
        "rank_rule": diagnostics["rank_rule"],
        "chance_top1": 1 / n,
        "chance_mean_rank": (n + 1) / 2,
        "ranks": list(range(1, K + 1)),
        "energy_fraction": energy.tolist(),
        "relative_reconstruction_error": error.tolist(),
        **metrics,
        "threshold": {
            "fraction": TRUNCATION_THRESHOLD,
            "rule": "smallest r whose value is >= fraction * value at r = len(S); energy uses fraction directly",
            "energy_rank": first_rank(energy, TRUNCATION_THRESHOLD),
            "row_top1_rank": first_rank(metrics["row_top1"], TRUNCATION_THRESHOLD * metrics["row_top1"][-1]),
            "column_top1_rank": first_rank(metrics["column_top1"], TRUNCATION_THRESHOLD * metrics["column_top1"][-1]),
        },
        "plot": TRUNCATION_PLOT_NAME,
    }


def plot_truncation_pairing(
    truncation: dict, *, n: int, centering: str, plot_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    try:
        ranks = truncation["ranks"]
        axes[0].plot(ranks, truncation["energy_fraction"], label="captured energy")
        axes[0].plot(ranks, truncation["row_top1"], label="row top-1")
        axes[0].plot(ranks, truncation["column_top1"], label="column top-1")
        axes[0].axhline(truncation["chance_top1"], linestyle=":", label="chance top-1")
        axes[0].set_xscale("log")
        axes[0].set_ylim(0, 1.01)
        axes[0].set_title("Energy vs matched-pair top-1")
        axes[0].set_xlabel("truncation rank r")
        axes[0].set_ylabel("fraction")
        axes[0].legend()

        axes[1].plot(ranks, truncation["row_mean_rank"], label="row mean rank")
        axes[1].plot(ranks, truncation["column_mean_rank"], label="column mean rank")
        axes[1].axhline(truncation["chance_mean_rank"], linestyle=":", label="chance mean rank")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_title("Matched-pair mean rank")
        axes[1].set_xlabel("truncation rank r")
        axes[1].set_ylabel("mean rank (1 = best)")
        axes[1].legend()
        fig.suptitle(f"{n} x {n} matrix, centering {centering}: matched pairs under rank-r truncation")
        fig.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(plot_path, lambda tmp: fig.savefig(tmp, format="png", dpi=150))
    finally:
        plt.close(fig)


def run_analysis(*, result_dir: Path, centering: str, device: str, overwrite: bool) -> str:
    inference_dir = result_dir / INFERENCE_DIR
    matrix_path = inference_dir / MATRIX_NAME
    if not matrix_path.is_file():
        raise ValueError(f"matrix file does not exist: {matrix_path}")
    output_dir = result_dir / ANALYSIS_DIR / centering
    factor_paths = {key: output_dir / name for key, name in FACTOR_NAMES.items()}
    metadata_path = output_dir / METADATA_NAME
    plot_path = output_dir / PLOT_NAME
    truncation_plot_path = output_dir / TRUNCATION_PLOT_NAME
    if not overwrite:
        for path in (*factor_paths.values(), metadata_path, plot_path, truncation_plot_path):
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )
    inference_metadata_path = inference_dir / "metadata.json"
    inference_metadata = (
        json.loads(inference_metadata_path.read_text(encoding="utf-8"))
        if inference_metadata_path.is_file()
        else None
    )
    matrix = np.load(matrix_path)
    m, n = matrix.shape
    source = (inference_metadata or {}).get("source") or {}
    if source.get("pairing") == "adjacent":
        if m != n:
            raise ValueError(f"adjacent pairing requires a square matrix, got {m} x {n}")
        pairing = pairing_diagnostics(matrix)
    else:
        pairing = None
    svd, svd_summary = compute_svd(matrix, centering=centering, device=device)
    spectrum, loglog_fit = visualize_svd(
        svd.S, rank_tolerance=svd.rank_tolerance, matrix_shape=(m, n), plot_path=plot_path
    )
    if pairing is not None:
        truncation = truncation_pairing(svd.U, svd.S, svd.Vt)
        plot_truncation_pairing(truncation, n=n, centering=centering, plot_path=truncation_plot_path)
    else:
        truncation = None
    for key, factor in (("U", svd.U), ("S", svd.S), ("Vt", svd.Vt)):
        write_atomic(factor_paths[key], lambda tmp: save_numpy(tmp, factor))
    metadata = {
        "input": {
            "matrix_path": str(matrix_path),
            "matrix_shape": [m, n],
            "matrix_dtype": str(matrix.dtype),
            "inference_metadata": inference_metadata,
        },
        "centering": {"strategy": centering, "formula": CENTERING_FORMULAS[centering]},
        "svd": {"factors": dict(FACTOR_NAMES), **svd_summary},
        "rank": {
            "value": svd.rank,
            "tolerance": svd.rank_tolerance,
            "tolerance_rule": RANK_TOLERANCE_RULE,
        },
        "spectrum": spectrum,
        "loglog_fit": loglog_fit,
    }
    pairing_line = ""
    if pairing is not None:
        metadata["pairing_diagnostics"] = pairing
        metadata["truncation_pairing"] = truncation
        d, o, g = (
            f"{value:.4g}" if value is not None else "n/a"
            for value in (pairing["diagonal_mean"], pairing["off_diagonal_mean"], pairing["mean_gap"])
        )
        pairing_line = (
            f"pairing: diagonal mean {d}, off-diagonal mean {o}, gap {g}; "
            f"top-1 row {pairing['row_top1']:.3f}, column {pairing['column_top1']:.3f}\n"
        )
        t = truncation["threshold"]
        pairing_line += (
            f"truncation: {t['energy_rank']} ranks for 90% energy; "
            f"90% of full-rank top-1 at r = {t['row_top1_rank']} (row), "
            f"{t['column_top1_rank']} (column); full-rank top-1 row "
            f"{truncation['row_top1'][-1]:.3f}, column {truncation['column_top1'][-1]:.3f}\n"
        )
    write_atomic(metadata_path, lambda tmp: save_json(tmp, metadata))
    return (
        f"output_dir: {output_dir}\n"
        f"centering: {centering}\n"
        f"matrix: {m} x {n}\n"
        f"singular values: max {svd_summary['singular_value_max']:.6g}, "
        f"min {svd_summary['singular_value_min']:.6g}; "
        f"reconstruction relative error {svd_summary['reconstruction_relative_error']:.3e}\n"
        f"rank: {svd.rank} (tolerance {svd.rank_tolerance:.6g})\n"
        f"loglog fit: {format_log_log_fit(loglog_fit)}\n"
        f"{pairing_line}"
        f"metadata: {metadata_path}\n"
        f"plot: {plot_path}"
    )
