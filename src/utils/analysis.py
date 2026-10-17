import argparse
import json
from pathlib import Path

import numpy as np

from .common import save_json, save_numpy, write_atomic
from .linalg import matrix_svd, numerical_rank


SVD_NAMES = ("svd_U.npy", "svd_S.npy", "svd_Vt.npy", "svd_metadata.json")
RANK_NAME = "rank.json"
PLOT_NAME = "svd_spectrum.png"
INFERENCE_DIR = "inference"
ANALYSIS_DIR = "analysis"
CENTERING_CHOICES = ("none", "global", "row")
CENTERING_FORMULAS = {
    "none": "M",
    "global": "M - mean(M)",
    "row": "M - mean(M, axis=1, keepdims=True)",
}


def center_matrix(matrix: np.ndarray, centering: str) -> np.ndarray:
    if centering not in CENTERING_CHOICES:
        raise ValueError(
            f"unknown centering: {centering}; expected one of {', '.join(CENTERING_CHOICES)}"
        )
    data = matrix.astype(np.float64)
    if centering == "global":
        return data - data.mean()
    if centering == "row":
        return data - data.mean(axis=1, keepdims=True)
    return data


def compute_svd(
    *, matrix_path: Path, output_dir: Path, centering: str, overwrite: bool
) -> dict[str, int | float | str]:
    if not matrix_path.is_file():
        raise ValueError(f"matrix file does not exist: {matrix_path}")
    if centering not in CENTERING_CHOICES:
        raise ValueError(
            f"unknown centering: {centering}; expected one of {', '.join(CENTERING_CHOICES)}"
        )
    for name in SVD_NAMES:
        path = output_dir / name
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists: {path}; pass --overwrite to replace it"
            )

    matrix = np.load(matrix_path)
    data = center_matrix(matrix, centering)
    result = matrix_svd(data)
    frobenius_norm = float(np.linalg.norm(data))
    reconstruction_relative_error = (
        float(np.linalg.norm(data - (result.U * result.S) @ result.Vt) / frobenius_norm)
        if frobenius_norm != 0
        else 0.0
    )
    m, n = matrix.shape
    matrix_metadata_path = matrix_path.parent / "metadata.json"
    metadata = {
        "input_path": str(matrix_path),
        "matrix_shape": [m, n],
        "input_dtype": str(matrix.dtype),
        "compute_dtype": "float64",
        "centering": centering,
        "centering_formula": CENTERING_FORMULAS[centering],
        "svd": "thin: matrix = U @ diag(S) @ Vt",
        "singular_value_max": float(result.S[0]),
        "singular_value_min": float(result.S[-1]),
        "frobenius_norm": frobenius_norm,
        "reconstruction_relative_error": reconstruction_relative_error,
        "matrix_metadata": (
            json.loads(matrix_metadata_path.read_text(encoding="utf-8"))
            if matrix_metadata_path.exists()
            else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, factor in zip(SVD_NAMES[:3], (result.U, result.S, result.Vt)):
        write_atomic(output_dir / name, lambda tmp: save_numpy(tmp, factor))
    write_atomic(output_dir / SVD_NAMES[3], lambda tmp: save_json(tmp, metadata))
    return {
        "output_dir": str(output_dir),
        "centering": centering,
        "rows": m,
        "columns": n,
        "singular_value_max": float(result.S[0]),
        "singular_value_min": float(result.S[-1]),
        "reconstruction_relative_error": reconstruction_relative_error,
    }


def compute_rank(*, svd_dir: Path, overwrite: bool) -> dict[str, int | float | str]:
    singular_values_path = svd_dir / "svd_S.npy"
    metadata_path = svd_dir / "svd_metadata.json"
    for path in (singular_values_path, metadata_path):
        if not path.is_file():
            raise ValueError(f"SVD input does not exist: {path}")
    output_path = svd_dir / RANK_NAME
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )

    singular_values = np.load(singular_values_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = tuple(metadata["matrix_shape"])
    dtype = np.dtype(metadata["input_dtype"])
    rank, tolerance = numerical_rank(singular_values, shape, dtype)
    rank_metadata = {
        "svd_dir": str(svd_dir),
        "matrix_shape": list(shape),
        "input_dtype": str(dtype),
        "rank": rank,
        "rank_tolerance": tolerance,
        "rank_tolerance_rule": "S.max() * max(matrix.shape) * finfo(input_dtype).eps",
        "singular_value_count": int(singular_values.size),
    }
    svd_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(output_path, lambda tmp: save_json(tmp, rank_metadata))
    return {
        "output_path": str(output_path),
        "rank": rank,
        "rank_tolerance": tolerance,
        "singular_value_count": int(singular_values.size),
    }


def visualize_svd(*, svd_dir: Path, overwrite: bool) -> Path:
    singular_values_path = svd_dir / "svd_S.npy"
    rank_path = svd_dir / RANK_NAME
    for path in (singular_values_path, rank_path):
        if not path.is_file():
            raise ValueError(f"visualization input does not exist: {path}")
    output_path = svd_dir / PLOT_NAME
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )

    singular_values = np.load(singular_values_path)
    squared_values = singular_values**2
    total_energy = np.sum(squared_values)
    if total_energy == 0:
        raise ValueError("all singular values are zero; nothing to plot")
    metadata = json.loads(rank_path.read_text(encoding="utf-8"))
    rank = metadata["rank"]
    tolerance = metadata["rank_tolerance"]
    m, n = metadata["matrix_shape"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = np.arange(1, singular_values.size + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    try:
        axes[0].set_xscale("log")
        axes[0].semilogy(indices, np.where(singular_values > 0, singular_values, np.nan))
        axes[0].axvline(rank, color="red", linestyle="--", label=f"rank = {rank}")
        axes[0].axhline(tolerance, color="gray", linestyle=":", label="tolerance")
        axes[0].set(title="Singular values", xlabel="index", ylabel="singular value")
        axes[0].legend()
        axes[1].plot(indices, np.cumsum(squared_values) / total_energy)
        axes[1].axvline(rank, color="red", linestyle="--", label=f"rank = {rank}")
        axes[1].set(
            title="Cumulative energy",
            xlabel="index",
            ylabel="fraction of squared Frobenius norm",
            ylim=(0, 1.01),
        )
        axes[1].legend()
        fig.suptitle(f"{m} x {n} matrix")
        fig.tight_layout()
        svd_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(output_path, lambda tmp: fig.savefig(tmp, format="png", dpi=150))
    finally:
        plt.close(fig)
    return output_path


def add_analysis_command(subparsers, *, result_dir: Path) -> None:
    parser = subparsers.add_parser(
        "analysis",
        help="Run SVD, rank, and visualization on the inference matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result-dir", type=Path, default=result_dir,
        help="Experiment result directory; reads <result-dir>/inference, writes <result-dir>/analysis/<centering>",
    )
    parser.add_argument(
        "--centering", choices=CENTERING_CHOICES, default="none",
        help="Secondary centering applied before SVD",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")


def run_analysis(
    *, result_dir: Path, matrix_name: str, centering: str, overwrite: bool
) -> str:
    if centering not in CENTERING_CHOICES:
        raise ValueError(
            f"unknown centering: {centering}; expected one of {', '.join(CENTERING_CHOICES)}"
        )
    matrix_path = result_dir / INFERENCE_DIR / matrix_name
    output_dir = result_dir / ANALYSIS_DIR / centering
    if not overwrite:
        for name in (*SVD_NAMES, RANK_NAME, PLOT_NAME):
            path = output_dir / name
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )
    svd = compute_svd(
        matrix_path=matrix_path,
        output_dir=output_dir,
        centering=centering,
        overwrite=overwrite,
    )
    rank = compute_rank(svd_dir=output_dir, overwrite=overwrite)
    plot_path = visualize_svd(svd_dir=output_dir, overwrite=overwrite)
    return (
        f"output_dir: {output_dir}\n"
        f"centering: {centering}\n"
        f"matrix: {svd['rows']} x {svd['columns']}\n"
        f"singular values: max {svd['singular_value_max']:.6g}, "
        f"min {svd['singular_value_min']:.6g}; "
        f"reconstruction relative error {svd['reconstruction_relative_error']:.3e}\n"
        f"rank: {rank['rank']} (tolerance {rank['rank_tolerance']:.6g})\n"
        f"plot: {plot_path}"
    )
