from pathlib import Path

import numpy as np

from .common import write_atomic
from .linalg import MatrixSVD, matrix_svd


SVD_FORM = "thin: matrix = U @ diag(S) @ Vt"
RANK_TOLERANCE_RULE = "S.max() * max(matrix.shape) * finfo(float64).eps"
SPECTRUM_SELECTION = "singular values strictly greater than rank tolerance, descending"
CUMULATIVE_ENERGY_FORMULA = "cumsum(S_kept**2) / sum(S_kept**2)"
LOGLOG_FIT_FORMULA = "log10(singular_value) = slope * log10(index) + intercept, index starting at 1"
CENTERING_CHOICES = ("none", "global", "row", "deflate1")
CENTERING_FORMULAS = {
    "none": "M",
    "global": "M - mean(M)",
    "row": "M - mean(M, axis=1, keepdims=True)",
    "deflate1": "M - S[0] * outer(U[:, 0], Vt[0, :]), where M = U @ diag(S) @ Vt",
}


def fit_log_log(values: np.ndarray) -> dict[str, float | int | str | None]:
    indices = np.arange(1, values.size + 1)
    if values.size < 2:
        slope = intercept = None
    else:
        slope, intercept = np.polyfit(np.log10(indices), np.log10(values), 1)
        slope, intercept = float(slope), float(intercept)
    return {
        "slope": slope,
        "intercept": intercept,
        "point_count": int(values.size),
        "log_base": 10,
        "formula": LOGLOG_FIT_FORMULA,
    }


def format_log_log_fit(fit: dict) -> str:
    if fit["slope"] is None:
        return f"skipped (fewer than 2 points: {fit['point_count']})"
    return (
        f"k = {fit['slope']:.6g}, b = {fit['intercept']:.6g} "
        f"({fit['point_count']} points, log10)"
    )


def plot_spectrum_panels(
    value_ax,
    energy_ax,
    values: np.ndarray,
    energy_fraction: np.ndarray,
    fit: dict,
    *,
    index_label: str,
    energy_title: str,
    energy_ylim: tuple[float, float] | None = None,
    rank: int | None = None,
) -> None:
    indices = np.arange(1, values.size + 1)
    value_ax.loglog(indices, values, label="singular values")
    if fit["slope"] is not None:
        value_ax.plot(
            indices,
            10 ** fit["intercept"] * indices.astype(np.float64) ** fit["slope"],
            color="tab:orange",
            linestyle=":",
            label=f"log-log fit: k = {fit['slope']:.4g}, b = {fit['intercept']:.4g}",
        )
    if rank is not None:
        value_ax.axvline(rank, color="red", linestyle="--", label=f"rank = {rank}")
    value_ax.set(title="Singular values", xlabel=index_label, ylabel="singular value")
    value_ax.legend()
    energy_ax.plot(indices, energy_fraction)
    if rank is not None:
        energy_ax.axvline(rank, color="red", linestyle="--", label=f"rank = {rank}")
    energy_ax.set(
        title=energy_title,
        xlabel=index_label,
        ylabel="fraction of squared Frobenius norm",
    )
    if energy_ylim is not None:
        energy_ax.set_ylim(*energy_ylim)
    if rank is not None:
        energy_ax.legend()


def center_matrix(matrix: np.ndarray, centering: str, *, device: str) -> np.ndarray:
    if centering not in CENTERING_CHOICES:
        raise ValueError(
            f"unknown centering: {centering}; expected one of {', '.join(CENTERING_CHOICES)}"
        )
    data = matrix.astype(np.float64)
    if centering == "global":
        return data - data.mean()
    if centering == "row":
        return data - data.mean(axis=1, keepdims=True)
    if centering == "deflate1":
        leading = matrix_svd(data, device=device)
        data -= leading.S[0] * np.outer(leading.U[:, 0], leading.Vt[0])
        return data
    return data


def compute_svd(
    matrix: np.ndarray, *, centering: str, device: str
) -> tuple[MatrixSVD, dict[str, int | float | str]]:
    data = center_matrix(matrix, centering, device=device)
    result = matrix_svd(data, device=device)
    frobenius_norm = float(np.linalg.norm(data))
    reconstruction_relative_error = (
        float(np.linalg.norm(data - (result.U * result.S) @ result.Vt) / frobenius_norm)
        if frobenius_norm != 0
        else 0.0
    )
    summary = {
        "form": SVD_FORM,
        "compute_dtype": "float64",
        "device": device,
        "singular_value_count": int(result.S.size),
        "singular_value_max": float(result.S[0]),
        "singular_value_min": float(result.S[-1]),
        "frobenius_norm": frobenius_norm,
        "reconstruction_relative_error": reconstruction_relative_error,
    }
    return result, summary


def visualize_svd(
    singular_values: np.ndarray,
    *,
    rank_tolerance: float,
    matrix_shape: tuple[int, int],
    plot_path: Path,
) -> tuple[dict, dict]:
    kept = singular_values[singular_values > rank_tolerance]
    if kept.size == 0:
        raise ValueError("all singular values are zero; nothing to plot")
    fit = fit_log_log(kept)
    kept_squared = kept**2
    energy = np.cumsum(kept_squared) / kept_squared.sum()
    m, n = matrix_shape

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    try:
        plot_spectrum_panels(
            axes[0], axes[1], kept, energy, fit,
            index_label="index", energy_title="Cumulative energy",
            energy_ylim=(0, 1.01), rank=int(kept.size),
        )
        fig.suptitle(f"{m} x {n} matrix")
        fig.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(plot_path, lambda tmp: fig.savefig(tmp, format="png", dpi=150))
    finally:
        plt.close(fig)
    spectrum = {
        "plot": plot_path.name,
        "selection": SPECTRUM_SELECTION,
        "index_base": 1,
        "singular_values": kept.tolist(),
        "cumulative_energy_formula": CUMULATIVE_ENERGY_FORMULA,
        "cumulative_energy_fraction": energy.tolist(),
    }
    return spectrum, fit
