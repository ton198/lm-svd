"""NeuralSVD decomposition of exp1's stored inference matrix."""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.analysis import (
    CENTERING_CHOICES,
    CENTERING_FORMULAS,
    SVD_FORM,
    center_matrix,
    fit_log_log,
    format_log_log_fit,
    plot_spectrum_panels,
)
from utils.common import save_json, save_numpy, write_atomic
from utils.experiment_paths import ANALYSIS_DIR, INFERENCE_DIR
from utils.neural_svd import NestedLoRASVDLoss, cosine_lr, neural_svd_spectrum


MATRIX_NAME = "matrix.npy"
INFERENCE_METADATA_NAME = "metadata.json"
FACTOR_NAMES = {"U": "svd_U.npy", "S": "svd_S.npy", "Vt": "svd_Vt.npy"}
LOSS_NAME = "loss_history.npy"
METADATA_NAME = "metadata.json"
PLOT_NAME = "svd_spectrum.png"
DEFLATE1_FORMULA = (
    "M - S[0] * outer(U[:, 0], Vt[0, :]), where component 0 is learned by "
    "NeuralSVD trained with rank + 1 on M and then dropped"
)
ORTHOGONALITY_DEFINITION = (
    "max |off-diagonal| of the Gram matrix of unit-norm learned singular vectors"
)
CAPTURED_ENERGY_FORMULA = "cumsum(S**2) / frobenius_norm**2"


@dataclass(frozen=True)
class NeuralSVDFit:
    """Float64 matrix factors in learned order and per-step training losses."""

    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    loss_history: np.ndarray


def fit_neural_svd(
    data: torch.Tensor,
    *,
    top_k: int,
    rank: int,
    history_batch: int,
    future_batch: int,
    steps: int,
    lr: float,
    min_lr: float,
    seed: int,
    log_every: int,
) -> NeuralSVDFit:
    """Fit dense row/column lookup tables with Adam on sampled future-major blocks.

    ``data`` is a float32 matrix already on the target device. Returned factors
    use unit-norm singular vectors and retain the learned component order.
    """
    m, n = data.shape
    future_count = n // top_k
    generator = torch.Generator().manual_seed(seed)
    rows = nn.Embedding(m, rank)
    columns = nn.Embedding(n, rank)
    nn.init.normal_(rows.weight, 0, 0.1, generator=generator)
    nn.init.normal_(columns.weight, 0, 0.1, generator=generator)
    rows = rows.to(data.device)
    columns = columns.to(data.device)

    loss_fn = NestedLoRASVDLoss(rank, sequential=True).to(data.device)
    optimizer = torch.optim.Adam([rows.weight, columns.weight], lr=lr)
    offsets = torch.arange(top_k)
    loss_history = []
    started = time.perf_counter()

    for step in range(1, steps + 1):
        current_lr = cosine_lr(
            step,
            base_lr=lr,
            min_lr=min_lr,
            decay_start_step=0,
            decay_steps=steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        row_index = torch.randperm(m, generator=generator)[:history_batch]
        future_index = torch.randperm(future_count, generator=generator)[:future_batch]
        column_index = (future_index[:, None] * top_k + offsets).reshape(-1)
        row_index = row_index.to(data.device)
        column_index = column_index.to(data.device)
        loss = loss_fn(
            rows(row_index), columns(column_index), data[row_index][:, column_index]
        )
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"non-finite loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss_value)
        if step % log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"step {step} loss {loss_value:.6g} lr {current_lr:.6g} elapsed {elapsed:.1f}s",
                flush=True,
            )

    spectrum = neural_svd_spectrum(
        rows.weight.detach().double(), columns.weight.detach().double()
    )
    return NeuralSVDFit(
        U=(spectrum.left / math.sqrt(m)).cpu().numpy(),
        S=(spectrum.singular_values * math.sqrt(m * n)).cpu().numpy(),
        Vt=(spectrum.right / math.sqrt(n)).T.cpu().numpy(),
        loss_history=np.asarray(loss_history, dtype=np.float64),
    )


def run_analysis(
    *,
    exp1_result_dir: Path,
    result_dir: Path,
    centering: str,
    device: str,
    rank: int,
    history_batch: int,
    future_batch: int,
    steps: int,
    lr: float,
    min_lr: float,
    seed: int,
    log_every: int,
    overwrite: bool,
) -> str:
    """Fit NeuralSVD to exp1's matrix and compare with its exact decomposition."""
    if centering not in CENTERING_CHOICES:
        raise ValueError(
            f"unknown centering: {centering}; expected one of {', '.join(CENTERING_CHOICES)}"
        )
    if min_lr >= lr:
        raise ValueError(f"min_lr {min_lr} must be smaller than lr {lr}")

    inference_dir = exp1_result_dir / INFERENCE_DIR
    matrix_path = inference_dir / MATRIX_NAME
    if not matrix_path.is_file():
        raise ValueError(f"matrix file does not exist: {matrix_path}")
    inference_metadata_path = inference_dir / INFERENCE_METADATA_NAME
    if not inference_metadata_path.is_file():
        raise ValueError(f"inference metadata does not exist: {inference_metadata_path}")

    exp1_analysis_dir = exp1_result_dir / ANALYSIS_DIR / centering
    exp1_factor_paths = {
        key: exp1_analysis_dir / name for key, name in FACTOR_NAMES.items()
    }
    for path in exp1_factor_paths.values():
        if not path.is_file():
            raise ValueError(
                f"exp1 analysis output does not exist: {path}; "
                f"run exp1 analysis --centering {centering} first"
            )

    output_dir = result_dir / ANALYSIS_DIR / centering
    factor_paths = {key: output_dir / name for key, name in FACTOR_NAMES.items()}
    loss_path = output_dir / LOSS_NAME
    metadata_path = output_dir / METADATA_NAME
    plot_path = output_dir / PLOT_NAME
    if not overwrite:
        for path in (*factor_paths.values(), loss_path, plot_path, metadata_path):
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )

    inference_metadata = json.loads(
        inference_metadata_path.read_text(encoding="utf-8")
    )
    matrix = np.load(matrix_path)
    m, n = matrix.shape
    top_k = inference_metadata["top_k"]
    future_count = inference_metadata["future_count"]
    if n != future_count * top_k:
        raise ValueError(
            f"matrix has {n} columns, expected future_count * top_k = "
            f"{future_count * top_k}"
        )
    trained_rank = rank + (centering == "deflate1")
    if trained_rank > min(m, n):
        raise ValueError(
            f"rank {trained_rank} exceeds min(matrix.shape) {min(m, n)}"
        )
    if history_batch > m:
        raise ValueError(f"history_batch {history_batch} exceeds history count {m}")
    if future_batch > future_count:
        raise ValueError(
            f"future_batch {future_batch} exceeds future count {future_count}"
        )

    try:
        device = torch.device(device)
    except RuntimeError as error:
        raise ValueError(f"invalid device: {device}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use device='cpu'")
    S1 = np.load(exp1_factor_paths["S"]).astype(np.float64)
    U1 = np.asarray(
        np.load(exp1_factor_paths["U"], mmap_mode="r")[:, :rank],
        dtype=np.float64,
    )
    Vt1 = np.asarray(
        np.load(exp1_factor_paths["Vt"], mmap_mode="r")[:rank],
        dtype=np.float64,
    )
    if U1.shape[0] != m or Vt1.shape[1] != n or S1.size < rank:
        raise ValueError("exp1 analysis factors do not match the inference matrix")

    # Deflation must use the learned leading component, never an exact SVD.
    data = center_matrix(
        matrix, "none" if centering == "deflate1" else centering, device=str(device)
    )
    fit = fit_neural_svd(
        torch.as_tensor(data, dtype=torch.float32, device=device),
        top_k=top_k,
        rank=trained_rank,
        history_batch=history_batch,
        future_batch=future_batch,
        steps=steps,
        lr=lr,
        min_lr=min_lr,
        seed=seed,
        log_every=log_every,
    )
    if centering == "deflate1":
        data = data - fit.S[0] * np.outer(fit.U[:, 0], fit.Vt[0])
        U, S, Vt = fit.U[:, 1:], fit.S[1:], fit.Vt[1:]
    else:
        U, S, Vt = fit.U, fit.S, fit.Vt
    loss_history = fit.loss_history

    frobenius_norm = float(np.linalg.norm(data))
    reconstruction_relative_error = (
        float(np.linalg.norm(data - (U * S) @ Vt) / frobenius_norm)
        if frobenius_norm != 0
        else 0.0
    )
    captured_energy_fraction = np.cumsum(S**2) / frobenius_norm**2
    left_gram = U.T @ U
    right_gram = Vt @ Vt.T
    np.fill_diagonal(left_gram, 0)
    np.fill_diagonal(right_gram, 0)
    left_orthogonality_error = float(np.max(np.abs(left_gram)))
    right_orthogonality_error = float(np.max(np.abs(right_gram)))
    loglog_fit = fit_log_log(S)
    singular_value_relative_error = np.abs(S - S1[:rank]) / S1[:rank]
    left_abs_cosine = np.abs(np.sum(U * U1, axis=0))
    right_abs_cosine = np.abs(np.sum(Vt * Vt1, axis=1))
    left_subspace_overlap = float(np.linalg.norm(U1.T @ U)**2 / rank)
    right_subspace_overlap = float(np.linalg.norm(Vt1 @ Vt.T)**2 / rank)
    exp1_captured_energy_fraction = np.cumsum(S1[:rank]**2) / np.sum(S1**2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(24, 4.5))
    try:
        idx = np.arange(1, rank + 1)
        plot_spectrum_panels(
            axes[0], axes[1], S, captured_energy_fraction, loglog_fit,
            index_label="index", energy_title="Cumulative captured energy",
        )
        axes[0].loglog(
            idx, S1[:rank], color="tab:green", linestyle="--", label="exp1 exact SVD"
        )
        axes[0].legend()
        axes[1].lines[0].set_label("NeuralSVD")
        axes[1].plot(
            idx, exp1_captured_energy_fraction,
            color="tab:green", linestyle="--", label="exp1 exact SVD",
        )
        axes[1].legend()
        axes[2].plot(idx, left_abs_cosine, label="left |cos|")
        axes[2].plot(idx, right_abs_cosine, label="right |cos|")
        axes[2].set_ylim(0, 1.01)
        axes[2].set(
            title="Alignment with exp1 singular vectors",
            xlabel="index",
            ylabel="|cosine|",
        )
        axes[2].legend()
        axes[3].plot(np.arange(1, steps + 1), loss_history)
        axes[3].set(title="Training loss", xlabel="step", ylabel="loss")
        fig.suptitle(
            f"{m} x {n} matrix, NeuralSVD rank {rank}, centering {centering}"
        )
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, factor in (("U", U), ("S", S), ("Vt", Vt)):
            write_atomic(factor_paths[key], lambda tmp: save_numpy(tmp, factor))
        write_atomic(loss_path, lambda tmp: save_numpy(tmp, loss_history))
        write_atomic(plot_path, lambda tmp: fig.savefig(tmp, format="png", dpi=150))
    finally:
        plt.close(fig)

    metadata = {
        "input": {
            "matrix_path": str(matrix_path),
            "matrix_shape": [m, n],
            "matrix_dtype": str(matrix.dtype),
            "inference_metadata": inference_metadata,
        },
        "centering": {
            "strategy": centering,
            "formula": (
                DEFLATE1_FORMULA if centering == "deflate1"
                else CENTERING_FORMULAS[centering]
            ),
        },
        "neural_svd": {
            "rank": rank,
            "trained_rank": trained_rank,
            "nesting": "sequential",
            "optimizer": "Adam",
            "init": "normal(0, 0.1) row and column lookup tables",
            "history_batch": history_batch,
            "future_batch": future_batch,
            "top_k": top_k,
            "sampling": (
                "per step: history_batch histories and future_batch futures uniformly "
                "without replacement; all top_k columns of each sampled future"
            ),
            "steps": steps,
            "lr_schedule": {
                "name": "cosine",
                "base_lr": lr,
                "min_lr": min_lr,
                "decay_start_step": 0,
                "decay_steps": steps,
            },
            "seed": seed,
            "device": str(device),
            "final_loss": float(loss_history[-1]),
            "loss_history": LOSS_NAME,
        },
        "svd": {
            "factors": dict(FACTOR_NAMES),
            "form": SVD_FORM,
            "order": "learned",
            "compute_dtype": "float64",
            "singular_value_count": rank,
            "singular_value_max": float(np.max(S)),
            "singular_value_min": float(np.min(S)),
            "frobenius_norm": frobenius_norm,
            "reconstruction_relative_error": reconstruction_relative_error,
            "left_orthogonality_error": left_orthogonality_error,
            "right_orthogonality_error": right_orthogonality_error,
            "orthogonality_definition": ORTHOGONALITY_DEFINITION,
        },
        "spectrum": {
            "plot": PLOT_NAME,
            "index_base": 1,
            "singular_values": S.tolist(),
            "captured_energy_formula": CAPTURED_ENERGY_FORMULA,
            "captured_energy_fraction": captured_energy_fraction.tolist(),
        },
        "loglog_fit": loglog_fit,
        "comparison": {
            "exp1_analysis_dir": str(exp1_analysis_dir),
            "component_count": rank,
            "pairing": "index-matched: learned order vs exp1 descending",
            "exp1_singular_values": S1[:rank].tolist(),
            "singular_value_relative_error": singular_value_relative_error.tolist(),
            "left_abs_cosine": left_abs_cosine.tolist(),
            "right_abs_cosine": right_abs_cosine.tolist(),
            "left_subspace_overlap": left_subspace_overlap,
            "right_subspace_overlap": right_subspace_overlap,
            "exp1_captured_energy_fraction": exp1_captured_energy_fraction.tolist(),
            "formulas": {
                "singular_value_relative_error": "|S - S_exp1| / S_exp1",
                "abs_cosine": "|<u_i, u_exp1_i>|",
                "subspace_overlap": "||U_exp1^T U||_F^2 / rank",
            },
        },
    }
    write_atomic(metadata_path, lambda tmp: save_json(tmp, metadata))
    return (
        f"output_dir: {output_dir}\n"
        f"centering: {centering}\n"
        f"matrix: {m} x {n}\n"
        f"singular values max/min: {metadata['svd']['singular_value_max']:.6g} / "
        f"{metadata['svd']['singular_value_min']:.6g}; "
        f"reconstruction relative error {reconstruction_relative_error:.3e}\n"
        f"left/right orthogonality error: {left_orthogonality_error:.3e} / "
        f"{right_orthogonality_error:.3e}\n"
        f"loglog fit: {format_log_log_fit(loglog_fit)}\n"
        f"exp1 comparison: median singular value relative error "
        f"{np.median(singular_value_relative_error):.6g}, "
        f"median left/right |cos| {np.median(left_abs_cosine):.6g} / "
        f"{np.median(right_abs_cosine):.6g}, "
        f"subspace overlap left/right {left_subspace_overlap:.6g} / "
        f"{right_subspace_overlap:.6g}\n"
        f"metadata: {metadata_path}\n"
        f"plot: {plot_path}"
    )
