import json
import math
from pathlib import Path

import numpy as np
import torch

from utils.analysis import ANALYSIS_DIR
from utils.common import save_json, save_numpy, write_atomic


def learned_feature_moments(
    history_weight: torch.Tensor,
    future_weight: torch.Tensor,
    token_gram: torch.Tensor,
    *,
    rank: int,
    hidden_size: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return float64 f_rms, g_rms, and right_gram on token_gram.device."""
    with torch.no_grad():
        device = token_gram.device
        history_features_t = history_weight.to(device=device, dtype=torch.float64)
        future_count = future_weight.shape[0]
        future_weights = future_weight.view(future_count, rank, hidden_size)
        f_rms_t = history_features_t.pow(2).mean(dim=0).sqrt()
        g_sq = torch.zeros(rank, dtype=torch.float64, device=device)
        right_gram_t = torch.zeros((rank, rank), dtype=torch.float64, device=device)
        for start in range(0, future_count, 1024):
            weights = future_weights[start : start + 1024].to(
                device=device, dtype=torch.float64
            )
            weighted_gram = (
                weights.reshape(-1, hidden_size) @ token_gram
            ).view_as(weights)
            g_sq += (weighted_gram * weights).sum(dim=(0, 2))
            right_gram_t += torch.einsum("cld,cmd->lm", weighted_gram, weights)
        g_rms = (g_sq / columns).sqrt()
        right_gram = right_gram_t / columns
        return f_rms_t, g_rms, right_gram


def analyze_exp4(
    *, result_dir: Path, device: str | torch.device, overwrite: bool
) -> str:
    """Analyze the learned full-vocabulary factors without materializing g."""
    try:
        device = torch.device(device)
    except RuntimeError as error:
        raise ValueError(f"invalid device: {device}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use device='cpu'")
    train_dir = result_dir / "train"
    input_paths = [
        train_dir / name
        for name in ("features.pt", "token_gram.npy", "loss_history.npy", "metadata.json")
    ]
    for path in input_paths:
        if not path.is_file():
            raise ValueError(f"training input does not exist: {path}")

    output_dir = result_dir / ANALYSIS_DIR
    output_paths = [
        output_dir / name
        for name in (
            "singular_values.npy",
            "left_singular_vectors.npy",
            "spectrum.json",
            "neural_svd_spectrum.png",
        )
    ]
    if not overwrite:
        for path in output_paths:
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )

    metadata = json.loads(input_paths[3].read_text(encoding="utf-8"))
    history_count = metadata["history_count"]
    future_count = metadata["future_count"]
    vocab_size = metadata["vocab_size"]
    hidden_size = metadata["hidden_size"]
    rank = metadata["rank"]
    columns = future_count * vocab_size
    state = torch.load(input_paths[0], map_location="cpu", weights_only=True)
    token_gram = torch.from_numpy(np.load(input_paths[1])).to(
        device=device, dtype=torch.float64
    )
    f_rms_t, g_rms_t, right_gram_t = learned_feature_moments(
        state["history.weight"],
        state["future.weight"],
        token_gram,
        rank=rank,
        hidden_size=hidden_size,
        columns=columns,
    )
    history_features = state["history.weight"].to(dtype=torch.float64).cpu().numpy()
    f_rms = f_rms_t.cpu().numpy()
    g_rms = g_rms_t.cpu().numpy()
    right_gram = right_gram_t.cpu().numpy()
    loss_history = np.asarray(np.load(input_paths[2]), dtype=np.float64)
    if np.any(f_rms == 0) or np.any(g_rms == 0):
        raise ValueError("every feature column must have nonzero norm")

    # Keep learned order, matching neural_svd_spectrum rather than sorting.
    singular_values = f_rms * g_rms * math.sqrt(history_count * columns)
    left_singular_vectors = history_features / (f_rms * math.sqrt(history_count))
    left_gram = left_singular_vectors.T @ left_singular_vectors
    normalized_right_gram = right_gram / np.outer(g_rms, g_rms)
    np.fill_diagonal(left_gram, 0)
    np.fill_diagonal(normalized_right_gram, 0)
    left_orthogonality_error = float(np.max(np.abs(left_gram)))
    right_orthogonality_error = float(np.max(np.abs(normalized_right_gram)))
    frobenius_norm_sq_estimate = (
        metadata["entry_mean_square"] * history_count * columns
    )
    captured_energy_fraction = np.cumsum(singular_values**2) / frobenius_norm_sq_estimate
    spectrum = {
        "singular_values": singular_values.tolist(),
        "left_orthogonality_error": left_orthogonality_error,
        "right_orthogonality_error": right_orthogonality_error,
        "frobenius_norm_sq_estimate": frobenius_norm_sq_estimate,
        "captured_energy_fraction": captured_energy_fraction.tolist(),
        "matrix_shape": [history_count, columns],
        "rank": rank,
        "train_metadata": metadata,
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = np.arange(1, rank + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    try:
        axes[0].semilogy(indices, singular_values)
        axes[0].set(
            title="Singular values", xlabel="learned index", ylabel="singular value"
        )
        axes[1].plot(indices, captured_energy_fraction)
        axes[1].set(
            title="Cumulative captured energy",
            xlabel="learned index",
            ylabel="fraction of squared Frobenius norm",
        )
        axes[2].plot(np.arange(1, loss_history.size + 1), loss_history)
        axes[2].set(title="Training loss", xlabel="step", ylabel="loss")
        fig.suptitle(f"{history_count} x {columns} matrix, NeuralSVD rank {rank}")
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(output_paths[0], lambda tmp: save_numpy(tmp, singular_values))
        write_atomic(output_paths[1], lambda tmp: save_numpy(tmp, left_singular_vectors))
        write_atomic(output_paths[2], lambda tmp: save_json(tmp, spectrum))
        write_atomic(output_paths[3], lambda tmp: fig.savefig(tmp, format="png", dpi=150))
    finally:
        plt.close(fig)

    return "\n".join(
        (
            f"output_dir: {output_dir}",
            f"matrix: {history_count} x {columns}",
            f"training: step {metadata['steps']}, converged {metadata['converged']}",
            f"singular values max/min: {singular_values.max():.6g} / {singular_values.min():.6g}",
            f"left_orthogonality_error: {left_orthogonality_error:.6g}",
            f"right_orthogonality_error: {right_orthogonality_error:.6g}",
            f"captured energy at {rank}: {captured_energy_fraction[-1]:.6g}",
            f"plot: {output_paths[3]}",
        )
    )
