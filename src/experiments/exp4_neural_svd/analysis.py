import json
from pathlib import Path

import numpy as np
import torch

from utils.analysis import fit_log_log, format_log_log_fit, plot_spectrum_panels
from utils.common import save_json, save_numpy, write_atomic
from utils.experiment_paths import ANALYSIS_DIR, TRAIN_DIR

from .spectrum import learned_spectrum


TRAIN_FILES = {
    "features": "features.pt",
    "token_gram": "token_gram.npy",
    "loss_history": "loss_history.npy",
    "metadata": "metadata.json",
}
FACTOR_NAMES = {"S": "singular_values.npy", "U": "left_singular_vectors.npy"}
METADATA_NAME = "metadata.json"
PLOT_NAME = "neural_svd_spectrum.png"
FROBENIUS_NORM_SQ_FORMULA = "entry_mean_square * history_count * columns"
CAPTURED_ENERGY_FORMULA = "cumsum(S**2) / frobenius_norm_sq_estimate"
ORTHOGONALITY_DEFINITION = (
    "max |off-diagonal| of the Gram matrix of unit-normalized learned features"
)


def run_analysis(*, result_dir: Path, device: str, overwrite: bool) -> str:
    """Analyze the learned full-vocabulary factors without materializing g."""
    train_dir = result_dir / TRAIN_DIR
    train_paths = {key: train_dir / name for key, name in TRAIN_FILES.items()}
    for path in train_paths.values():
        if not path.is_file():
            raise ValueError(f"training input does not exist: {path}")
    output_dir = result_dir / ANALYSIS_DIR
    factor_paths = {key: output_dir / name for key, name in FACTOR_NAMES.items()}
    metadata_path = output_dir / METADATA_NAME
    plot_path = output_dir / PLOT_NAME
    if not overwrite:
        for path in (*factor_paths.values(), metadata_path, plot_path):
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )

    train_metadata = json.loads(train_paths["metadata"].read_text(encoding="utf-8"))
    history_count = train_metadata["history_count"]
    columns = train_metadata["future_count"] * train_metadata["vocab_size"]
    rank = train_metadata["rank"]
    state = torch.load(train_paths["features"], map_location="cpu", weights_only=True)
    spectrum = learned_spectrum(
        state,
        np.load(train_paths["token_gram"]),
        history_count=history_count,
        columns=columns,
        rank=rank,
        hidden_size=train_metadata["hidden_size"],
        device=device,
    )
    loss_history = np.asarray(np.load(train_paths["loss_history"]), dtype=np.float64)
    singular_values = spectrum.singular_values
    frobenius_norm_sq_estimate = (
        train_metadata["entry_mean_square"] * history_count * columns
    )
    captured_energy_fraction = np.cumsum(singular_values**2) / frobenius_norm_sq_estimate
    loglog_fit = fit_log_log(singular_values)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    try:
        plot_spectrum_panels(
            axes[0],
            axes[1],
            singular_values,
            captured_energy_fraction,
            loglog_fit,
            index_label="learned index",
            energy_title="Cumulative captured energy",
        )
        axes[2].plot(np.arange(1, loss_history.size + 1), loss_history)
        axes[2].set(title="Training loss", xlabel="step", ylabel="loss")
        fig.suptitle(f"{history_count} x {columns} matrix, NeuralSVD rank {rank}")
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(factor_paths["S"], lambda tmp: save_numpy(tmp, singular_values))
        write_atomic(
            factor_paths["U"], lambda tmp: save_numpy(tmp, spectrum.left_singular_vectors)
        )
        write_atomic(plot_path, lambda tmp: fig.savefig(tmp, format="png", dpi=150))
    finally:
        plt.close(fig)

    metadata = {
        "input": {
            "train_dir": str(train_dir),
            "files": dict(TRAIN_FILES),
            "matrix_shape": [history_count, columns],
            "train_metadata": train_metadata,
        },
        "spectrum": {
            "plot": PLOT_NAME,
            "factors": dict(FACTOR_NAMES),
            "order": "learned",
            "index_base": 1,
            "learned_rank": rank,
            "singular_values": singular_values.tolist(),
            "frobenius_norm_sq_estimate": frobenius_norm_sq_estimate,
            "frobenius_norm_sq_formula": FROBENIUS_NORM_SQ_FORMULA,
            "captured_energy_formula": CAPTURED_ENERGY_FORMULA,
            "captured_energy_fraction": captured_energy_fraction.tolist(),
        },
        "orthogonality": {
            "left_error": spectrum.left_orthogonality_error,
            "right_error": spectrum.right_orthogonality_error,
            "definition": ORTHOGONALITY_DEFINITION,
        },
        "loglog_fit": loglog_fit,
    }
    write_atomic(metadata_path, lambda tmp: save_json(tmp, metadata))
    return "\n".join(
        (
            f"output_dir: {output_dir}",
            f"matrix: {history_count} x {columns}",
            f"training: step {train_metadata['steps']}, converged {train_metadata['converged']}",
            f"singular values max/min: {singular_values.max():.6g} / {singular_values.min():.6g}",
            f"left_orthogonality_error: {spectrum.left_orthogonality_error:.6g}",
            f"right_orthogonality_error: {spectrum.right_orthogonality_error:.6g}",
            f"captured energy at {rank}: {captured_energy_fraction[-1]:.6g}",
            f"loglog fit: {format_log_log_fit(loglog_fit)}",
            f"metadata: {metadata_path}",
            f"plot: {plot_path}",
        )
    )
