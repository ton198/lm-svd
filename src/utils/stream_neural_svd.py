from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.common import save_json, save_numpy, write_atomic
from utils.inference import CausalLMInference
from utils.neural_svd import NestedLoRASVDLoss, cosine_lr


class TokenStream:
    def __init__(self, path: Path) -> None:
        self.tokens = np.load(path, mmap_mode="r")
        if self.tokens.ndim != 1 or self.tokens.dtype.kind not in "iu":
            raise ValueError(f"token stream must be a 1-D integer array: {path}")
        json_path = path.with_suffix(".json")
        if not json_path.is_file():
            raise ValueError(f"token stream metadata does not exist: {json_path}")
        self.metadata = json.loads(json_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.tokens)

    def sample_starts(
        self, count: int, window: int, generator: torch.Generator
    ) -> torch.Tensor:
        if len(self) < window:
            raise ValueError(
                f"token stream has {len(self)} tokens, fewer than window {window}"
            )
        return torch.randint(0, len(self) - window + 1, (count,), generator=generator)

    def windows(self, starts: torch.Tensor, window: int) -> torch.Tensor:
        indices = starts.cpu().numpy()[:, None] + np.arange(window)
        return torch.from_numpy(self.tokens[indices].astype(np.int64, copy=False))


def centered_unembedding(inference: CausalLMInference) -> torch.Tensor:
    head = inference.model.get_output_embeddings()
    if getattr(head, "bias", None) is not None:
        raise ValueError("streaming NeuralSVD requires an output layer without bias")
    embeddings = head.weight.detach().float()
    return embeddings - embeddings.mean(dim=0, keepdim=True)


def centered_logit_block(
    inference: CausalLMInference,
    token_matrix: torch.Tensor,
    history_ids: torch.Tensor,
    future_ids: torch.Tensor,
) -> torch.Tensor:
    history_count, history_length = history_ids.shape
    future_count, future_length = future_ids.shape
    # Histories are the outer axis; futures vary fastest within each history.
    pairs = torch.cat(
        (
            history_ids[:, None, :].expand(-1, future_count, -1),
            future_ids[None, :, :].expand(history_count, -1, -1),
        ),
        dim=-1,
    ).reshape(history_count * future_count, history_length + future_length)
    states = inference.final_hidden_states(pairs)[:, -1]
    return (states @ token_matrix.T).view(
        history_count, future_count * token_matrix.shape[0]
    )


def column_features(weights: torch.Tensor, token_matrix: torch.Tensor) -> torch.Tensor:
    # Match block columns: future index first, vocabulary index second.
    return torch.einsum("fld,vd->fvl", weights, token_matrix).reshape(
        weights.shape[0] * token_matrix.shape[0], weights.shape[1]
    )


@dataclass(frozen=True)
class ProbeSpectrum:
    singular_values: np.ndarray  # float64, (rank,), learned order
    left_singular_vectors: np.ndarray  # float64, (histories, rank), unit-norm columns
    left_orthogonality_error: float
    right_orthogonality_error: float


@torch.no_grad()
def learned_probe_spectrum(
    features: torch.nn.Module,
    history_ids: torch.Tensor,
    future_ids: torch.Tensor,
    token_gram: torch.Tensor,
    *,
    vocab_size: int,
    chunk: int = 256,
) -> ProbeSpectrum:
    """Measure learned-order feature norms in operator units on a fixed probe."""
    device = token_gram.device
    history_count = history_ids.shape[0]
    future_count = future_ids.shape[0]
    history_features = torch.cat(
        [
            features.row_features(history_ids[start : start + chunk].to(device)).double()
            for start in range(0, history_count, chunk)
        ],
        dim=0,
    )
    f_rms = history_features.square().mean(dim=0).sqrt()
    rank = history_features.shape[1]
    right_gram = torch.zeros((rank, rank), dtype=torch.float64, device=device)
    for start in range(0, future_count, chunk):
        weights = features.column_weights(
            future_ids[start : start + chunk].to(device)
        ).double()
        weighted_gram = weights @ token_gram
        right_gram += torch.einsum("cld,cmd->lm", weighted_gram, weights)
    right_gram /= future_count * vocab_size
    g_rms = right_gram.diagonal().sqrt()
    if torch.any(f_rms == 0) or torch.any(g_rms == 0):
        raise ValueError("every feature column must have nonzero norm")

    singular_values = (f_rms * g_rms).cpu().numpy()
    left_singular_vectors = (
        history_features / (f_rms * math.sqrt(history_count))
    ).cpu().numpy()
    left_gram = left_singular_vectors.T @ left_singular_vectors
    g_rms_array = g_rms.cpu().numpy()
    normalized_right_gram = right_gram.cpu().numpy() / np.outer(
        g_rms_array, g_rms_array
    )
    np.fill_diagonal(left_gram, 0)
    np.fill_diagonal(normalized_right_gram, 0)
    return ProbeSpectrum(
        singular_values=singular_values,
        left_singular_vectors=left_singular_vectors,
        left_orthogonality_error=float(np.max(np.abs(left_gram))),
        right_orthogonality_error=float(np.max(np.abs(normalized_right_gram))),
    )


@dataclass(frozen=True)
class ProbeReference:
    singular_values: np.ndarray  # float64, (histories,), descending operator units
    left_singular_vectors: np.ndarray  # float64, (histories, histories)
    energy: float


@torch.no_grad()
def probe_reference(
    inference: CausalLMInference,
    token_matrix: torch.Tensor,
    history_ids: torch.Tensor,
    future_ids: torch.Tensor,
    *,
    future_batch: int,
) -> ProbeReference:
    """Compute the exact probe spectrum through its history-by-history Gram."""
    tokens = token_matrix.double()
    token_gram = tokens.T @ tokens
    history_ids = history_ids.to(inference.device)
    future_ids = future_ids.to(inference.device)
    history_count, history_length = history_ids.shape
    future_count, future_length = future_ids.shape
    gram = torch.zeros(
        (history_count, history_count), dtype=torch.float64, device=token_gram.device
    )
    for start in range(0, future_count, future_batch):
        futures = future_ids[start : start + future_batch]
        count = futures.shape[0]
        # Unlike centered_logit_block, futures are outermost here: each future
        # contributes one (histories, hidden) matrix to the history Gram.
        pairs = torch.cat(
            (
                history_ids[None, :, :].expand(count, -1, -1),
                futures[:, None, :].expand(-1, history_count, -1),
            ),
            dim=-1,
        ).reshape(count * history_count, history_length + future_length)
        states = inference.final_hidden_states(pairs)[:, -1].double().reshape(
            count, history_count, token_matrix.shape[1]
        )
        gram += torch.einsum("fid,fkd->ik", states @ token_gram, states)

    normalization = history_count * future_count * token_matrix.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    singular_values = (eigenvalues.flip(0).clamp_min(0) / normalization).sqrt()
    return ProbeReference(
        singular_values=singular_values.cpu().numpy(),
        left_singular_vectors=eigenvectors.flip(1).cpu().numpy(),
        energy=float((gram.trace() / normalization).item()),
    )


def train_stream_neural_svd(
    *,
    features: nn.Module,
    inference: CausalLMInference,
    stream: TokenStream,
    stream_path: Path,
    model_path: Path,
    device: str,
    rank: int,
    nesting: str,
    history_tokens: int,
    future_tokens: int,
    history_batch: int,
    future_batch: int,
    probe_histories: int,
    probe_futures: int,
    probe_seed: int,
    max_steps: int | None,
    eval_every: int,
    tol: float,
    patience: int,
    checkpoint_every: int,
    lr: float,
    lr_schedule: str,
    min_lr: float,
    lr_decay_steps: int,
    seed: int,
    log_every: int,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, int | float | str]:
    outputs = [
        output_dir / "features.pt",
        output_dir / "loss_history.npy",
        output_dir / "probe.npz",
        output_dir / "metadata.json",
    ]
    checkpoint_path = output_dir / "checkpoint.pt"
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"unknown lr_schedule: {lr_schedule}")
    if lr_schedule == "cosine" and min_lr >= lr:
        raise ValueError(f"min_lr {min_lr} must be smaller than lr {lr}")
    if probe_histories < rank:
        raise ValueError(
            f"probe_histories {probe_histories} must be at least rank {rank}"
        )

    probe_generator = torch.Generator().manual_seed(probe_seed)
    history_starts = stream.sample_starts(
        probe_histories, history_tokens, probe_generator
    )
    future_starts = stream.sample_starts(probe_futures, future_tokens, probe_generator)
    probe_history_ids = stream.windows(history_starts, history_tokens)
    probe_future_ids = stream.windows(future_starts, future_tokens)
    max_token = int(stream.tokens.max())
    if max_token >= inference.vocab_size:
        raise ValueError(
            f"token stream contains id {max_token} outside vocabulary {inference.vocab_size}"
        )
    if stream.tokens.dtype.kind == "i":
        min_token = int(stream.tokens.min())
        if min_token < 0:
            raise ValueError(
                f"token stream contains id {min_token} outside vocabulary {inference.vocab_size}"
            )

    if overwrite:
        for path in [*outputs, checkpoint_path]:
            path.unlink(missing_ok=True)
    elif not checkpoint_path.exists():
        for path in outputs:
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )

    # Never put the frozen language model into training mode through the features.
    inference.model.eval()
    token_matrix = centered_unembedding(inference)
    hidden_size = token_matrix.shape[1]
    token_matrix_double = token_matrix.double()
    token_gram = token_matrix_double.T @ token_matrix_double
    del token_matrix_double
    generator = torch.Generator().manual_seed(seed)
    loss_fn = NestedLoRASVDLoss(rank, sequential=(nesting == "sequential")).to(device)
    optimizer = torch.optim.Adam(features.parameters(), lr=lr)
    config = {
        "stream_path": str(stream_path),
        "model_path": str(model_path),
        "stream_tokens": len(stream),
        "rank": rank,
        "nesting": nesting,
        "history_tokens": history_tokens,
        "future_tokens": future_tokens,
        "history_batch": history_batch,
        "future_batch": future_batch,
        "lr": lr,
        "lr_schedule": lr_schedule,
        "min_lr": min_lr,
        "lr_decay_steps": lr_decay_steps,
        "optimizer": "Adam",
        "seed": seed,
        "probe_histories": probe_histories,
        "probe_futures": probe_futures,
        "probe_seed": probe_seed,
        "eval_every": eval_every,
        "tol": tol,
        "patience": patience,
        "features": features.config,
    }
    lr_schedule_state = {"name": lr_schedule, "base_lr": lr}
    if lr_schedule == "cosine":
        lr_schedule_state.update(
            min_lr=min_lr, decay_start_step=0, decay_steps=lr_decay_steps
        )
    step = 0
    loss_history: list[float] = []
    entry_mean_square_sum = 0.0
    previous_singular_values: list[float] | None = None
    stable_evaluations = 0
    last_relative_change: float | None = None
    converged = False
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        saved_config = checkpoint["config"]
        if saved_config != config:
            diffs = ", ".join(
                f"{key}: {saved_config.get(key)!r} != {current!r}"
                for key, current in config.items()
                if saved_config.get(key) != current
            )
            raise ValueError(
                f"checkpoint config mismatch ({diffs}); pass --overwrite to start over"
            )
        features.load_state_dict(checkpoint["features"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["generator"])
        step = checkpoint["step"]
        loss_history = checkpoint["loss_history"].tolist()
        entry_mean_square_sum = checkpoint["entry_mean_square_sum"]
        previous_singular_values = checkpoint["previous_singular_values"]
        stable_evaluations = checkpoint["stable_evaluations"]
        last_relative_change = checkpoint["last_relative_change"]
        converged = checkpoint["converged"]
        print(f"resuming from step {step}", flush=True)
        del checkpoint

    static_metadata = {
        "stream_path": str(stream_path),
        "model_path": str(model_path),
        "device": device,
        "stream_tokens": len(stream),
        "stream_metadata": stream.metadata,
        "vocab_size": inference.vocab_size,
        "hidden_size": hidden_size,
        "history_tokens": history_tokens,
        "future_tokens": future_tokens,
        "value": "logit(token | history + future) - mean(logits over full vocabulary)",
        "operator": (
            "(T v)(h) = E_{f,z}[A(h,(f,z)) v(f,z)]; "
            "h, f uniform-start stream windows, z uniform over vocabulary"
        ),
        "singular_value_units": "operator",
        "features": features.config,
        "parameter_counts": {
            "trainable_total": sum(
                p.numel() for p in features.parameters() if p.requires_grad
            ),
            "modules": {
                name: sum(p.numel() for p in child.parameters() if p.requires_grad)
                for name, child in features.named_children()
            },
        },
        "rank": rank,
        "nesting": nesting,
        "history_batch": history_batch,
        "future_batch": future_batch,
        "lr": lr,
        "seed": seed,
        "optimizer": "Adam",
        "lr_schedule": lr_schedule_state,
        "eval_every": eval_every,
        "tol": tol,
        "patience": patience,
        "probe_histories": probe_histories,
        "probe_futures": probe_futures,
        "probe_seed": probe_seed,
    }

    def save_checkpoint() -> None:
        payload = {
            "config": config,
            "lr_schedule": lr_schedule_state,
            "step": step,
            "features": features.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
            "loss_history": torch.tensor(loss_history, dtype=torch.float64),
            "entry_mean_square_sum": entry_mean_square_sum,
            "previous_singular_values": previous_singular_values,
            "stable_evaluations": stable_evaluations,
            "last_relative_change": last_relative_change,
            "converged": converged,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(checkpoint_path, lambda tmp: torch.save(payload, tmp))

    def save_probe(path: Path) -> None:
        with path.open("wb") as handle:
            np.savez(
                handle,
                history_starts=history_starts.numpy(),
                future_starts=future_starts.numpy(),
            )

    def write_outputs() -> None:
        metadata = {
            **static_metadata,
            "final_lr": current_lr,
            "steps": step,
            "pairs_evaluated": step * history_batch * future_batch,
            "entry_mean_square": entry_mean_square_sum / step,
            "final_loss": float(loss_history[-1]),
            "converged": converged,
            "final_relative_change": last_relative_change,
            "probe_singular_values": previous_singular_values,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(outputs[0], lambda tmp: torch.save(features.state_dict(), tmp))
        write_atomic(
            outputs[1], lambda tmp: save_numpy(tmp, np.asarray(loss_history, dtype=np.float64))
        )
        write_atomic(outputs[2], save_probe)
        write_atomic(outputs[3], lambda tmp: save_json(tmp, metadata))

    schedule_end = lr_decay_steps if lr_schedule == "cosine" else None
    stops_on_convergence = lr_schedule == "constant"
    current_lr = optimizer.param_groups[0]["lr"]
    started = time.perf_counter()
    while (
        not (converged and stops_on_convergence)
        and (max_steps is None or step < max_steps)
        and (schedule_end is None or step < schedule_end)
    ):
        step += 1
        current_lr = (
            lr
            if lr_schedule == "constant"
            else cosine_lr(
                step,
                base_lr=lr,
                min_lr=min_lr,
                decay_start_step=0,
                decay_steps=lr_decay_steps,
            )
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        batch_history_starts = stream.sample_starts(
            history_batch, history_tokens, generator
        )
        batch_future_starts = stream.sample_starts(future_batch, future_tokens, generator)
        history_ids = stream.windows(batch_history_starts, history_tokens).to(device)
        future_ids = stream.windows(batch_future_starts, future_tokens).to(device)
        block = centered_logit_block(inference, token_matrix, history_ids, future_ids)
        loss = loss_fn(
            features.row_features(history_ids),
            column_features(features.column_weights(future_ids), token_matrix),
            block,
        )
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"non-finite loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss_value)
        entry_mean_square_sum += block.double().pow(2).mean().item()
        if step % log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"step {step} loss {loss_value:.6g} lr {current_lr:.6g} elapsed {elapsed:.1f}s",
                flush=True,
            )

        if step % eval_every == 0:
            singular_values = learned_probe_spectrum(
                features,
                probe_history_ids,
                probe_future_ids,
                token_gram,
                vocab_size=inference.vocab_size,
            ).singular_values
            if previous_singular_values is None:
                relative_change = math.inf
            else:
                prev = np.asarray(previous_singular_values)
                if np.any(prev <= 0):
                    relative_change = math.inf
                else:
                    relative_change = float(
                        np.max(np.abs(singular_values - prev) / prev)
                    )
            stable_evaluations = (
                stable_evaluations + 1 if relative_change < tol else 0
            )
            previous_singular_values = singular_values.tolist()
            last_relative_change = relative_change
            converged = stable_evaluations >= patience
            print(
                f"step {step} singular-value relative change {relative_change:.3g} "
                f"stable {stable_evaluations}/{patience}",
                flush=True,
            )
        if not (converged and stops_on_convergence) and (
            step % checkpoint_every == 0
            or step == max_steps
            or step == schedule_end
        ):
            save_checkpoint()
            write_outputs()

    if converged and stops_on_convergence:
        write_outputs()
        checkpoint_path.unlink(missing_ok=True)
        stop_reason = "converged"
    else:
        stop_reason = (
            "lr_schedule_end"
            if schedule_end is not None and step >= schedule_end
            else "max_steps"
        )
    return {
        "output_dir": str(output_dir),
        "steps": step,
        "pairs_evaluated": step * history_batch * future_batch,
        "final_loss": float(loss_history[-1]),
        "stop_reason": stop_reason,
    }


def run_stream_analysis(
    *,
    result_dir: Path,
    build_features: Callable[[dict, CausalLMInference], nn.Module],
    device: str,
    reference_future_batch: int,
    overwrite: bool,
) -> str:
    """Compare learned features with the exact SVD of the saved training probe."""
    from utils.analysis import fit_log_log, format_log_log_fit, plot_spectrum_panels
    from utils.experiment_paths import ANALYSIS_DIR, TRAIN_DIR

    train_files = {
        "features": "features.pt",
        "loss_history": "loss_history.npy",
        "probe": "probe.npz",
        "metadata": "metadata.json",
    }
    train_dir = result_dir / TRAIN_DIR
    train_paths = {key: train_dir / name for key, name in train_files.items()}
    for path in train_paths.values():
        if not path.is_file():
            raise ValueError(f"training input does not exist: {path}")
    output_dir = result_dir / ANALYSIS_DIR
    factor_names = {
        "S": "singular_values.npy",
        "reference_S": "reference_singular_values.npy",
        "U": "left_singular_vectors.npy",
        "reference_U": "reference_left_singular_vectors.npy",
    }
    factor_paths = {key: output_dir / name for key, name in factor_names.items()}
    plot_path = output_dir / "neural_svd_spectrum.png"
    metadata_path = output_dir / "metadata.json"
    if not overwrite:
        for path in (*factor_paths.values(), plot_path, metadata_path):
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )

    train_metadata = json.loads(train_paths["metadata"].read_text(encoding="utf-8"))
    inference = CausalLMInference(Path(train_metadata["model_path"]), device)
    stream = TokenStream(Path(train_metadata["stream_path"]))
    features = build_features(train_metadata["features"], inference)
    features.load_state_dict(
        torch.load(train_paths["features"], map_location="cpu", weights_only=True)
    )
    features.to(device)
    features.eval()
    with np.load(train_paths["probe"]) as probe:
        history_ids = stream.windows(
            torch.from_numpy(probe["history_starts"]),
            train_metadata["history_tokens"],
        )
        future_ids = stream.windows(
            torch.from_numpy(probe["future_starts"]),
            train_metadata["future_tokens"],
        )
    history_count = history_ids.shape[0]
    future_count = future_ids.shape[0]
    rank = train_metadata["rank"]
    token_matrix = centered_unembedding(inference)
    tokens = token_matrix.double()
    token_gram = tokens.T @ tokens
    spectrum = learned_probe_spectrum(
        features,
        history_ids,
        future_ids,
        token_gram,
        vocab_size=inference.vocab_size,
    )
    reference = probe_reference(
        inference,
        token_matrix,
        history_ids,
        future_ids,
        future_batch=reference_future_batch,
    )
    singular_values = spectrum.singular_values
    reference_values = reference.singular_values[:rank]
    left_vectors = spectrum.left_singular_vectors
    reference_vectors = reference.left_singular_vectors[:, :rank]
    singular_value_relative_error = singular_values / reference_values - 1
    left_alignment = np.abs(np.sum(left_vectors * reference_vectors, axis=0))
    left_subspace_overlap = np.empty(rank, dtype=np.float64)
    for k in range(1, rank + 1):
        basis, _ = np.linalg.qr(left_vectors[:, :k])
        left_subspace_overlap[k - 1] = (
            np.linalg.norm(reference_vectors[:, :k].T @ basis, ord="fro") ** 2 / k
        )
    captured_energy_fraction = np.cumsum(singular_values**2) / reference.energy
    reference_captured_energy_fraction = (
        np.cumsum(reference_values**2) / reference.energy
    )
    loglog_fit = fit_log_log(singular_values)
    loss_history = np.asarray(np.load(train_paths["loss_history"]), dtype=np.float64)

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
            index_label="index",
            energy_title="Cumulative captured energy (probe)",
        )
        indices = np.arange(1, rank + 1)
        axes[0].loglog(
            indices, reference_values, linestyle="--", label="probe exact SVD"
        )
        axes[1].lines[0].set_label("learned")
        axes[1].plot(
            indices,
            reference_captured_energy_fraction,
            linestyle="--",
            label="probe exact SVD",
        )
        axes[0].legend()
        axes[1].legend()
        axes[2].plot(np.arange(1, loss_history.size + 1), loss_history)
        axes[2].set(title="Training loss", xlabel="step", ylabel="loss")
        fig.suptitle(
            f"stream NeuralSVD rank {rank}, probe {history_count} x {future_count}"
        )
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(factor_paths["S"], lambda tmp: save_numpy(tmp, singular_values))
        write_atomic(
            factor_paths["reference_S"], lambda tmp: save_numpy(tmp, reference_values)
        )
        write_atomic(factor_paths["U"], lambda tmp: save_numpy(tmp, left_vectors))
        write_atomic(
            factor_paths["reference_U"], lambda tmp: save_numpy(tmp, reference_vectors)
        )
        write_atomic(plot_path, lambda tmp: fig.savefig(tmp, format="png", dpi=150))
    finally:
        plt.close(fig)

    metadata = {
        "input": {
            "train_dir": str(train_dir),
            "files": train_files,
            "train_metadata": train_metadata,
        },
        "probe": {
            "histories": history_count,
            "futures": future_count,
            "seed": train_metadata["probe_seed"],
            "energy": reference.energy,
            "energy_formula": "trace(K) / (Hp * Fp * V), K = sum_f S_f G S_f^T",
        },
        "spectrum": {
            "singular_values": singular_values.tolist(),
            "units": "operator",
            "order": "learned",
            "captured_energy_fraction": captured_energy_fraction.tolist(),
            "captured_energy_formula": "cumsum(singular_values**2) / probe.energy",
        },
        "reference": {
            "singular_values": reference_values.tolist(),
            "captured_energy_fraction": reference_captured_energy_fraction.tolist(),
            "definition": (
                "exact SVD of the probe sub-table scaled by 1/sqrt(Hp * Fp * V)"
            ),
        },
        "comparison": {
            "singular_value_relative_error": singular_value_relative_error.tolist(),
            "singular_value_relative_error_definition": (
                "learned singular value / reference singular value - 1; "
                "learned order versus reference descending order"
            ),
            "left_alignment": left_alignment.tolist(),
            "left_alignment_definition": (
                "abs(dot(learned left vector, reference left vector))"
            ),
            "left_subspace_overlap": left_subspace_overlap.tolist(),
            "left_subspace_overlap_definition": (
                "||U_ref[:, :k]^T Q_k||_F^2 / k, "
                "Q_k = qr(U_learned[:, :k]).Q, k = 1,...,rank"
            ),
        },
        "orthogonality": {
            "left_error": spectrum.left_orthogonality_error,
            "right_error": spectrum.right_orthogonality_error,
            "definition": (
                "max |off-diagonal| of the Gram matrix of unit-normalized "
                "learned features"
            ),
        },
        "loglog_fit": loglog_fit,
    }
    write_atomic(metadata_path, lambda tmp: save_json(tmp, metadata))
    alignment_count = min(rank, 8)
    return "\n".join(
        (
            f"output_dir: {output_dir}",
            f"training: step {train_metadata['steps']}, converged {train_metadata['converged']}",
            f"singular values max/min: {singular_values.max():.6g} / {singular_values.min():.6g}",
            f"reference singular value top-1: {reference_values[0]:.6g}",
            f"left alignment min (first {alignment_count}): {left_alignment[:alignment_count].min():.6g}",
            f"captured energy at {rank}: {captured_energy_fraction[-1]:.6g}",
            f"reference captured energy at {rank}: {reference_captured_energy_fraction[-1]:.6g}",
            f"left_orthogonality_error: {spectrum.left_orthogonality_error:.6g}",
            f"right_orthogonality_error: {spectrum.right_orthogonality_error:.6g}",
            f"loglog fit: {format_log_log_fit(loglog_fit)}",
            f"metadata: {metadata_path}",
            f"plot: {plot_path}",
        )
    )
