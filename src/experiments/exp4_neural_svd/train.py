from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from utils.common import Sample, load_samples, save_json, save_numpy, write_atomic
from utils.inference import CausalLMInference
from utils.lm_table import build_row_table, read_source_metadata
from utils.neural_svd import NestedLoRASVDLoss

from .spectrum import learned_feature_moments


def centered_unembedding(inference: CausalLMInference) -> torch.Tensor:
    head = inference.model.get_output_embeddings()
    if getattr(head, "bias", None) is not None:
        raise ValueError("exp4 requires an output layer without bias")
    embeddings = head.weight.detach().float()
    return embeddings - embeddings.mean(dim=0, keepdim=True)


def centered_logit_block(
    inference: CausalLMInference,
    histories: list[Sample],
    futures: list[Sample],
) -> torch.Tensor:
    sequences = [
        history.token_ids + future.token_ids
        for history in histories
        for future in futures
    ]
    logits = inference.next_token_logits_batch(sequences)
    centered = logits - logits.mean(dim=-1, keepdim=True)
    return centered.view(len(histories), len(futures) * inference.vocab_size)


class Exp4Features(nn.Module):
    def __init__(
        self,
        history_count: int,
        future_count: int,
        rank: int,
        hidden_size: int,
        token_rms: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.history = nn.Embedding(history_count, rank, sparse=True)
        self.future = nn.Embedding(future_count, rank * hidden_size, sparse=True)
        self.rank = rank
        self.hidden_size = hidden_size
        nn.init.normal_(self.history.weight, mean=0, std=0.1, generator=generator)
        nn.init.normal_(
            self.future.weight,
            mean=0,
            std=0.1 / (math.sqrt(hidden_size) * token_rms),
            generator=generator,
        )

    def row_features(self, history_index: torch.Tensor) -> torch.Tensor:
        return self.history(history_index)

    def column_features(
        self, future_index: torch.Tensor, token_matrix: torch.Tensor
    ) -> torch.Tensor:
        weights = self.future(future_index).view(-1, self.rank, self.hidden_size)
        return torch.einsum("fld,vd->fvl", weights, token_matrix).reshape(
            len(future_index) * token_matrix.shape[0], self.rank
        )


def train_exp4(
    *,
    samples_path: Path,
    model_path: Path,
    device: str,
    history_count: int | None,
    future_count: int | None,
    rank: int,
    nesting: str,
    history_batch: int,
    future_batch: int,
    max_steps: int | None,
    eval_every: int,
    tol: float,
    patience: int,
    checkpoint_every: int,
    lr: float,
    seed: int,
    log_every: int,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, int | float | str]:
    outputs = [
        output_dir / "features.pt",
        output_dir / "token_gram.npy",
        output_dir / "loss_history.npy",
        output_dir / "histories.parquet",
        output_dir / "futures.parquet",
        output_dir / "metadata.json",
    ]
    checkpoint_path = output_dir / "checkpoint.pt"

    histories, futures = load_samples(samples_path, history_count, future_count)
    if history_batch > len(histories):
        raise ValueError(
            f"history_batch {history_batch} exceeds history count {len(histories)}"
        )
    if future_batch > len(futures):
        raise ValueError(
            f"future_batch {future_batch} exceeds future count {len(futures)}"
        )
    if overwrite:
        for path in [*outputs, checkpoint_path]:
            path.unlink(missing_ok=True)
    else:
        for path in outputs:
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )
    inference = CausalLMInference(model_path, device)
    token_matrix = centered_unembedding(inference)
    hidden_size = token_matrix.shape[1]
    token_rms = token_matrix.pow(2).mean().sqrt().item()
    generator = torch.Generator().manual_seed(seed)
    features = Exp4Features(
        len(histories), len(futures), rank, hidden_size, token_rms, generator
    ).to(device)
    loss_fn = NestedLoRASVDLoss(rank, sequential=(nesting == "sequential")).to(device)
    optimizer = torch.optim.SparseAdam(features.parameters(), lr=lr)
    token_matrix_double = token_matrix.double()
    token_gram_t = token_matrix_double.T @ token_matrix_double
    del token_matrix_double
    columns = len(futures) * inference.vocab_size
    config = {
        "samples_path": str(samples_path),
        "model_path": str(model_path),
        "history_count": len(histories),
        "future_count": len(futures),
        "rank": rank,
        "nesting": nesting,
        "history_batch": history_batch,
        "future_batch": future_batch,
        "lr": lr,
        "seed": seed,
        "eval_every": eval_every,
        "tol": tol,
        "patience": patience,
    }
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
        diffs = ", ".join(
            f"{key}: {saved_config.get(key)!r} != {current!r}"
            for key, current in config.items()
            if saved_config.get(key) != current
        )
        if saved_config != config:
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
        del checkpoint
        print(f"resuming from step {step}", flush=True)

    def save_checkpoint() -> None:
        payload = {
            "config": config,
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

    started = time.perf_counter()
    while not converged and (max_steps is None or step < max_steps):
        step += 1
        history_indices = torch.randperm(len(histories), generator=generator)[:history_batch]
        future_indices = torch.randperm(len(futures), generator=generator)[:future_batch]
        block = centered_logit_block(
            inference,
            [histories[index] for index in history_indices.tolist()],
            [futures[index] for index in future_indices.tolist()],
        )
        loss = loss_fn(
            features.row_features(history_indices.to(device)),
            features.column_features(future_indices.to(device), token_matrix),
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
                f"step {step} loss {loss_value:.6g} elapsed {elapsed:.1f}s",
                flush=True,
            )

        if step % eval_every == 0:
            f_rms, g_rms, _ = learned_feature_moments(
                features.history.weight.detach(),
                features.future.weight.detach(),
                token_gram_t,
                rank=rank,
                hidden_size=hidden_size,
                columns=columns,
            )
            singular_values = (
                f_rms * g_rms * math.sqrt(len(histories) * columns)
            ).cpu().numpy()
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
        if step % checkpoint_every == 0 or converged or step == max_steps:
            save_checkpoint()

    pairs_evaluated = step * history_batch * future_batch
    if not converged:
        return {
            "output_dir": str(output_dir),
            "rows": len(histories),
            "columns": columns,
            "steps": step,
            "pairs_evaluated": pairs_evaluated,
            "final_loss": loss_history[-1],
            "stop_reason": "max_steps",
        }

    row_table = build_row_table([sample.sample_id for sample in histories])
    future_table = pa.table(
        {
            "future_index": pa.array(range(len(futures)), type=pa.int32()),
            "future_sample_id": pa.array(
                [sample.sample_id for sample in futures], type=pa.int32()
            ),
        }
    )
    metadata = {
        "samples_path": str(samples_path),
        "model_path": str(model_path),
        "device": device,
        "history_count": len(histories),
        "future_count": len(futures),
        "vocab_size": inference.vocab_size,
        "hidden_size": hidden_size,
        "matrix_shape": [len(histories), columns],
        "value": "logit(token | history + future) - mean(logits over full vocabulary)",
        "column_order": "future-major, then token id",
        "feature_parameterization": (
            "f = history lookup (H, rank); g(f, z) = W_f @ (e_z - mean_e), "
            "W_f in R^{rank x hidden}"
        ),
        "rank": rank,
        "nesting": nesting,
        "history_batch": history_batch,
        "future_batch": future_batch,
        "steps": step,
        "lr": lr,
        "seed": seed,
        "optimizer": "SparseAdam",
        "pairs_evaluated": pairs_evaluated,
        "entry_mean_square": entry_mean_square_sum / step,
        "final_loss": float(loss_history[-1]),
        "eval_every": eval_every,
        "tol": tol,
        "patience": patience,
        "converged": True,
        "final_relative_change": last_relative_change,
        "source_metadata": read_source_metadata(samples_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(outputs[0], lambda tmp: torch.save(features.state_dict(), tmp))
    write_atomic(outputs[1], lambda tmp: save_numpy(tmp, token_gram_t.cpu().numpy()))
    write_atomic(
        outputs[2], lambda tmp: save_numpy(tmp, np.asarray(loss_history, dtype=np.float64))
    )
    write_atomic(outputs[3], lambda tmp: pq.write_table(row_table, tmp))
    write_atomic(outputs[4], lambda tmp: pq.write_table(future_table, tmp))
    write_atomic(outputs[5], lambda tmp: save_json(tmp, metadata))
    checkpoint_path.unlink(missing_ok=True)
    return {
        "output_dir": str(output_dir),
        "rows": len(histories),
        "columns": columns,
        "steps": step,
        "pairs_evaluated": pairs_evaluated,
        "final_loss": float(loss_history[-1]),
        "stop_reason": "converged",
    }
