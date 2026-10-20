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
    steps: int,
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
    if not overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )

    histories, futures = load_samples(samples_path, history_count, future_count)
    if history_batch > len(histories):
        raise ValueError(
            f"history_batch {history_batch} exceeds history count {len(histories)}"
        )
    if future_batch > len(futures):
        raise ValueError(
            f"future_batch {future_batch} exceeds future count {len(futures)}"
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
    loss_history = np.empty(steps, dtype=np.float64)
    entry_mean_square_sum = 0.0
    started = time.perf_counter()
    for step in range(1, steps + 1):
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
        loss_history[step - 1] = loss_value
        entry_mean_square_sum += block.double().pow(2).mean().item()
        if step % log_every == 0 or step == steps:
            elapsed = time.perf_counter() - started
            print(
                f"step {step}/{steps} loss {loss_value:.6g} elapsed {elapsed:.1f}s",
                flush=True,
            )

    token_matrix_double = token_matrix.double()
    token_gram = (token_matrix_double.T @ token_matrix_double).cpu().numpy()
    row_table = build_row_table([sample.sample_id for sample in histories])
    future_table = pa.table(
        {
            "future_index": pa.array(range(len(futures)), type=pa.int32()),
            "future_sample_id": pa.array(
                [sample.sample_id for sample in futures], type=pa.int32()
            ),
        }
    )
    columns = len(futures) * inference.vocab_size
    pairs_evaluated = steps * history_batch * future_batch
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
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "optimizer": "SparseAdam",
        "pairs_evaluated": pairs_evaluated,
        "entry_mean_square": entry_mean_square_sum / steps,
        "final_loss": float(loss_history[-1]),
        "source_metadata": read_source_metadata(samples_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(outputs[0], lambda tmp: torch.save(features.state_dict(), tmp))
    write_atomic(outputs[1], lambda tmp: save_numpy(tmp, token_gram))
    write_atomic(outputs[2], lambda tmp: save_numpy(tmp, loss_history))
    write_atomic(outputs[3], lambda tmp: pq.write_table(row_table, tmp))
    write_atomic(outputs[4], lambda tmp: pq.write_table(future_table, tmp))
    write_atomic(outputs[5], lambda tmp: save_json(tmp, metadata))
    return {
        "output_dir": str(output_dir),
        "rows": len(histories),
        "columns": columns,
        "steps": steps,
        "pairs_evaluated": pairs_evaluated,
        "final_loss": float(loss_history[-1]),
    }
