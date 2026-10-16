from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from utils import (
    CausalLMInference,
    Sample,
    build_column_table,
    build_row_table,
    load_samples,
    read_source_metadata,
    save_json,
    save_numpy,
    write_atomic,
)


def centered_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits - logits.mean()


def select_header_tokens(
    inference: CausalLMInference, futures: list[Sample], top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= top_k <= inference.vocab_size:
        raise ValueError(f"top_k must be in [1, {inference.vocab_size}]")
    token_ids = np.empty((len(futures), top_k), dtype=np.int32)
    centered_values = np.empty((len(futures), top_k), dtype=np.float32)
    for future_index, future in enumerate(futures):
        logits = inference.next_token_logits(future.token_ids)
        values = centered_logits(logits).numpy()
        order = np.lexsort((np.arange(len(values)), -values))[:top_k]
        token_ids[future_index] = order
        centered_values[future_index] = values[order]
    return token_ids, centered_values


def build_matrix(
    inference: CausalLMInference,
    histories: list[Sample],
    futures: list[Sample],
    header_token_ids: np.ndarray,
) -> np.ndarray:
    top_k = header_token_ids.shape[1]
    matrix = np.empty((len(histories), len(futures) * top_k), dtype=np.float32)
    started = time.perf_counter()
    for history_index, history in enumerate(histories):
        for future_index, future in enumerate(futures):
            logits = inference.next_token_logits(history.token_ids + future.token_ids)
            centered = centered_logits(logits)
            start = future_index * top_k
            matrix[history_index, start : start + top_k] = centered[
                header_token_ids[future_index]
            ].numpy()
        print(
            f"history {history_index + 1}/{len(histories)} done, elapsed {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return matrix


def build_exp1_table(
    *,
    samples_path: Path,
    model_path: Path,
    device: str,
    history_count: int | None,
    future_count: int | None,
    top_k: int,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, int | str]:
    outputs = [
        output_dir / "matrix.npy",
        output_dir / "rows.parquet",
        output_dir / "columns.parquet",
        output_dir / "metadata.json",
    ]
    if not overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )
    histories, futures = load_samples(samples_path, history_count, future_count)
    inference = CausalLMInference(model_path, device)
    header_token_ids, future_only_values = select_header_tokens(inference, futures, top_k)
    matrix = build_matrix(inference, histories, futures, header_token_ids)

    row_table = build_row_table([sample.sample_id for sample in histories])
    column_table = build_column_table(
        [sample.sample_id for sample in futures], header_token_ids, future_only_values
    )
    column_count = len(futures) * top_k
    source_metadata = read_source_metadata(samples_path)
    metadata = {
        "samples_path": str(samples_path),
        "model_path": str(model_path),
        "device": device,
        "history_count": len(histories),
        "future_count": len(futures),
        "top_k": top_k,
        "header_context": "future_only",
        "value": "logit(token | history + future) - mean(logits over full vocabulary)",
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": "float32",
        "source_metadata": source_metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(outputs[0], lambda tmp: save_numpy(tmp, matrix))
    write_atomic(outputs[1], lambda tmp: pq.write_table(row_table, tmp))
    write_atomic(outputs[2], lambda tmp: pq.write_table(column_table, tmp))
    write_atomic(outputs[3], lambda tmp: save_json(tmp, metadata))

    return {
        "output_dir": str(output_dir),
        "rows": len(histories),
        "columns": column_count,
        "top_k": top_k,
        "model_path": str(model_path),
    }
