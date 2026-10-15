from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from experiments.common import Sample, load_samples, save_json, save_numpy, write_atomic
from utils import CausalLMInference


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

    row_table = pa.table(
        {
            "row_index": pa.array(range(len(histories)), type=pa.int32()),
            "history_sample_id": pa.array(
                [sample.sample_id for sample in histories], type=pa.int32()
            ),
        }
    )
    column_count = len(futures) * top_k
    columns = {
        "column_index": pa.array(range(column_count), type=pa.int32()),
        "future_sample_id": pa.array(
            [sample.sample_id for sample in futures for _ in range(top_k)], type=pa.int32()
        ),
        "rank": pa.array([rank for _ in futures for rank in range(top_k)], type=pa.int32()),
        "token_id": pa.array(header_token_ids.reshape(-1), type=pa.int32()),
        "future_only_centered_logit": pa.array(
            future_only_values.reshape(-1), type=pa.float32()
        ),
    }
    column_table = pa.table(columns)
    source_metadata = pq.read_schema(samples_path).metadata or {}
    raw_source_metadata = source_metadata.get(b"lm_svd")
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
        "source_metadata": json.loads(raw_source_metadata) if raw_source_metadata else None,
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
