from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from experiments.common import load_samples, save_json, save_numpy, write_atomic
from utils import CausalLMInference


def build_exp2_pfh(
    *,
    samples_path: Path,
    model_path: Path,
    device: str,
    history_count: int | None,
    future_count: int | None,
    batch_size: int,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, int | str]:
    outputs = [output_dir / "feature_log_prob.npy", output_dir / "metadata.json"]
    if not overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )

    histories, futures = load_samples(samples_path, history_count, future_count)
    inference = CausalLMInference(model_path, device)
    matrix = np.empty((len(histories), len(futures)), dtype=np.float64)
    started = time.perf_counter()
    continuation_ids = [future.token_ids for future in futures]
    for history_index, history in enumerate(histories):
        matrix[history_index] = inference.continuation_log_probs(
            history.token_ids, continuation_ids, batch_size
        ).numpy()
        print(
            f"history {history_index + 1}/{len(histories)} done, elapsed {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    source_metadata = pq.read_schema(samples_path).metadata or {}
    raw_source_metadata = source_metadata.get(b"lm_svd")
    metadata = {
        "samples_path": str(samples_path),
        "model_path": str(model_path),
        "device": device,
        "history_count": len(histories),
        "future_count": len(futures),
        "batch_size": batch_size,
        "value": "sum_t log p(future_t | history + future_<t>)",
        "rows": "histories",
        "columns": "futures",
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": "float64",
        "history_sample_ids": [sample.sample_id for sample in histories],
        "future_sample_ids": [sample.sample_id for sample in futures],
        "source_metadata": json.loads(raw_source_metadata) if raw_source_metadata else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(outputs[0], lambda tmp: save_numpy(tmp, matrix))
    write_atomic(outputs[1], lambda tmp: save_json(tmp, metadata))
    return {
        "output_dir": str(output_dir),
        "rows": len(histories),
        "columns": len(futures),
        "model_path": str(model_path),
    }
