from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch

from experiments.exp1_L_M_table.table import build_matrix, select_header_tokens
from utils import (
    CausalLMInference,
    build_column_table,
    build_row_table,
    load_samples,
    read_source_metadata,
    save_json,
    save_numpy,
    write_atomic,
)


def build_exp3_diff(
    *,
    samples_path: Path,
    model_a_path: Path,
    model_b_path: Path,
    device: str,
    history_count: int | None,
    future_count: int | None,
    top_k: int,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, int | str]:
    outputs = [
        output_dir / "matrix.npy",
        output_dir / "matrix_a.npy",
        output_dir / "matrix_b.npy",
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
    inference_a = CausalLMInference(model_a_path, device)
    header_token_ids, future_only_values = select_header_tokens(inference_a, futures, top_k)
    matrix_a = build_matrix(inference_a, histories, futures, header_token_ids)
    vocab_a = inference_a.vocab_size
    del inference_a
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()

    inference_b = CausalLMInference(model_b_path, device)
    if inference_b.vocab_size != vocab_a:
        raise ValueError(
            f"model vocab sizes differ: {model_a_path} has {vocab_a}, "
            f"{model_b_path} has {inference_b.vocab_size}"
        )
    matrix_b = build_matrix(inference_b, histories, futures, header_token_ids)
    del inference_b
    matrix = matrix_a - matrix_b

    row_table = build_row_table([sample.sample_id for sample in histories])
    column_table = build_column_table(
        [sample.sample_id for sample in futures], header_token_ids, future_only_values
    )
    metadata = {
        "samples_path": str(samples_path),
        "model_a_path": str(model_a_path),
        "model_b_path": str(model_b_path),
        "device": device,
        "history_count": len(histories),
        "future_count": len(futures),
        "top_k": top_k,
        "header_context": "future_only",
        "header_model": "model_a",
        "value": "L_A - L_B; L = logit(token | history + future) - mean(logits over full vocabulary), header tokens selected by model A",
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": "float32",
        "source_metadata": read_source_metadata(samples_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(outputs[0], lambda tmp: save_numpy(tmp, matrix))
    write_atomic(outputs[1], lambda tmp: save_numpy(tmp, matrix_a))
    write_atomic(outputs[2], lambda tmp: save_numpy(tmp, matrix_b))
    write_atomic(outputs[3], lambda tmp: pq.write_table(row_table, tmp))
    write_atomic(outputs[4], lambda tmp: pq.write_table(column_table, tmp))
    write_atomic(outputs[5], lambda tmp: save_json(tmp, metadata))

    return {
        "output_dir": str(output_dir),
        "rows": len(histories),
        "columns": len(futures) * top_k,
        "top_k": top_k,
        "model_a_path": str(model_a_path),
        "model_b_path": str(model_b_path),
    }
