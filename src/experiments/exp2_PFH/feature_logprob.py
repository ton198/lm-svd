from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
import torch

from utils.common import save_json, save_numpy, write_atomic
from utils.inference import CausalLMInference
from data_pipline.windows import load_split_text, tokenize_stream

from .analysis import MATRIX_NAME


PMI_VALUE = "pmi[h, f] = log p(h + f) - log p(h) - log p(f), no BOS: log p(x) = sum_{t>=2} log p(x_t | x_<t), so the first token of each sequence is unscored; log p(h) cancels, pmi = sum_t log p(f_t | h + f_<t) - sum_{t>=2} log p(f_t | f_<t)"


@dataclass(frozen=True)
class Exp2Inputs:
    histories: list[list[int]]
    futures: list[list[int]]
    source: dict


def wikitext_inputs(
    *, dataset_dir: Path, config: str, split: str, model_path: Path,
    history_count: int = 512, future_count: int = 512, history_tokens: int = 64,
    future_tokens: int = 32, seed: int = 0, pairing: str = "independent",
) -> Exp2Inputs:
    if pairing not in ("independent", "adjacent"):
        raise ValueError(f"unknown pairing: {pairing}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if min(history_count, future_count, history_tokens, future_tokens) <= 0:
        raise ValueError("sample counts and window lengths must be positive")
    if pairing == "adjacent" and history_count != future_count:
        raise ValueError("adjacent pairing requires history_count == future_count")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, local_files_only=True)
    length = history_tokens + future_tokens
    if length > tokenizer.model_max_length:
        raise ValueError(f"history+future length {length} exceeds tokenizer model maximum {tokenizer.model_max_length}")
    tokens = tokenize_stream(load_split_text(dataset_dir, config, split), tokenizer)
    required = length if pairing == "adjacent" else max(history_tokens, future_tokens)
    if len(tokens) < required:
        raise ValueError(f"WikiText has {len(tokens)} tokens, fewer than window {required}")
    generator = torch.Generator().manual_seed(seed)
    history_starts = torch.randint(len(tokens) - (length if pairing == "adjacent" else history_tokens) + 1,
                                  (history_count,), generator=generator).numpy()
    if pairing == "adjacent":
        future_starts = history_starts + history_tokens
    else:
        future_starts = torch.randint(len(tokens) - future_tokens + 1, (future_count,), generator=generator).numpy()
    return Exp2Inputs(
        histories=tokens[history_starts[:, None] + np.arange(history_tokens)].astype(np.int64).tolist(),
        futures=tokens[future_starts[:, None] + np.arange(future_tokens)].astype(np.int64).tolist(),
        source={
            "kind": "wikitext", "dataset_dir": str(dataset_dir), "config": config, "split": split,
            "tokenizer_path": str(model_path), "stream_tokens": len(tokens),
            "history_tokens": history_tokens, "future_tokens": future_tokens,
            "seed": seed, "pairing": pairing,
            "history_starts": history_starts.tolist(), "future_starts": future_starts.tolist(),
            "sampling": "uniform starts with replacement; independent windows or adjacent continuations",
        },
    )


def build_exp2_pfh(
    *,
    inputs: Exp2Inputs,
    model_path: Path,
    device: str,
    batch_size: int,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, int | str]:
    outputs = [output_dir / MATRIX_NAME, output_dir / "metadata.json"]
    if not overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )

    histories, futures = inputs.histories, inputs.futures
    inference = CausalLMInference(model_path, device)
    matrix = np.empty((len(histories), len(futures)), dtype=np.float64)
    started = time.perf_counter()
    future_log_probs = inference.sequence_log_probs(futures, batch_size).numpy()
    for history_index, history in enumerate(histories):
        matrix[history_index] = (
            inference.continuation_log_probs(history, futures, batch_size).numpy()
            - future_log_probs
        )
        print(
            f"history {history_index + 1}/{len(histories)} done, elapsed {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    metadata = {
        "source": inputs.source,
        "model_path": str(model_path),
        "device": device,
        "history_count": len(histories),
        "future_count": len(futures),
        "batch_size": batch_size,
        "value": PMI_VALUE,
        "rows": "histories",
        "columns": "futures",
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": "float64",
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
