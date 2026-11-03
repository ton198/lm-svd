from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from utils.common import load_samples, save_json, save_numpy, write_atomic
from utils.inference import CausalLMInference
from utils.sample_store import SampleStoreReader, read_store_config
from utils.stream_neural_svd import TokenStream

from .analysis import MATRIX_NAME


PMI_VALUE = "pmi[h, f] = log p(h + f) - log p(h) - log p(f), no BOS: log p(x) = sum_{t>=2} log p(x_t | x_<t), so the first token of each sequence is unscored; log p(h) cancels, pmi = sum_t log p(f_t | h + f_<t) - sum_{t>=2} log p(f_t | f_<t)"


@dataclass(frozen=True)
class Exp2Inputs:
    histories: list[list[int]]
    futures: list[list[int]]
    source: dict


def parquet_inputs(
    samples_path: Path, history_count: int | None, future_count: int | None
) -> Exp2Inputs:
    histories, futures = load_samples(samples_path, history_count, future_count)
    source_metadata = pq.read_schema(samples_path).metadata or {}
    raw_source_metadata = source_metadata.get(b"lm_svd")
    return Exp2Inputs(
        histories=[sample.token_ids for sample in histories],
        futures=[sample.token_ids for sample in futures],
        source={
            "kind": "parquet",
            "samples_path": str(samples_path),
            "history_sample_ids": [sample.sample_id for sample in histories],
            "future_sample_ids": [sample.sample_id for sample in futures],
            "source_metadata": json.loads(raw_source_metadata) if raw_source_metadata else None,
        },
    )


def sample_store_inputs(
    *,
    store_dir: Path,
    stream_path: Path,
    model_path: Path,
    history_count: int,
    future_count: int,
    history_tokens: int,
    future_tokens: int,
    seed: int,
    pairing: str,
) -> Exp2Inputs:
    if pairing not in ("independent", "adjacent"):
        raise ValueError(f"unknown pairing: {pairing}")
    if pairing == "adjacent" and history_count != future_count:
        raise ValueError("adjacent pairing requires history_count == future_count")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if history_count <= 0 or future_count <= 0:
        raise ValueError("history_count and future_count must be positive")
    stream = TokenStream(stream_path)
    config = read_store_config(store_dir)
    history_batch = min(history_count, config["block_histories"])
    future_batch = min(future_count, config["block_futures"]) if pairing == "independent" else 0
    if history_batch <= 0 or config["block_futures"] <= 0:
        raise ValueError("sample store block dimensions must be positive")
    reader = SampleStoreReader(
        store_dir, model_path=model_path, stream_path=stream_path, stream=stream,
        hidden_size=config["hidden_size"], vocab_size=config["vocab_size"],
        history_tokens=history_tokens, future_tokens=future_tokens,
        history_batch=history_batch, future_batch=future_batch,
    )
    reader.wait_until_ready()
    generator = torch.Generator().manual_seed(seed)
    history_parts = []
    future_parts = []
    blocks = []
    histories_read = futures_read = 0
    if pairing == "independent":
        while histories_read < history_count or futures_read < future_count:
            hs, fs, block = reader.sample_starts(generator)
            hs = hs[:history_count - histories_read]
            fs = fs[:future_count - futures_read]
            block["history_indices"] = block["history_indices"][:len(hs)]
            block["future_indices"] = block["future_indices"][:len(fs)]
            block["history_range"] = [histories_read, histories_read + len(hs)]
            block["future_range"] = [futures_read, futures_read + len(fs)]
            history_parts.append(hs)
            future_parts.append(fs)
            histories_read += len(hs)
            futures_read += len(fs)
            blocks.append(block)
        history_starts = torch.cat(history_parts)
        future_starts = torch.cat(future_parts)
    else:
        limit = len(stream) - history_tokens - future_tokens
        while histories_read < history_count:
            hs, _, block = reader.sample_starts(generator)
            hs = hs[:history_count - histories_read]
            block["history_indices"] = block["history_indices"][:len(hs)]
            invalid = hs[hs > limit]
            if len(invalid):
                start = int(invalid[0])
                raise ValueError(
                    f"history start {start} has no {future_tokens}-token continuation in the stream; choose another --seed"
                )
            block["future_indices"] = []
            block["history_range"] = [histories_read, histories_read + len(hs)]
            block["future_range"] = [histories_read, histories_read + len(hs)]
            history_parts.append(hs)
            blocks.append(block)
            histories_read += len(hs)
        history_starts = torch.cat(history_parts)
        future_starts = history_starts + history_tokens
    return Exp2Inputs(
        histories=stream.windows(history_starts, history_tokens).tolist(),
        futures=stream.windows(future_starts, future_tokens).tolist(),
        source={
            "kind": "sample_store",
            "store_dir": str(store_dir),
            "store_config": reader.config,
            "stream_path": str(stream_path),
            "stream_metadata": stream.metadata,
            "history_tokens": history_tokens,
            "future_tokens": future_tokens,
            "seed": seed,
            "pairing": pairing,
            "sampling": (
                "uniform blocks with replacement and independent row and column permutations via torch.Generator().manual_seed(seed)"
                if pairing == "independent" else
                "uniform blocks with replacement and history row permutations via torch.Generator().manual_seed(seed); future i is the future_tokens stream tokens immediately after history i (future_starts = history_starts + history_tokens), so diagonal entries are matched pairs"
            ),
            "history_starts": history_starts.tolist(),
            "future_starts": future_starts.tolist(),
            "blocks": blocks,
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
