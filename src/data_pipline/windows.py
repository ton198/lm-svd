from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


@dataclass(frozen=True)
class Window:
    group: Literal["history", "future"]
    start_token: int
    token_ids: np.ndarray


def load_split_text(dataset_dir: Path, config: str, split: str) -> Iterator[list[str]]:
    pattern = dataset_dir / config / f"{split}-*.parquet"
    paths = sorted((dataset_dir / config).glob(f"{split}-*.parquet"))
    if not paths:
        raise ValueError(f"no parquet files match {pattern}")
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=10_000, columns=["text"]):
            texts = batch.column("text").to_pylist()
            if any(text is None for text in texts):
                raise ValueError(f"null text in {path}")
            yield texts


def tokenize_stream(text_batches: Iterable[list[str]], tokenizer) -> np.ndarray:
    vocabulary_size = len(tokenizer)
    dtype = np.uint16 if vocabulary_size <= 65_536 else np.uint32
    token_batches: list[np.ndarray] = []
    for batch in text_batches:
        encoded = tokenizer(batch, add_special_tokens=False, verbose=False)["input_ids"]
        flat_ids = (token for row in encoded for token in row)
        values = np.fromiter(flat_ids, dtype=dtype)
        if values.size and np.any(values >= vocabulary_size):
            raise ValueError("tokenizer returned token ID outside its vocabulary")
        token_batches.append(values)
    if not token_batches or not any(batch.size for batch in token_batches):
        raise ValueError("tokenized split is empty")
    return np.concatenate(token_batches)


def select_windows(
    stream: np.ndarray,
    history_count: int,
    history_tokens: int,
    future_count: int,
    future_tokens: int,
    seed: int,
) -> list[Window]:
    slot_tokens = max(history_tokens, future_tokens)
    slot_count = len(stream) // slot_tokens
    order = np.random.default_rng(seed).permutation(slot_count)
    targets = {"history": history_count, "future": future_count}
    lengths = {"history": history_tokens, "future": future_tokens}
    groups: dict[str, list[Window]] = {"history": [], "future": []}
    seen: set[tuple[int, ...]] = set()
    next_group = "history"
    for slot in order:
        while next_group in groups and len(groups[next_group]) >= targets[next_group]:
            next_group = "future" if next_group == "history" else "done"
            if next_group == "done":
                break
        if next_group == "done":
            break
        group = next_group
        start = int(slot) * slot_tokens
        token_ids = stream[start : start + lengths[group]]
        key = tuple(int(token) for token in token_ids)
        if key in seen:
            continue
        seen.add(key)
        groups[group].append(Window(group, start, token_ids.copy()))
    windows = groups["history"] + groups["future"]
    if len(windows) != history_count + future_count:
        needed = history_count + future_count
        raise ValueError(
            f"not enough distinct non-overlapping windows: need {needed}, selected {len(windows)} "
            f"from {slot_count} slots of {slot_tokens} tokens"
        )
    return sorted(groups["history"], key=lambda window: window.start_token) + sorted(
        groups["future"], key=lambda window: window.start_token
    )


def build_windows_table(
    *,
    dataset_dir: Path,
    config: str,
    split: str,
    tokenizer_path: Path,
    history_count: int,
    history_tokens: int,
    future_count: int,
    future_tokens: int,
    seed: int,
    output: Path,
    overwrite: bool,
) -> dict[str, int | str]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
    if not tokenizer_path.is_dir():
        raise ValueError(f"tokenizer directory does not exist: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), use_fast=True, local_files_only=True
        )
    except OSError as error:
        raise ValueError(f"cannot load local tokenizer from {tokenizer_path}: {error}") from error

    stream = tokenize_stream(load_split_text(dataset_dir, config, split), tokenizer)
    windows = select_windows(
        stream, history_count, history_tokens, future_count, future_tokens, seed
    )
    rows = []
    for sample_id, window in enumerate(windows):
        token_ids = window.token_ids.tolist()
        rows.append(
            (
                sample_id,
                window.group,
                len(token_ids),
                window.start_token,
                token_ids,
                tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            )
        )
    metadata = {
        "config": config,
        "split": split,
        "tokenizer_path": str(tokenizer_path),
        "seed": seed,
        "history_count": history_count,
        "history_tokens": history_tokens,
        "future_count": future_count,
        "future_tokens": future_tokens,
        "stream_tokens": len(stream),
        "slot_tokens": max(history_tokens, future_tokens),
    }
    schema = pa.schema(
        [
            ("sample_id", pa.int32()),
            ("group", pa.string()),
            ("token_count", pa.int32()),
            ("start_token", pa.int64()),
            ("token_ids", pa.list_(pa.int32())),
            ("text", pa.string()),
        ],
        metadata={b"lm_svd": json.dumps(metadata, ensure_ascii=False).encode("utf-8")},
    )
    arrays = [
        pa.array([row[index] for row in rows], type=field.type)
        for index, field in enumerate(schema)
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        pq.write_table(table, temporary)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "output": str(output),
        "stream_tokens": len(stream),
        "history_count": history_count,
        "history_tokens": history_tokens,
        "future_count": future_count,
        "future_tokens": future_tokens,
    }
