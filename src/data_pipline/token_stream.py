from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from utils.common import save_json, save_numpy, write_atomic

from .windows import load_split_text, tokenize_stream


def build_token_stream(
    *,
    dataset_dir: Path,
    config: str,
    split: str,
    tokenizer_path: Path,
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
    metadata = {
        "config": config,
        "split": split,
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": len(tokenizer),
        "stream_tokens": len(stream),
        "dtype": str(stream.dtype),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, lambda path: save_numpy(path, stream))
    write_atomic(output.with_suffix(".json"), lambda path: save_json(path, metadata))
    return {"output": str(output), "stream_tokens": len(stream)}
