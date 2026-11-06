from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from transformers import AutoTokenizer

from utils.common import save_json, save_numpy, write_atomic

from .windows import load_split_text, tokenize_stream


def tokenizer_fingerprint(tokenizer) -> str:
    """Identify tokenization semantics independent of the tokenizer directory."""
    config = json.loads(tokenizer.backend_tokenizer.to_str())
    # These runtime options are not used by our unpadded, untruncated tokenizer calls.
    config.pop("padding", None)
    config.pop("truncation", None)
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "vocab_size": len(tokenizer),
        "stream_tokens": len(stream),
        "dtype": str(stream.dtype),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, lambda path: save_numpy(path, stream))
    write_atomic(output.with_suffix(".json"), lambda path: save_json(path, metadata))
    return {"output": str(output), "stream_tokens": len(stream)}


class TokenStream:
    def __init__(self, path: Path) -> None:
        self.tokens = np.load(path, mmap_mode="r")
        if self.tokens.ndim != 1 or self.tokens.dtype.kind not in "iu":
            raise ValueError(f"token stream must be a 1-D integer array: {path}")
        json_path = path.with_suffix(".json")
        if not json_path.is_file():
            raise ValueError(f"token stream metadata does not exist: {json_path}")
        self.metadata = json.loads(json_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.tokens)

    def sample_starts(
        self, count: int, window: int, generator: torch.Generator
    ) -> torch.Tensor:
        if len(self) < window:
            raise ValueError(
                f"token stream has {len(self)} tokens, fewer than window {window}"
            )
        return torch.randint(0, len(self) - window + 1, (count,), generator=generator)

    def windows(self, starts: torch.Tensor, window: int) -> torch.Tensor:
        indices = starts.cpu().numpy()[:, None] + np.arange(window)
        return torch.from_numpy(self.tokens[indices].astype(np.int64, copy=False))
