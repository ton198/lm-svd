"""On-disk training samples and model-free random sampling."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import torch

STORE_FORMAT = "lm-svd-sample-store-v2"
DEFAULT_MAX_GIB = 64.0


def sample_dtype(history_tokens: int, future_tokens: int, vocab_size: int) -> np.dtype:
    return np.dtype([
        ("h", "<i8", (history_tokens,)),
        ("f", "<i8", (future_tokens,)),
        ("logits", "<f4", (vocab_size,)),
    ])


def slot_bytes(samples_per_block: int, history_tokens: int, future_tokens: int, vocab_size: int) -> int:
    dtype = sample_dtype(history_tokens, future_tokens, vocab_size)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {
        "descr": np.lib.format.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": (samples_per_block,),
    })
    return header.tell() + samples_per_block * dtype.itemsize


def read_store_config(store_dir: Path) -> dict:
    path = store_dir / "store.json"
    if not path.is_file():
        raise ValueError(
            f"sample store does not exist: {store_dir}; start data-pipline sample-store produce"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("format") != STORE_FORMAT:
        raise ValueError(
            f"unsupported sample store format {config.get('format')!r}; "
            "rebuild explicitly with data-pipline sample-store produce --overwrite"
        )
    return config


def read_shard_states(store_dir: Path, shards: int) -> list[dict]:
    states = []
    for shard in range(shards):
        path = store_dir / f"state-{shard:03d}.json"
        states.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file()
                      else {"next_sequence": 0, "filled": 0})
    return states


def slot_path(store_dir: Path, slot: int) -> Path:
    return store_dir / "slots" / f"{slot:07d}.npy"


class SampleSampler:
    """Uniformly sample persisted (history, future, raw next-token logits).

    A seed reproduces sampling only while store contents stay fixed. Returned
    CPU tensors own their memory, even when producers replace the selected slot.
    """

    def __init__(self, store_dir: str | Path, *, seed: int = 0) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.store_dir = Path(store_dir)
        self.config = read_store_config(self.store_dir)
        self._generator = np.random.default_rng(seed)
        self._dtype = sample_dtype(
            self.config["history_tokens"], self.config["future_tokens"], self.config["vocab_size"]
        )

    def _filled(self) -> list[int]:
        states = read_shard_states(self.store_dir, self.config["shards"])
        filled = [state["filled"] for state in states]
        if any(n < 0 or n > self.config["slots_per_shard"] for n in filled):
            raise ValueError(f"invalid sample store progress in {self.store_dir}")
        return filled

    def wait_until_ready(self) -> None:
        """Wait for the first complete block; production can continue afterward."""
        while not sum(self._filled()):
            time.sleep(0.1)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        filled = self._filled()
        block_size = self.config["samples_per_block"]
        total = sum(filled) * block_size
        if not total:
            raise ValueError(f"no published samples in {self.store_dir}")
        index = int(self._generator.integers(total))
        local_slot, row_index = divmod(index, block_size)
        for shard, count in enumerate(filled):
            if local_slot < count:
                break
            local_slot -= count
        slot = local_slot * self.config["shards"] + shard
        # Opening once keeps all three fields on one inode across atomic replaces.
        block = np.load(slot_path(self.store_dir, slot), mmap_mode="r", allow_pickle=False)
        try:
            if block.shape != (block_size,) or block.dtype != self._dtype:
                raise ValueError(f"invalid sample block {slot} in {self.store_dir}")
            return tuple(torch.from_numpy(block[name][row_index].copy())
                         for name in ("h", "f", "logits"))
        finally:
            block._mmap.close()
