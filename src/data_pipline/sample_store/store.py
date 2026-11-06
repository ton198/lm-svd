"""On-disk training samples and model-free random sampling."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from .state import read_store_state

STORE_FORMAT = "lm-svd-sample-store-v3"
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

    def _published(self) -> list[int]:
        slots = sorted(int(slot) for slot in read_store_state(self.store_dir)["published"])
        if any(slot < 0 or slot >= self.config["num_slots"] for slot in slots):
            raise ValueError(f"invalid sample store progress in {self.store_dir}")
        return slots

    def wait_until_ready(self) -> None:
        """Wait for the first complete block; production can continue afterward."""
        while not self._published():
            time.sleep(0.1)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = self._published()
        block_size = self.config["samples_per_block"]
        total = len(slots) * block_size
        if not total:
            raise ValueError(f"no published samples in {self.store_dir}")
        index = int(self._generator.integers(total))
        local_slot, row_index = divmod(index, block_size)
        slot = slots[local_slot]
        # Opening once keeps all three fields on one inode across atomic replaces.
        block = np.load(slot_path(self.store_dir, slot), mmap_mode="r", allow_pickle=False)
        try:
            if block.shape != (block_size,) or block.dtype != self._dtype:
                raise ValueError(f"invalid sample block {slot} in {self.store_dir}")
            return (
                torch.from_numpy(block["h"][row_index].copy()),
                torch.from_numpy(block["f"][row_index].copy()),
                torch.from_numpy(block["logits"][row_index].copy()),
            )
        finally:
            block._mmap.close()
