from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from utils.common import save_json, write_atomic
from utils.inference import CausalLMInference
from utils.stream_neural_svd import TokenStream, teacher_hidden_grid

STORE_FORMAT = "lm-svd-sample-store-v1"
DEFAULT_MAX_GIB = 64.0
MIN_READY_BLOCKS = 64
WAIT_SECONDS = 30
FLOAT16_MAX = 65504.0


def slot_bytes(block_histories: int, block_futures: int, hidden_size: int) -> int:
    return block_histories * block_futures * hidden_size * 2 + (block_histories + block_futures) * 8


def store_config(*, model_path: Path, stream_path: Path, stream_tokens: int, vocab_size: int,
                 hidden_size: int, history_tokens: int, future_tokens: int, block_histories: int,
                 block_futures: int, max_bytes: int, shards: int, seed: int) -> dict:
    slot = slot_bytes(block_histories, block_futures, hidden_size)
    slots_per_shard = (max_bytes // slot) // shards
    if slots_per_shard == 0:
        raise ValueError(f"max_bytes {max_bytes} holds fewer than {shards} slots of {slot} bytes")
    return {
        "format": STORE_FORMAT,
        "model_path": str(model_path),
        "model_name": model_path.name,
        "stream_path": str(stream_path),
        "stream_name": stream_path.name,
        "stream_tokens": stream_tokens,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "history_tokens": history_tokens,
        "future_tokens": future_tokens,
        "block_histories": block_histories,
        "block_futures": block_futures,
        "state_dtype": "float16",
        "slot_bytes": slot,
        "max_bytes": max_bytes,
        "shards": shards,
        "slots_per_shard": slots_per_shard,
        "num_slots": slots_per_shard * shards,
        "seed": seed,
        "value": "final-position teacher hidden state of history+future; centered logits = states @ centered_unembedding.T",
    }


def read_store_config(store_dir: Path) -> dict:
    path = store_dir / "store.json"
    if not path.is_file():
        raise ValueError(f"sample store does not exist: {store_dir}; start a producer with: sample-store produce")
    return json.loads(path.read_text(encoding="utf-8"))


def read_shard_states(store_dir: Path, shards: int) -> list[dict]:
    result = []
    for shard in range(shards):
        path = store_dir / f"state-{shard:03d}.json"
        result.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file()
                      else {"next_sequence": 0, "filled": 0})
    return result


def slot_path(store_dir: Path, slot: int) -> Path:
    return store_dir / "slots" / f"{slot:07d}.npz"


def block_seed(seed: int, shard: int, sequence: int) -> int:
    return int(np.random.SeedSequence([seed, shard, sequence]).generate_state(1, dtype=np.uint64)[0])


def produce_blocks(*, store_dir: Path, model_path: Path, stream_path: Path, device: str,
                   history_tokens: int, future_tokens: int, block_histories: int,
                   block_futures: int, max_bytes: int, shards: int, shard: int, seed: int,
                   pair_batch: int, max_blocks: int | None, log_every: int, overwrite: bool) -> dict:
    if not 0 <= shard < shards:
        raise ValueError(f"shard {shard} outside [0, {shards})")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    inference = CausalLMInference(model_path, device)
    stream = TokenStream(stream_path)
    max_token = int(stream.tokens.max())
    if max_token >= inference.vocab_size:
        raise ValueError(f"token stream contains id {max_token} outside vocabulary {inference.vocab_size}")
    if stream.tokens.dtype.kind == "i":
        min_token = int(stream.tokens.min())
        if min_token < 0:
            raise ValueError(f"token stream contains id {min_token} outside vocabulary {inference.vocab_size}")
    length = history_tokens + future_tokens
    if length > inference.max_length:
        raise ValueError(f"history+future length {length} exceeds model maximum {inference.max_length}")
    config = store_config(
        model_path=model_path, stream_path=stream_path, stream_tokens=len(stream),
        vocab_size=inference.vocab_size,
        hidden_size=inference.model.get_output_embeddings().weight.shape[1],
        history_tokens=history_tokens, future_tokens=future_tokens,
        block_histories=block_histories, block_futures=block_futures,
        max_bytes=max_bytes, shards=shards, seed=seed,
    )
    if overwrite:
        shutil.rmtree(store_dir, ignore_errors=True)
    (store_dir / "slots").mkdir(parents=True, exist_ok=True)
    # write_atomic uses a fixed temporary filename, so first-time initialization
    # and configuration checks must be serialized across producers.
    directory_fd = os.open(store_dir, os.O_RDONLY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        config_path = store_dir / "store.json"
        if config_path.exists():
            saved = read_store_config(store_dir)
            if saved != config:
                keys = dict.fromkeys((*saved, *config))
                diffs = ", ".join(f"{key}: {saved.get(key)!r} != {config.get(key)!r}"
                                  for key in keys if saved.get(key) != config.get(key))
                raise ValueError(f"sample store config mismatch ({diffs}); pass --overwrite to recreate {store_dir}")
        else:
            write_atomic(config_path, lambda tmp: save_json(tmp, config))
    finally:
        os.close(directory_fd)

    fd = os.open(store_dir / f"shard-{shard:03d}.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"another producer holds shard {shard} of {store_dir}") from error
        sequence = read_shard_states(store_dir, shards)[shard]["next_sequence"]
        slots_per_shard = config["slots_per_shard"]
        produced = 0
        start_time = time.monotonic()
        while max_blocks is None or produced < max_blocks:
            gen = torch.Generator().manual_seed(block_seed(seed, shard, sequence))
            hs = stream.sample_starts(block_histories, history_tokens, gen)
            fs = stream.sample_starts(block_futures, future_tokens, gen)
            states = teacher_hidden_grid(
                inference, stream.windows(hs, history_tokens), stream.windows(fs, future_tokens),
                pair_batch=pair_batch,
            ).permute(1, 0, 2).contiguous()
            if not torch.isfinite(states).all() or states.abs().max() > FLOAT16_MAX:
                raise ValueError(f"block {sequence} of shard {shard} has hidden states that are non-finite or exceed float16 range")
            slot = (sequence % slots_per_shard) * shards + shard

            def save_slot(tmp: Path) -> None:
                with tmp.open("wb") as handle:
                    np.savez(handle, history_starts=hs.numpy(), future_starts=fs.numpy(),
                             states=states.to(torch.float16).cpu().numpy(),
                             sequence=np.int64(sequence), shard=np.int64(shard))

            write_atomic(slot_path(store_dir, slot), save_slot)
            sequence += 1
            produced += 1
            filled = min(sequence, slots_per_shard)
            state = {"next_sequence": sequence, "filled": filled}
            write_atomic(store_dir / f"state-{shard:03d}.json", lambda tmp: save_json(tmp, state))
            if produced % log_every == 0:
                elapsed = time.monotonic() - start_time
                print(f"shard {shard} block {sequence} slot {slot} filled {filled}/{slots_per_shard} {produced / elapsed:.3f} blocks/s", flush=True)
        return {"store_dir": str(store_dir), "shard": shard, "produced": produced,
                "next_sequence": sequence, "filled": min(sequence, slots_per_shard),
                "slots_per_shard": slots_per_shard, "num_slots": config["num_slots"]}
    finally:
        os.close(fd)


class SampleStoreReader:
    def __init__(self, store_dir: Path, *, model_path: Path, stream_path: Path, stream: TokenStream,
                 hidden_size: int, vocab_size: int, history_tokens: int, future_tokens: int,
                 history_batch: int, future_batch: int) -> None:
        self.store_dir = store_dir
        self.config = read_store_config(store_dir)
        config = self.config
        if config["format"] != STORE_FORMAT:
            raise ValueError(f"unsupported sample store format {config['format']!r} in {store_dir}")
        expected = {
            "model_name": model_path.name, "stream_name": stream_path.name,
            "stream_tokens": len(stream), "hidden_size": hidden_size, "vocab_size": vocab_size,
            "history_tokens": history_tokens, "future_tokens": future_tokens,
        }
        for key, value in expected.items():
            if config[key] != value:
                raise ValueError(f"sample store {store_dir} does not match training ({key}: store {config[key]!r} != training {value!r})")
        for name, key, count in (
            ("history", "block_histories", history_batch),
            ("future", "block_futures", future_batch),
        ):
            limit = config[key]
            if count > limit:
                raise ValueError(f"{name}_batch {count} exceeds sample store {key} {limit}")
        self.history_batch = history_batch
        self.future_batch = future_batch
        self._filled = [0] * config["shards"]

    @property
    def identity(self) -> dict:
        return {"store_dir": str(self.store_dir), **self.config}

    def wait_until_ready(self) -> None:
        ready = min(MIN_READY_BLOCKS, self.config["num_slots"])
        while True:
            self._filled = [state["filled"] for state in read_shard_states(self.store_dir, self.config["shards"])]
            total = sum(self._filled)
            if total >= ready:
                return
            print(f"waiting for sample store {self.store_dir}: {total}/{ready} blocks", flush=True)
            time.sleep(WAIT_SECONDS)

    def sample(self, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sum(self._filled) < self.config["num_slots"]:
            self._filled = [state["filled"] for state in read_shard_states(self.store_dir, self.config["shards"])]
        local = int(torch.randint(sum(self._filled), (1,), generator=generator))
        for shard, filled in enumerate(self._filled):
            if local < filled:
                break
            local -= filled
        slot = local * self.config["shards"] + shard
        with np.load(slot_path(self.store_dir, slot)) as data:
            hs = torch.from_numpy(data["history_starts"])
            fs = torch.from_numpy(data["future_starts"])
            states = torch.from_numpy(data["states"])
        rows = torch.randperm(self.config["block_histories"], generator=generator)[:self.history_batch]
        cols = torch.randperm(self.config["block_futures"], generator=generator)[:self.future_batch]
        return hs[rows], fs[cols], states[rows][:, cols]
