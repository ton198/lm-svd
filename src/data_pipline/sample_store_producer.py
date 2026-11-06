"""Continuously produce independent WikiText pairs and their teacher logits."""

from __future__ import annotations

import fcntl
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from utils.common import save_json, save_numpy, write_atomic
from utils.inference import CausalLMInference

from .sample_store import (
    STORE_FORMAT, read_shard_states, read_store_config, sample_dtype, slot_bytes, slot_path,
)
from .token_stream import TokenStream, tokenizer_fingerprint


def block_seed(seed: int, shard: int, sequence: int) -> int:
    return int(np.random.SeedSequence([seed, shard, sequence]).generate_state(1, dtype=np.uint64)[0])


def model_file_identity(model_path: Path) -> dict:
    """Detect local checkpoint replacement before resuming a rolling dataset."""
    paths = {model_path / "config.json"}
    for pattern in ("*.safetensors", "*.bin", "*.index.json"):
        paths.update(model_path.glob(pattern))
    result = {}
    for path in sorted(paths):
        stat = path.stat()
        result[path.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def produce_blocks(
    *, store_dir: Path, model_path: Path, stream_path: Path, device: str = "cuda",
    history_tokens: int = 64, future_tokens: int = 32, samples_per_block: int = 64,
    max_bytes: int = 64 * 1024**3, shards: int = 1, shard: int = 0, seed: int = 0,
    batch_size: int = 64, max_blocks: int | None = None, log_every: int = 100,
    overwrite: bool = False,
) -> dict:
    for key, value in (("history_tokens", history_tokens), ("future_tokens", future_tokens),
                       ("samples_per_block", samples_per_block), ("max_bytes", max_bytes),
                       ("shards", shards), ("batch_size", batch_size), ("log_every", log_every)):
        if value <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= shard < shards:
        raise ValueError(f"shard {shard} outside [0, {shards})")
    if seed < 0 or (max_blocks is not None and max_blocks < 0):
        raise ValueError("seed and max_blocks must be non-negative")

    store_dir = Path(store_dir)
    model_path, stream_path = Path(model_path), Path(stream_path)
    store_dir.mkdir(parents=True, exist_ok=True)
    # Hold a shared directory lock for the producer lifetime. Rebuild needs an
    # exclusive lock, so it cannot remove another producer's locks or live files.
    directory_fd = os.open(store_dir, os.O_RDONLY)
    shard_fd = None
    try:
        try:
            fcntl.flock(directory_fd, (fcntl.LOCK_EX if overwrite else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"sample store {store_dir} has active producers or is being rebuilt") from error
        with (store_dir / ".init.lock").open("a") as init_lock:
            fcntl.flock(init_lock, fcntl.LOCK_EX)
            saved = None
            if (store_dir / "store.json").exists() and not overwrite:
                saved = read_store_config(store_dir)
            # On restart reject duplicate ownership before loading the model.
            if not overwrite:
                shard_fd = os.open(store_dir / f"shard-{shard:03d}.lock", os.O_RDWR | os.O_CREAT, 0o644)
                try:
                    fcntl.flock(shard_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    raise ValueError(f"another producer holds shard {shard} of {store_dir}") from error
            stream = TokenStream(stream_path)
            model_files = model_file_identity(model_path)
            inference = CausalLMInference(model_path, device)
            if model_file_identity(model_path) != model_files:
                raise ValueError("model files changed while loading; restart with a stable checkpoint")
            if history_tokens + future_tokens > inference.max_length:
                raise ValueError("history+future length exceeds model maximum")
            if len(stream) < max(history_tokens, future_tokens):
                raise ValueError("token stream is shorter than a sample window")
            if int(stream.tokens.min()) < 0 or int(stream.tokens.max()) >= inference.vocab_size:
                raise ValueError("token stream contains IDs outside the model vocabulary")
            if stream.metadata.get("vocab_size") != inference.vocab_size:
                raise ValueError("token stream vocabulary does not match model vocabulary")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, local_files_only=True)
            teacher_tokenizer = tokenizer_fingerprint(tokenizer)
            source_tokenizer = stream.metadata.get("tokenizer_fingerprint")
            if source_tokenizer is None:
                # Pre-upgrade streams record a path, but not a semantic snapshot.
                source_path = stream.metadata.get("tokenizer_path")
                if source_path is None:
                    raise ValueError("token stream has no tokenizer provenance; rebuild it with build-token-stream")
                source = AutoTokenizer.from_pretrained(str(source_path), use_fast=True, local_files_only=True)
                source_tokenizer = tokenizer_fingerprint(source)
            if source_tokenizer != teacher_tokenizer:
                raise ValueError("token stream tokenizer does not match the teacher tokenizer; rebuild the stream")
            block_bytes = slot_bytes(samples_per_block, history_tokens, future_tokens, inference.vocab_size)
            slots_per_shard = (max_bytes // block_bytes) // shards
            if not slots_per_shard:
                raise ValueError(f"max_bytes {max_bytes} cannot hold one {block_bytes}-byte block per shard")
            stat = stream_path.stat()
            config = {
                "format": STORE_FORMAT, "model_path": str(model_path.resolve()),
                "model_files": model_files, "tokenizer_fingerprint": teacher_tokenizer,
                "model_name": model_path.name, "stream_path": str(stream_path.resolve()),
                "stream_name": stream_path.stem, "stream_tokens": len(stream),
                "stream_metadata": stream.metadata,
                "stream_size": stat.st_size, "stream_mtime_ns": stat.st_mtime_ns,
                "history_tokens": history_tokens, "future_tokens": future_tokens,
                "vocab_size": inference.vocab_size, "logits_dtype": "float32",
                "samples_per_block": samples_per_block, "slot_bytes": block_bytes,
                "max_bytes": max_bytes, "shards": shards, "slots_per_shard": slots_per_shard,
                "num_slots": slots_per_shard * shards, "seed": seed,
                "value": "raw full-vocabulary next-token logits of concat(h,f), without BOS or separators",
            }
            if saved is not None and saved != config:
                differences = ", ".join(k for k in dict.fromkeys((*saved, *config)) if saved.get(k) != config.get(k))
                raise ValueError(f"sample store config mismatch ({differences}); use --overwrite to rebuild")
            if overwrite:
                shutil.rmtree(store_dir / "slots", ignore_errors=True)
                for pattern in ("state-*.json*", "shard-*.lock", "store.json*"):
                    for path in store_dir.glob(pattern):
                        path.unlink()
                shard_fd = os.open(store_dir / f"shard-{shard:03d}.lock", os.O_RDWR | os.O_CREAT, 0o644)
                fcntl.flock(shard_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (store_dir / "slots").mkdir(exist_ok=True)
            if saved is None:
                write_atomic(store_dir / "store.json", lambda tmp: save_json(tmp, config))
            # Other shards can initialize while this producer continues.
            if overwrite:
                fcntl.flock(directory_fd, fcntl.LOCK_SH)

        sequence = read_shard_states(store_dir, shards)[shard]["next_sequence"]
        produced = 0
        started = time.monotonic()
        dtype = sample_dtype(history_tokens, future_tokens, inference.vocab_size)
        while max_blocks is None or produced < max_blocks:
            generator = torch.Generator().manual_seed(block_seed(seed, shard, sequence))
            hs = stream.sample_starts(samples_per_block, history_tokens, generator)
            fs = stream.sample_starts(samples_per_block, future_tokens, generator)
            histories = stream.windows(hs, history_tokens)
            futures = stream.windows(fs, future_tokens)
            block = np.empty(samples_per_block, dtype=dtype)
            block["h"], block["f"] = histories.numpy(), futures.numpy()
            for start in range(0, samples_per_block, batch_size):
                pairs = torch.cat((histories[start:start + batch_size], futures[start:start + batch_size]), dim=1)
                logits = inference.next_token_logits_batch(pairs.tolist()).cpu().numpy()
                if not np.isfinite(logits).all():
                    raise ValueError(f"non-finite logits in shard {shard} block {sequence}")
                block["logits"][start:start + len(pairs)] = logits
            slot = (sequence % slots_per_shard) * shards + shard
            write_atomic(slot_path(store_dir, slot), lambda tmp: save_numpy(tmp, block))
            sequence += 1
            produced += 1
            state = {"next_sequence": sequence, "filled": min(sequence, slots_per_shard)}
            write_atomic(store_dir / f"state-{shard:03d}.json", lambda tmp: save_json(tmp, state))
            if produced % log_every == 0:
                rate = produced / (time.monotonic() - started)
                print(f"shard {shard} block {sequence} slot {slot} {rate:.3f} blocks/s", flush=True)
        return {"store_dir": str(store_dir), "shard": shard, "produced": produced,
                "next_sequence": sequence, "filled": min(sequence, slots_per_shard),
                "slots_per_shard": slots_per_shard, "num_slots": config["num_slots"]}
    finally:
        if shard_fd is not None:
            os.close(shard_fd)
        os.close(directory_fd)
