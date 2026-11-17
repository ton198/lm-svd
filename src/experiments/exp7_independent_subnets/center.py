"""exp7 on the table minus a column center taken from fixed histories.

exp5-exp7 subtract g, the entry of a column after its future alone. What is
left still has a mean over the histories in every column, and that mean is the
leading mode. Here the baseline of a column (future, z) is instead its mean
over M fixed histories h_1 .. h_M drawn once from the token stream:

    target(h, f, z) = A[h, (f, z)] - (1 / M) sum_m A[h_m, (f, z)]

An entry is <s(h + f), W_z - mean row of W>, with s the final hidden state at
the last token and W the unembedding, so the center of all tokens of a future
is one vector, the mean of s(h_m + f) over the fixed histories. It costs M
passes of the frozen model per future. A center pool holds that vector for a
fixed set of futures, computed once; training then draws its futures from the
pool and pairs each with a fresh history from the stream.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from data_pipline.token_stream import TokenStream
from experiments.exp4_table_neural_svd import TABLE_NAMES
from experiments.exp5_gpt2_mlp_table.features import future_only_logits, header_tokens, last_hidden
from experiments.exp5_gpt2_mlp_table.tables import EvalTable, load_eval_table
from utils.common import save_json, write_atomic
from utils.inference import CausalLMInference

FIRST_MODE = "none; every column minus its mean over the fixed histories of the center pool"
POOL_FILES = ("histories.npy", "futures.npy", "states.npy", "progress.json", "metadata.json")


def centered_rows(inference: CausalLMInference) -> torch.Tensor:
    """(V, E): the unembedding rows minus their mean row, on the model's device.

    The entry of token z after a sequence x is <s(x), row z>.
    """
    output = inference.model.get_output_embeddings()
    if output.bias is not None:
        raise ValueError("the column center needs an unembedding without a bias, as GPT-2's is")
    weight = output.weight.detach().float()
    return weight - weight.mean(dim=0)


@torch.no_grad()
def center_states(
    inference: CausalLMInference, history_ids: torch.Tensor, future_ids: torch.Tensor,
    chunk: int = 1024,
) -> torch.Tensor:
    """(F, E) float32: the mean over the histories (M, H) of s(history + future).

    Each history is read once per block of futures and its attention cache is
    shared by the futures behind it; chunk is the sequences per pass. On a GPU
    the passes run in half precision.
    """
    device = inference.device
    base = inference.model.base_model
    history_ids = history_ids.to(device)
    block_size = max(chunk // 16, 1)
    means = []
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for block_start in range(0, len(future_ids), block_size):
            block = future_ids[block_start:block_start + block_size].to(device)
            group = max(chunk // len(block), 1)
            total = None
            for start in range(0, len(history_ids), group):
                part = history_ids[start:start + group]
                cache = base(input_ids=part, use_cache=True).past_key_values
                cache.batch_repeat_interleave(len(block))
                hidden = base(
                    input_ids=block.repeat(len(part), 1), past_key_values=cache, use_cache=True,
                ).last_hidden_state[:, -1].float()
                hidden = hidden.unflatten(0, (len(part), len(block))).sum(dim=0)
                total = hidden if total is None else total + hidden
            means.append(total / len(history_ids))
    return torch.cat(means)


def _pool_config(
    *, model_path: Path, stream_path: Path, histories: int, futures: int,
    history_tokens: int, future_tokens: int, seed: int, hidden_size: int, vocab_size: int,
) -> dict:
    return {
        "model_path": str(model_path), "stream_path": str(stream_path),
        "histories": histories, "futures": futures, "history_tokens": history_tokens,
        "future_tokens": future_tokens, "seed": seed, "hidden_size": hidden_size,
        "vocab_size": vocab_size,
    }


def build_center_pool(
    *,
    pool_dir: Path,
    model_path: Path,
    stream_path: Path,
    device: str,
    histories: int,
    futures: int,
    history_tokens: int,
    future_tokens: int,
    seed: int,
    overwrite: bool,
    chunk: int = 1024,
    save_every: int = 4096,
) -> dict:
    """Compute the center of `futures` stream windows over `histories` fixed ones.

    Writes histories.npy (M, H), futures.npy (P, T), states.npy (P, E) and, last,
    metadata.json. progress.json counts the futures whose state is on disk; the
    same command continues an interrupted build. A finished pool of the same
    settings is left as it is; other settings in pool_dir need overwrite.
    """
    for name, value in (("histories", histories), ("futures", futures),
                        ("history_tokens", history_tokens), ("future_tokens", future_tokens),
                        ("save_every", save_every)):
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    pool_dir = Path(pool_dir)
    stream = TokenStream(Path(stream_path))
    inference = CausalLMInference(model_path, device)
    if stream.metadata.get("vocab_size") != inference.vocab_size:
        raise ValueError(
            f"token stream {stream_path} has vocabulary {stream.metadata.get('vocab_size')}, "
            f"the model has {inference.vocab_size}")
    if history_tokens + future_tokens > inference.max_length:
        raise ValueError(
            f"history ({history_tokens}) + future ({future_tokens}) tokens exceed the "
            f"model's {inference.max_length} positions")
    hidden_size = inference.model.config.hidden_size
    centered_rows(inference)  # rejects an unembedding with a bias before anything is written
    config = _pool_config(
        model_path=model_path, stream_path=stream_path, histories=histories, futures=futures,
        history_tokens=history_tokens, future_tokens=future_tokens, seed=seed,
        hidden_size=hidden_size, vocab_size=inference.vocab_size)

    if overwrite:
        for name in POOL_FILES:
            (pool_dir / name).unlink(missing_ok=True)
    metadata_path, progress_path = pool_dir / "metadata.json", pool_dir / "progress.json"
    saved_path = metadata_path if metadata_path.exists() else progress_path
    done = 0
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if saved["config"] != config:
            changed = {key: (saved["config"].get(key), config[key])
                       for key in config if saved["config"].get(key) != config[key]}
            raise ValueError(
                f"settings differ from the center pool in {pool_dir} (saved, now): {changed}; "
                "pass --overwrite to start over")
        if saved_path == metadata_path:
            print(f"{pool_dir} already holds a finished center pool", flush=True)
            return {"pool_dir": str(pool_dir), "futures": futures, "histories": histories}
        done = saved["done"]
    pool_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(seed)
    history_ids = stream.windows(
        stream.sample_starts(histories, history_tokens, generator), history_tokens)
    future_ids = stream.windows(
        stream.sample_starts(futures, future_tokens, generator), future_tokens)
    states_path = pool_dir / "states.npy"
    if done == 0:
        np.save(pool_dir / "histories.npy", history_ids.numpy().astype(np.int32))
        np.save(pool_dir / "futures.npy", future_ids.numpy().astype(np.int32))
        states = np.lib.format.open_memmap(
            states_path, mode="w+", dtype=np.float32, shape=(futures, hidden_size))
    else:
        states = np.lib.format.open_memmap(states_path, mode="r+")
        print(f"Resuming the center pool at future {done}", flush=True)

    def save_progress(count: int) -> None:
        # The states first: progress.json never counts a state that is not on disk.
        states.flush()
        write_atomic(progress_path, lambda tmp: save_json(tmp, {"config": config, "done": count}))

    started = time.perf_counter()
    first = done
    while done < futures:
        stop = min(done + save_every, futures)
        states[done:stop] = center_states(
            inference, history_ids, future_ids[done:stop], chunk).cpu().numpy()
        done = stop
        save_progress(done)
        elapsed = time.perf_counter() - started
        print(
            f"center pool: {done}/{futures} futures  "
            f"{(done - first) * histories / elapsed:.0f} passes/s  elapsed {elapsed:.0f}s",
            flush=True)
    del states
    write_atomic(metadata_path, lambda tmp: save_json(tmp, {
        "config": config,
        "state": "mean over the fixed histories of the final hidden state at the last token "
                 "of history + future",
        "windows": "histories and futures are uniform windows of the token stream, drawn "
                   "with replacement from the seed: the histories first, then the futures",
    }))
    progress_path.unlink(missing_ok=True)
    return {"pool_dir": str(pool_dir), "futures": futures, "histories": histories}


def read_pool_config(pool_dir: Path) -> dict:
    """The settings of a finished center pool."""
    path = Path(pool_dir) / "metadata.json"
    if not path.is_file():
        raise ValueError(
            f"no finished center pool in {pool_dir}; build it with `exp7 center-pool`")
    return json.loads(path.read_text(encoding="utf-8"))["config"]


class CenterPool:
    """A finished center pool: the fixed histories, the futures and their center states."""

    def __init__(self, pool_dir: Path) -> None:
        pool_dir = Path(pool_dir)
        self.config = read_pool_config(pool_dir)
        self.histories = torch.from_numpy(np.load(pool_dir / "histories.npy")).long()  # (M, H)
        self.futures = torch.from_numpy(np.load(pool_dir / "futures.npy")).long()  # (P, T)
        self.states = torch.from_numpy(np.load(pool_dir / "states.npy"))  # (P, E) float32
        if len(self.states) != len(self.futures):
            raise ValueError(
                f"center pool {pool_dir} holds {len(self.futures)} futures and "
                f"{len(self.states)} states")


def table_baseline(
    inference: CausalLMInference, pool: CenterPool, chunk: int = 1024
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """The center of table columns: (F, T) futures and (F, K) header tokens -> (F, K)."""
    rows = centered_rows(inference)

    def baseline(futures: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        states = center_states(inference, pool.histories, futures, chunk)
        return torch.einsum("fe,fke->fk", states, rows[header_ids.to(rows.device)])

    return baseline


class CenteredPairs:
    """Training pairs on the centered target: pool futures behind fresh stream histories."""

    def __init__(
        self, inference: CausalLMInference, pool: CenterPool, top_k: int, seed: int,
        encode_batch: int,
    ) -> None:
        self.inference = inference
        self.pool = pool
        self.top_k = top_k
        self.encode_batch = encode_batch
        self.stream = TokenStream(Path(pool.config["stream_path"]))
        if self.stream.metadata.get("vocab_size") != inference.vocab_size:
            raise ValueError(
                f"token stream {pool.config['stream_path']} has vocabulary "
                f"{self.stream.metadata.get('vocab_size')}, the model has {inference.vocab_size}")
        self.rows = centered_rows(inference)
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def draw(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Histories (B, H), futures (B, T), header ids (B, K) and targets (B, K).

        The header tokens are the top-k tokens after the future alone, as in the
        tables. The target of pair b on token z is
        <s(history_b + future_b) - center state of future_b, row z>.
        """
        device = self.inference.device
        index = torch.randint(0, len(self.pool.futures), (count,), generator=self.generator)
        futures = self.pool.futures[index]
        history_tokens = self.pool.config["history_tokens"]
        histories = self.stream.windows(
            self.stream.sample_starts(count, history_tokens, self.generator), history_tokens)
        future_hidden = last_hidden(self.inference, futures, self.encode_batch)
        header_ids, _ = header_tokens(
            future_only_logits(self.inference, future_hidden), self.top_k)
        state = last_hidden(
            self.inference, torch.cat([histories, futures], dim=1), self.encode_batch)
        difference = state - self.pool.states[index].to(device)
        target = torch.einsum("be,bke->bk", difference, self.rows[header_ids])
        return histories.to(device), futures.to(device), header_ids, target


def prepare_centered_run(
    *, inference_dir: Path, pool_dir: Path, model_path: Path, device: str, seed: int,
    encode_batch: int,
) -> tuple[CausalLMInference, dict[str, EvalTable], int, CenterPool]:
    """The frozen model, both tables minus the pool's center, top_k and the pool.

    Rejects missing tables, a negative seed and a pool that does not belong to
    the tables and the model.
    """
    if not (inference_dir / "metadata.json").is_file():
        raise ValueError(f"no finished tables in {inference_dir}; run exp4 inference first")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    pool = CenterPool(pool_dir)
    inference = CausalLMInference(model_path, device)
    for key, expected in (("vocab_size", inference.vocab_size),
                          ("hidden_size", inference.model.config.hidden_size)):
        if pool.config[key] != expected:
            raise ValueError(
                f"center pool {key} is {pool.config[key]}, but the model has {expected}")
    if Path(pool.config["model_path"]).name != Path(model_path).name:
        raise ValueError(
            f"center pool was built with {pool.config['model_path']}, not {model_path}")
    baseline = table_baseline(inference, pool)
    tables = {
        name: load_eval_table(inference_dir / name, inference, device, encode_batch, baseline)
        for name in TABLE_NAMES
    }
    reference = tables["val"]
    for key, expected in (("history_tokens", reference.table.histories.shape[1]),
                          ("future_tokens", reference.table.columns.shape[1] - 1)):
        if pool.config[key] != expected:
            raise ValueError(
                f"center pool {key} is {pool.config[key]}, but the tables need {expected}")
    top_k = reference.table.columns.shape[0] // reference.future_hidden.shape[0]
    return inference, tables, top_k, pool
