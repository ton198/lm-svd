"""Train exp5 on pairs made in the step and compare it with the exact SVDs.

Every step draws independent (history, future) windows from the token stream
and runs the frozen language model on each history + future; no sample store
is read. The frozen model also gives the hidden state of each history and of
each future, and the future's header tokens with their centered logits g. The
target of a pair is its table entry on those header tokens minus g, so the
fixed first mode 1 x g never enters the loss. At each evaluation the learned
features of both exp4 tables are compared with the exact SVD of that table
minus g.

With a center pool (exp7's `center-pool`) the target is instead the table
entry minus the column's mean over the pool's fixed histories; the futures then
come from the pool, and the tables are compared after the same subtraction.

With unfreeze_right the right side reads the future's tokens through a
trainable copy of the model's transformer. The model itself stays frozen and
still gives g, the header tokens and the history hidden states.

With finetune each side is its own trainable copy of the transformer with a
linear head: the left reads the history, the right reads the future followed
by each header token. The frozen model still gives g and the header tokens.

prepare_fresh_run, fresh_metadata and run_training are the parts exp6 shares:
the tables and the source of the pairs, the metadata every run records, and the
loop itself. prepare_run, check_store, draw_pairs and run_metadata serve the
form of exp7 that still reads a sample store.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data_pipline.sample_store import SampleSampler
from data_pipline.sample_store.producer import model_file_identity
from data_pipline.sample_store.store import read_store_config
from experiments.exp4_table_neural_svd import TABLE_NAMES
from experiments.exp4_table_neural_svd.train import (
    RANKS,
    compare_with_exact_svd,
    format_comparison,
)
from utils.common import save_json, write_atomic
from utils.inference import CausalLMInference
from utils.nestedlora import NestedLoRA

from . import LR_SCHEDULES
from .features import last_hidden
from .loss import paired_loss
from .model import build_finetune_model, build_model, trainable_encoder
from .pairs import StreamPairs, open_stream
from .tables import EvalTable, eval_features, load_eval_table

FIRST_MODE = "1 x centered future-only logit of the header token"
# Sequences per encoder pass when a finetuned network is evaluated on the tables.
FINETUNE_EVAL_CHUNK = 256


def sampler_seed(seed: int, step: int) -> int:
    """Seed of the store sampler for a run that starts or resumes at `step`."""
    return int(np.random.SeedSequence([seed, step]).generate_state(1)[0])


def check_store(
    config: dict, model_path: Path, *, history_tokens: int, future_tokens: int, vocab_size: int
) -> None:
    """Reject a store whose pairs do not belong to these tables and this model."""
    for key, expected in (
        ("history_tokens", history_tokens),
        ("future_tokens", future_tokens),
        ("vocab_size", vocab_size),
    ):
        if config[key] != expected:
            raise ValueError(
                f"sample store {key} is {config[key]}, but the tables and model need {expected}"
            )
    # Sizes only: a copied checkpoint keeps them but not its modification times.
    stored, local = (
        {name: entry["size"] for name, entry in files.items()}
        for files in (config["model_files"], model_file_identity(model_path))
    )
    if stored != local:
        raise ValueError(
            f"sample store was produced by other model files than {model_path}: "
            f"store has {stored}, the model directory has {local}"
        )


def draw_pairs(
    sampler: SampleSampler, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Histories (B, H), futures (B, F) and raw logits (B, V) of B store samples."""
    histories, futures, logits = zip(*(sampler.sample() for _ in range(batch_size)))
    return torch.stack(histories), torch.stack(futures), torch.stack(logits)


def _write_history(path: Path, history: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in history), encoding="utf-8")


def prepare_run(
    *,
    inference_dir: Path,
    store_dir: Path,
    model_path: Path,
    device: str,
    seed: int,
    encode_batch: int,
) -> tuple[CausalLMInference, dict[str, EvalTable], int, dict]:
    """The frozen model, both evaluation tables, top_k and the store's settings.

    Rejects missing tables, a negative seed and a store that does not belong to
    the tables and the model.
    """
    if not (inference_dir / "metadata.json").is_file():
        raise ValueError(f"no finished tables in {inference_dir}; run exp4 inference first")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    store_config = read_store_config(Path(store_dir))
    inference = CausalLMInference(model_path, device)
    tables = {
        name: load_eval_table(inference_dir / name, inference, device, encode_batch)
        for name in TABLE_NAMES
    }
    reference = tables["val"]
    top_k = reference.table.columns.shape[0] // reference.future_hidden.shape[0]
    check_store(
        store_config, model_path,
        history_tokens=reference.table.histories.shape[1],
        future_tokens=reference.table.columns.shape[1] - 1,
        vocab_size=inference.vocab_size,
    )
    return inference, tables, top_k, store_config


def run_metadata(
    *, device: str, top_k: int, tables: dict[str, EvalTable], model: NestedLoRA,
    store_config: dict,
) -> dict:
    """What every run records about its target, its pairs, its tables and its size."""
    return {
        "device": device,
        "top_k": top_k,
        "value": "logit(token | history + future) - mean(logits over full vocabulary) "
                 "- centered logit(token | future)",
        "sampling": "independent (history, future) pairs from the sample store; "
                    "every header token of each pair",
        "isolation": "store windows are drawn from the whole training stream and are not "
                     "excluded from the evaluation tables; a pair overlaps a validation "
                     "window with probability of about 0.1%",
        "ranks": list(RANKS),
        "table_shapes": {name: list(table.table.matrix.shape) for name, table in tables.items()},
        "tables": {
            name: {
                "residual_energy": table.residual_energy,
                "residual_top_mode": table.residual_top_mode,
            }
            for name, table in tables.items()
        },
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "store": store_config,
    }


@dataclass(frozen=True)
class PairSource:
    """Where a run's training pairs come from.

    config: what the run's settings record about the source and the target.
    metadata: the entries of run_metadata it replaces or adds.
    make(seed): the pairs, an object whose draw(count) returns histories (B, H),
    futures (B, T), header ids (B, K) and targets (B, K) on the model's device.
    """

    config: dict
    metadata: dict
    make: Callable[[int], object]


def prepare_fresh_run(
    *,
    inference_dir: Path,
    stream_path: Path,
    model_path: Path,
    device: str,
    seed: int,
    encode_batch: int,
    center_pool: Path | None = None,
) -> tuple[CausalLMInference, dict[str, EvalTable], int, PairSource]:
    """The frozen model, both evaluation tables, top_k and the source of the pairs.

    Without a center pool the pairs are fresh windows of the token stream and
    the target is the entry minus g. With one, the futures come from the pool,
    the histories from the pool's stream, and both the target and the tables
    lose the column's mean over the pool's fixed histories; stream_path is then
    unused. Rejects missing tables, a negative seed, a missing stream and a
    stream or pool that does not belong to the model and the tables.
    """
    if center_pool is not None:
        # Imported here: exp7 builds on this module.
        from experiments.exp7_independent_subnets.center import (
            FIRST_MODE as CENTER_FIRST_MODE,
        )
        from experiments.exp7_independent_subnets.center import (
            CenteredPairs,
            prepare_centered_run,
        )

        inference, tables, top_k, pool = prepare_centered_run(
            inference_dir=inference_dir, pool_dir=center_pool, model_path=model_path,
            device=device, seed=seed, encode_batch=encode_batch,
        )
        return inference, tables, top_k, PairSource(
            config={"center_pool": str(center_pool), "first_mode": CENTER_FIRST_MODE},
            metadata={
                "center_pool": pool.config,
                "value": "logit(token | history + future) - mean(logits over full vocabulary) "
                         "- the mean of that value over the fixed histories of the center pool",
                "sampling": "independent pairs of a center-pool future and a fresh history "
                            "window of the token stream, computed by the frozen model in the "
                            "step; every header token of each pair",
                "isolation": "pool futures and stream histories are drawn from the whole "
                             "training stream and are not excluded from the evaluation tables",
            },
            make=lambda pair_seed: CenteredPairs(inference, pool, top_k, pair_seed, encode_batch),
        )

    if not (inference_dir / "metadata.json").is_file():
        raise ValueError(f"no finished tables in {inference_dir}; run exp4 inference first")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    inference = CausalLMInference(model_path, device)
    stream = open_stream(stream_path, inference)
    tables = {
        name: load_eval_table(inference_dir / name, inference, device, encode_batch)
        for name in TABLE_NAMES
    }
    reference = tables["val"]
    top_k = reference.table.columns.shape[0] // reference.future_hidden.shape[0]
    history_tokens = reference.table.histories.shape[1]
    future_tokens = reference.table.columns.shape[1] - 1
    return inference, tables, top_k, PairSource(
        config={"stream_path": str(stream_path), "first_mode": FIRST_MODE},
        metadata={
            "stream": {"path": str(stream_path), "tokens": len(stream), **stream.metadata},
            "sampling": "independent history and future windows of the token stream, drawn "
                        "with replacement and computed by the frozen model in the step; "
                        "every header token of each pair",
            "isolation": "stream windows are drawn from the whole training stream and are "
                         "not excluded from the evaluation tables; a pair overlaps a "
                         "validation window with probability of about 0.1%",
        },
        make=lambda pair_seed: StreamPairs(
            inference, stream_path, history_tokens=history_tokens,
            future_tokens=future_tokens, top_k=top_k, seed=pair_seed,
            encode_batch=encode_batch),
    )


def fresh_metadata(
    *, device: str, top_k: int, tables: dict[str, EvalTable], model: NestedLoRA,
    source: PairSource,
) -> dict:
    """run_metadata of a run whose pairs are made in the step."""
    shared = run_metadata(
        device=device, top_k=top_k, tables=tables, model=model, store_config=None)
    del shared["store"]
    return {**shared, **source.metadata}


def lr_factor(schedule: str, steps: int) -> Callable[[int], float] | None:
    """The multiple of each learning rate at a 0-based step, or None for a constant rate.

    cosine falls from 1 at the first step to 0 after the last of `steps`.
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, not {schedule!r}")
    if schedule == "constant":
        return None
    return lambda step: 0.5 * (1.0 + math.cos(math.pi * step / steps))


def run_training(
    *,
    model: NestedLoRA,
    optimizer: torch.optim.Optimizer,
    forward: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
    features: Callable[[EvalTable], tuple[torch.Tensor, torch.Tensor]],
    config: dict,
    metadata: dict,
    description: str,
    tables: dict[str, EvalTable],
    top_k: int,
    source: PairSource,
    output_dir: Path,
    device: str,
    batch_size: int,
    steps: int,
    eval_every: int,
    seed: int,
    overwrite: bool,
    schedule: Callable[[int], float] | None = None,
) -> dict:
    """Train on the source's pairs; every eval_every steps compare both tables.

    forward(histories, futures, header_ids) returns the left features (B, L)
    and the right features (B * top_k, L) of a batch: the histories (B, H) and
    futures (B, F) are token ids and header_ids (B, top_k) the header tokens of
    each future, all on the model's device. features(table) returns the features
    of every row and column of an evaluation table. schedule(step) is the
    multiple of every group's learning rate at a 0-based step; an evaluation
    then records the rate of the first group as `lr`.

    Writes checkpoint.pt, checkpoint-best.pt, eval_history.jsonl and
    metadata.json (config and metadata) to output_dir, and continues an
    existing checkpoint whose config equals `config`.
    """
    checkpoint_path = output_dir / "checkpoint.pt"
    best_path = output_dir / "checkpoint-best.pt"
    history_path = output_dir / "eval_history.jsonl"
    if overwrite:
        for path in (checkpoint_path, best_path, history_path):
            path.unlink(missing_ok=True)

    step = 0
    history: list[dict] = []
    best_energy = None
    # Before a checkpoint's optimizer state replaces them with scheduled rates.
    base_rates = [group["lr"] for group in optimizer.param_groups]
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError(
                f"checkpoint settings {checkpoint['config']} differ from {config}; "
                "pass --overwrite to start over"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step, history = checkpoint["step"], checkpoint["history"]
        best_energy = checkpoint.get("best_val_energy")
        print(f"Resuming from step {step}", flush=True)
    elif history_path.exists():
        raise FileExistsError(
            f"output already exists: {history_path}; pass --overwrite to replace it"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(
        output_dir / "metadata.json", lambda tmp: save_json(tmp, {**config, **metadata})
    )

    # Seeded by the step, so a continued run does not replay the first pairs.
    pairs = source.make(sampler_seed(seed, step))
    print(
        f"Training on {device}: {batch_size} pairs x {top_k} header tokens per step, "
        f"{description}",
        flush=True,
    )

    loss_sum = energy_sum = 0.0
    window = 0
    started = window_started = time.perf_counter()
    while step < steps:
        model.train()
        histories, futures, header_ids, target = pairs.draw(batch_size)
        if schedule is not None:
            for group, rate in zip(optimizer.param_groups, base_rates):
                group["lr"] = rate * schedule(step)

        optimizer.zero_grad()
        left, right = forward(histories, futures, header_ids)
        loss = paired_loss(model, left, right, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        step += 1
        loss_sum += loss.item()
        energy_sum += target.square().mean().item()
        window += 1

        if step % eval_every == 0 or step == steps:
            now = time.perf_counter()
            # A perfect fit of every mode brings the loss to about -E[K^2].
            record = {
                "step": step,
                "train_loss": loss_sum / window,
                "kernel_energy": energy_sum / window,
                "seconds_per_step": (now - window_started) / window,
            }
            rate = ""
            if schedule is not None:
                record["lr"] = optimizer.param_groups[0]["lr"]
                rate = f"  lr {record['lr']:.2e}"
            lines = [
                f"step {step}: loss {record['train_loss']:.4f}  "
                f"E[K^2] {record['kernel_energy']:.4f}{rate}  "
                f"{record['seconds_per_step']:.3f}s/step  elapsed {now - started:.0f}s"
            ]
            for name, table in tables.items():
                record[name] = compare_with_exact_svd(*features(table), table.table)
                lines.append(format_comparison(name, record[name]))
            print("\n".join(lines), flush=True)
            history.append(record)
            loss_sum = energy_sum = 0.0
            window = 0
            val_energy = record["val"]["captured_energy"][-1]
            if best_energy is None or val_energy > best_energy:
                best_energy = val_energy
                write_atomic(best_path, lambda tmp: torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "val_energy": val_energy,
                    "config": config,
                }, tmp))
            write_atomic(checkpoint_path, lambda tmp: torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "history": history,
                "config": config,
                "best_val_energy": best_energy,
            }, tmp))
            write_atomic(history_path, lambda tmp: _write_history(tmp, history))
            window_started = time.perf_counter()
    return {"output_dir": str(output_dir), "step": step, "evaluations": len(history)}


def train_exp5(
    *,
    inference_dir: Path,
    stream_path: Path,
    model_path: Path,
    output_dir: Path,
    device: str,
    neigs: int,
    mlp_width: int,
    batch_size: int,
    steps: int,
    eval_every: int,
    lr: float,
    seed: int,
    overwrite: bool,
    unfreeze_right: bool = False,
    finetune: bool = False,
    encoder_lr: float = 2e-5,
    center_pool: Path | None = None,
    lr_schedule: str = "constant",
    encode_batch: int = 1024,
    eval_chunk: int = 2048,
) -> dict:
    """Train on pairs made in the step; every eval_every steps compare both exp4 tables.

    Writes checkpoint.pt, checkpoint-best.pt (the evaluation with the highest
    validation energy at the largest rank), eval_history.jsonl (one line per
    evaluation) and metadata.json to output_dir. An existing checkpoint with
    the same settings is continued up to `steps`; pass overwrite to start
    again. A continued run draws other pairs than an uninterrupted one would.

    The pairs are fresh windows of the token stream at stream_path with the
    entry minus g as the target, or with center_pool the pairs and the target
    of that pool (see prepare_fresh_run). lr_schedule "cosine" lowers every
    learning rate from its value to 0 over `steps`, which then belong to the
    run's settings.

    unfreeze_right trains a copy of the model's transformer as the right
    side's encoder, with learning rate encoder_lr. finetune trains one copy per
    side, each with a linear head; its encoders also use encoder_lr.
    """
    if finetune and unfreeze_right:
        raise ValueError(
            "finetune already trains the right side's encoder; pass finetune or "
            "unfreeze_right, not both"
        )
    schedule = lr_factor(lr_schedule, steps)
    inference, tables, top_k, source = prepare_fresh_run(
        inference_dir=inference_dir, stream_path=stream_path, model_path=model_path,
        device=device, seed=seed, encode_batch=encode_batch, center_pool=center_pool,
    )
    hidden_size = tables["val"].history_hidden.shape[1]

    config = {
        "inference_dir": str(inference_dir), "model_path": str(model_path),
        "neigs": neigs, "mlp_width": mlp_width, "batch_size": batch_size, "lr": lr,
        "seed": seed, "optimizer": "Adam", **source.config,
    }
    if schedule is not None:
        # The rate of a step depends on the length of the run.
        config.update(lr_schedule=lr_schedule, steps=steps)
    if unfreeze_right:
        # Added only here, so checkpoints of frozen runs keep matching.
        config.update(unfreeze_right=True, encoder_lr=encoder_lr)
    if finetune:
        del config["mlp_width"]  # linear heads
        config.update(finetune=True, encoder_lr=encoder_lr)

    torch.manual_seed(seed)
    if finetune:
        model = build_finetune_model(inference, neigs).to(device)
        network = model.model
        groups = [
            {"params": [*network.left_head.parameters(), *network.right_head.parameters()],
             "lr": lr},
            {"params": [*network.left_encoder.parameters(),
                        *network.right_encoder.parameters()], "lr": encoder_lr},
        ]

        def forward(histories, futures, header_ids):
            return model(histories, futures, header_ids)
    else:
        model = build_model(
            hidden_size, neigs, top_k, mlp_width,
            right_encoder=trainable_encoder(inference) if unfreeze_right else None,
        ).to(device)
        network = model.model
        groups = [
            {"params": [*network.left.parameters(), *network.right.parameters()], "lr": lr}
        ]
        if unfreeze_right:
            groups.append({"params": list(network.right_encoder.parameters()), "lr": encoder_lr})

        def forward(histories, futures, header_ids):
            return model(
                last_hidden(inference, histories, encode_batch),
                futures if unfreeze_right else last_hidden(inference, futures, encode_batch),
            )

    chunk = FINETUNE_EVAL_CHUNK if finetune else eval_chunk
    return run_training(
        model=model,
        optimizer=torch.optim.Adam(groups),
        forward=forward,
        features=lambda table: eval_features(model, table, chunk),
        config=config,
        metadata={
            "hidden_size": hidden_size,
            "hidden_state": "final layer after ln_f at the last token; "
                            "history and future are encoded separately",
            **({
                "left_encoder": "trainable copy of the model's transformer, dropout off, "
                                "reading the history; LayerNorm + linear head",
                "right_encoder": "trainable copy of the model's transformer, dropout off, "
                                 "reading the future followed by each header token in one "
                                 "sequence (each header token sees the future and itself, at "
                                 "position |future|); LayerNorm + linear head",
            } if finetune else {
                "right_encoder": "trainable copy of the model's transformer, dropout off"
                if unfreeze_right else "the frozen model",
            }),
            **fresh_metadata(
                device=device, top_k=top_k, tables=tables, model=model, source=source),
        },
        description=f"rank {neigs}, "
                    + ("linear heads" if finetune else f"MLP width {mlp_width}")
                    + ("" if schedule is None else f", {lr_schedule} learning rate"),
        tables=tables, top_k=top_k, source=source, output_dir=output_dir, device=device,
        batch_size=batch_size, steps=steps, eval_every=eval_every, seed=seed,
        overwrite=overwrite, schedule=schedule,
    )
