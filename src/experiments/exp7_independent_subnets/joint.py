"""Train all modes of exp7 at once: no stage, no frozen mode, no stopped gradient.

The modes are the staged run's (model.py): one pair of sub-networks per mode and
no trainable parameter shared between two modes, mode l seeded as stage l seeds
it. Every step updates all of them under joint nesting (stage.joint_loss), the
mean over the ranks k = 1..L of the rank-k loss of the first k modes. This is
exp5's loss on exp7's network; in the staged run mode l instead sees the earlier
modes only as a fixed residual and the later ones never.

The pairs are made in the step as in exp5 (prepare_fresh_run): the entry minus g
on fresh windows of the token stream, or with a center pool the entry minus the
column's mean over the pool's fixed histories. Every evaluation compares all L
modes with the exact SVD of both exp4 tables at ranks 1..L, under the staged
run's names (selection, report). Nothing is selected: the last weights are the
result.
"""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import torch

from experiments.exp4_table_neural_svd.train import (
    compare_with_exact_svd,
    format_comparison,
)
from experiments.exp5_gpt2_mlp_table.train import (
    fresh_metadata,
    prepare_fresh_run,
    sampler_seed,
)
from utils.common import save_json, write_atomic

from . import PRETRAINED, SUBNETS, TABLE_KEYS, WARMUP_STEPS
from .model import check_lengths, token_vectors
from .stage import joint_loss, nesting_weights
from .train import (
    ENCODERS,
    INPUT_SPREAD_FLOOR,
    MODE_SEED_OFFSET,
    _build_mode,
    _mode_path,
    _write_lines,
    input_spread,
    mode_table_features,
    warmup_factor,
)


def joint_outputs(
    modes: torch.nn.ModuleList,
    vectors: torch.Tensor,
    histories: torch.Tensor,
    futures: torch.Tensor,
    header_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outputs of all modes with gradient: left (B, L) and right (B, K, L)."""
    left = torch.stack([mode.left_features(histories) for mode in modes], dim=1)
    right = torch.stack(
        [mode.right_features(futures, header_ids, vectors) for mode in modes], dim=2)
    return left, right


def train_exp7_joint(
    *,
    inference_dir: Path,
    stream_path: Path,
    model_path: Path,
    output_dir: Path,
    device: str,
    subnet: str,
    modes: int,
    steps: int,
    scratch_layers: int,
    scratch_width: int,
    scratch_heads: int,
    batch_size: int,
    eval_every: int,
    lr: float,
    encoder_lr: float,
    seed: int,
    overwrite: bool,
    roberta_path: Path,
    center_pool: Path | None = None,
    encode_batch: int = 1024,
    eval_chunk: int = 256,
) -> dict:
    """Train `modes` modes together for `steps` steps on pairs made in the step.

    Learning rates are the staged run's: a pretrained kind trains its encoders
    with encoder_lr and its heads with lr, scratch modes train everything with
    lr after a linear warmup over the first WARMUP_STEPS steps of the run.

    Writes eval_history.jsonl (one line per evaluation, every eval_every steps
    and at the last step), checkpoint.pt and metadata.json to output_dir, and
    modes/mode-XX.pt when the run reaches `steps`. A checkpoint.pt of the same
    settings is continued up to `steps`, which may be larger than before; the
    number of modes is a setting. overwrite deletes the outputs and starts again.
    """
    if subnet not in SUBNETS:
        raise ValueError(f"subnet must be one of {SUBNETS}, got {subnet!r}")
    for name, value in (("modes", modes), ("steps", steps), ("eval_every", eval_every),
                        ("batch_size", batch_size)):
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    if subnet == "scratch" and scratch_width % scratch_heads:
        raise ValueError(
            f"scratch_width must be divisible by scratch_heads, got {scratch_width} "
            f"and {scratch_heads}")

    inference, tables, top_k, source = prepare_fresh_run(
        inference_dir=inference_dir, stream_path=stream_path, model_path=model_path,
        device=device, seed=seed, encode_batch=encode_batch, center_pool=center_pool,
    )
    config = {
        "training": "joint", "inference_dir": str(inference_dir),
        "model_path": str(model_path), "roberta_path": str(roberta_path), "subnet": subnet,
        "modes": modes, "scratch_layers": scratch_layers, "scratch_width": scratch_width,
        "scratch_heads": scratch_heads, "batch_size": batch_size, "eval_every": eval_every,
        "lr": lr, "encoder_lr": encoder_lr, "seed": seed, "optimizer": "Adam",
        **source.config,
    }
    selection_name = next(name for name, key in TABLE_KEYS.items() if key == "selection")
    selection_table = tables[selection_name]
    exact = selection_table.table
    exact_modes = min(exact.left_vectors.shape[1], exact.right_vectors.shape[1])
    if modes > exact_modes:
        raise ValueError(
            f"--modes {modes} exceeds the {exact_modes} exact singular vectors of the "
            f"selection table (exp4 {selection_name!r} in {inference_dir})")

    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.pt"
    history_path = output_dir / "eval_history.jsonl"
    checkpoint = None
    if not overwrite and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        saved = checkpoint["config"]
        if saved != config:
            changed = {
                key: (saved.get(key), config.get(key))
                for key in sorted(set(saved) | set(config)) if saved.get(key) != config.get(key)}
            raise ValueError(
                f"settings differ from the checkpoint in {output_dir} (saved, now): {changed}; "
                "pass --overwrite to start over")
    elif not overwrite and history_path.exists():
        raise FileExistsError(
            f"output already exists: {history_path}; pass --overwrite to replace it")

    def build_all() -> torch.nn.ModuleList:
        built = []
        for number in range(1, modes + 1):
            # The seed of stage `number` of the staged run: the same first weights.
            torch.manual_seed(sampler_seed(seed, MODE_SEED_OFFSET + number))
            built.append(_build_mode(inference, config))
        return torch.nn.ModuleList(built)

    network = build_all()
    if subnet == "roberta":
        check_lengths(
            network[0], history_tokens=exact.histories.shape[1],
            future_tokens=exact.columns.shape[1] - 1)
    if overwrite:
        for name in ("checkpoint.pt", "eval_history.jsonl"):
            (output_dir / name).unlink(missing_ok=True)
        shutil.rmtree(output_dir / "modes", ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    network.to(device)
    if subnet in PRETRAINED:
        optimizer = torch.optim.Adam([
            {"params": [p for mode in network
                        for p in (*mode.left_head.parameters(), *mode.right_head.parameters())],
             "lr": lr},
            {"params": [p for mode in network for p in (
                *mode.left_encoder.parameters(), *mode.right_encoder.parameters())],
             "lr": encoder_lr},
        ])
        scheduler = None
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_factor)
    step = 0
    history: list[dict] = []
    if checkpoint is not None:
        network.load_state_dict(checkpoint["modes_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step, history = checkpoint["step"], checkpoint["history"]
        del checkpoint
        print(f"Resuming from step {step}", flush=True)
    # Rewritten from the checkpoint: a run killed between the two files left it short.
    write_atomic(history_path, lambda tmp: _write_lines(tmp, history))

    shared = fresh_metadata(
        device=device, top_k=top_k, tables=tables, model=network[0], source=source)
    shared["ranks"] = f"1..{modes}"
    shared["table_shapes"] = {
        TABLE_KEYS[name]: list(table.table.matrix.shape) for name, table in tables.items()}
    shared["tables"] = {
        TABLE_KEYS[name]: {"exp4_table": name, "role": "reported only", **values}
        for name, values in shared["tables"].items()}
    write_atomic(output_dir / "metadata.json", lambda tmp: save_json(tmp, {
        **config,
        "warmup_steps": 0 if subnet in PRETRAINED else WARMUP_STEPS,
        "encoder": ENCODERS[subnet],
        "loss": "joint nesting: the mean over the ranks k = 1..modes of the rank-k loss of "
                "the first k modes; all modes are trained in every step and none is detached",
        "trainable_parameters_of": "one mode; no two modes share a trainable parameter",
        **shared,
    }))

    vectors = token_vectors(inference.model.get_output_embeddings().weight).to(device)
    weights, pair_weights = (x.to(device) for x in nesting_weights(modes))
    # Seeded by the step the run starts from, so a continued run does not replay pairs.
    pairs = source.make(sampler_seed(seed, step))
    rates = (f"heads lr {lr}, encoders lr {encoder_lr}" if subnet in PRETRAINED else
             f"lr {lr} with a linear warmup over the first {WARMUP_STEPS} steps")
    print(
        f"Training {modes} modes together on {device}: {batch_size} pairs x {top_k} header "
        f"tokens per step, {steps} steps, {subnet} sub-networks, {rates}",
        flush=True,
    )

    loss_sum = kernel_sum = 0.0
    window = 0
    started = window_started = time.perf_counter()
    while step < steps:
        histories, futures, header_ids, target = pairs.draw(batch_size)
        network.train()
        optimizer.zero_grad()
        loss = joint_loss(
            *joint_outputs(network, vectors, histories, futures, header_ids), target,
            weights, pair_weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss {loss.item()} at step {step + 1}")
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step += 1
        loss_sum += loss.item()
        kernel_sum += target.square().mean().item()
        window += 1
        if step % eval_every and step != steps:
            continue

        now = time.perf_counter()
        record = {
            "step": step,
            "train_loss": loss_sum / window,
            "kernel_energy": kernel_sum / window,
            "seconds_per_step": (now - window_started) / window,
        }
        for name, table in tables.items():
            features = [
                mode_table_features(mode, table, vectors, eval_chunk) for mode in network]
            record[TABLE_KEYS[name]] = compare_with_exact_svd(
                torch.stack([rows for rows, _ in features], dim=1),
                torch.stack([columns for _, columns in features], dim=1),
                table.table, ranks=tuple(range(1, modes + 1)))
        if not math.isfinite(record["selection"]["captured_energy"][-1]):
            raise RuntimeError(f"non-finite captured energy at step {step}")
        lines = [
            f"step {step}: loss {record['train_loss']:.4f}  "
            f"E[K^2] {record['kernel_energy']:.4f}  "
            f"{record['seconds_per_step']:.3f}s/step  elapsed {now - started:.0f}s"
        ]
        lines.extend(format_comparison(key, record[key]) for key in TABLE_KEYS.values())
        for number, mode in enumerate(network, start=1):
            lines.extend(
                (f"WARNING: the {side} encoder of mode {number} gives every sequence the "
                 f"same vector (spread {value:.1e}); it no longer reads its input. "
                 "A lower --lr avoids this.")
                for side, value in zip(("left", "right"), input_spread(mode, selection_table))
                if value < INPUT_SPREAD_FLOOR)
        print("\n".join(lines), flush=True)
        history.append(record)
        # The checkpoint first: eval_history.jsonl never holds a record it does not.
        write_atomic(checkpoint_path, lambda tmp: torch.save({
            "config": config,
            "step": step,
            "history": history,
            "modes_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        }, tmp))
        write_atomic(history_path, lambda tmp: _write_lines(tmp, history))
        loss_sum = kernel_sum = 0.0
        window = 0
        window_started = time.perf_counter()

    (output_dir / "modes").mkdir(exist_ok=True)
    for number, mode in enumerate(network, start=1):
        state = {key: value.detach().cpu() for key, value in mode.state_dict().items()}
        saved = {"state_dict": state, "mode": number, "step": step}
        write_atomic(_mode_path(output_dir, number), lambda tmp, saved=saved: torch.save(saved, tmp))
    return {
        "output_dir": str(output_dir), "step": step, "modes": modes,
        "evaluations": len(history),
    }
