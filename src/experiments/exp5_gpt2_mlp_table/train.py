"""Train exp5 on sample-store pairs and compare it with the exact SVDs.

Every step draws independent (history, future) pairs from the sample store.
The frozen language model gives the hidden state of each history and of each
future, and the future's header tokens with their centered logits g. The
target of a pair is its table entry on those header tokens minus g, so the
fixed first mode 1 x g never enters the loss. At each evaluation the learned
features of both exp4 tables are compared with the exact SVD of that table
minus g.

With unfreeze_right the right side reads the future's tokens through a
trainable copy of the model's transformer. The model itself stays frozen and
still gives g, the header tokens and the history hidden states.
"""

from __future__ import annotations

import json
import time
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

from .features import future_only_logits, header_tokens, last_hidden, pair_targets
from .loss import paired_loss
from .model import build_model, trainable_encoder
from .tables import eval_features, load_eval_table

FIRST_MODE = "1 x centered future-only logit of the header token"
# A fit of the table ends near -E[K^2]. A loss below this multiple of it means the
# network is fitting which history came with which future, which happens when the
# same store pairs are drawn many times.
MEMORIZING_RATIO = 1.2


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


def train_exp5(
    *,
    inference_dir: Path,
    store_dir: Path,
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
    encoder_lr: float = 2e-5,
    encode_batch: int = 1024,
    eval_chunk: int = 2048,
) -> dict:
    """Train on store pairs; every eval_every steps compare both exp4 tables.

    Writes checkpoint.pt, checkpoint-best.pt (the evaluation with the highest
    validation energy at the largest rank), eval_history.jsonl (one line per
    evaluation) and metadata.json to output_dir. An existing checkpoint with
    the same settings is continued up to `steps`; pass overwrite to start
    again. The store changes while producers run, so a continued run draws
    other pairs than an uninterrupted one would.

    unfreeze_right trains a copy of the model's transformer as the right
    side's encoder, with learning rate encoder_lr.
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
    hidden_size = reference.history_hidden.shape[1]
    check_store(
        store_config, model_path,
        history_tokens=reference.table.histories.shape[1],
        future_tokens=reference.table.columns.shape[1] - 1,
        vocab_size=inference.vocab_size,
    )

    config = {
        "inference_dir": str(inference_dir), "store_dir": str(store_dir),
        "model_path": str(model_path), "neigs": neigs, "mlp_width": mlp_width,
        "batch_size": batch_size, "lr": lr, "seed": seed, "optimizer": "Adam",
        "first_mode": FIRST_MODE,
    }
    if unfreeze_right:
        # Added only here, so checkpoints of frozen runs keep matching.
        config.update(unfreeze_right=True, encoder_lr=encoder_lr)
    checkpoint_path = output_dir / "checkpoint.pt"
    best_path = output_dir / "checkpoint-best.pt"
    history_path = output_dir / "eval_history.jsonl"
    if overwrite:
        for path in (checkpoint_path, best_path, history_path):
            path.unlink(missing_ok=True)

    torch.manual_seed(seed)
    model = build_model(
        hidden_size, neigs, top_k, mlp_width,
        right_encoder=trainable_encoder(inference) if unfreeze_right else None,
    ).to(device)
    network = model.model
    groups = [{"params": [*network.left.parameters(), *network.right.parameters()], "lr": lr}]
    if unfreeze_right:
        groups.append({"params": list(network.right_encoder.parameters()), "lr": encoder_lr})
    optimizer = torch.optim.Adam(groups)
    step = 0
    history: list[dict] = []
    best_energy = None
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
    write_atomic(output_dir / "metadata.json", lambda tmp: save_json(tmp, {
        **config,
        "device": device,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "hidden_state": "final layer after ln_f at the last token; "
                        "history and future are encoded separately",
        "right_encoder": "trainable copy of the model's transformer, dropout off"
        if unfreeze_right else "the frozen model",
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
    }))

    # Seeded by the step, so a continued run does not replay the first pairs.
    sampler = SampleSampler(store_dir, seed=sampler_seed(seed, step))
    print(f"Waiting for the first sample block in {store_dir}", flush=True)
    sampler.wait_until_ready()
    print(
        f"Training on {device}: {batch_size} pairs x {top_k} header tokens per step, "
        f"rank {neigs}, MLP width {mlp_width}",
        flush=True,
    )

    loss_sum = energy_sum = 0.0
    window = 0
    started = window_started = time.perf_counter()
    while step < steps:
        model.train()
        histories, futures, pair_logits = draw_pairs(sampler, batch_size)
        history_hidden = last_hidden(inference, histories, encode_batch)
        future_hidden = last_hidden(inference, futures, encode_batch)
        header_ids, g = header_tokens(future_only_logits(inference, future_hidden), top_k)
        target = pair_targets(pair_logits.to(device), header_ids, g)

        optimizer.zero_grad()
        left, right = model(
            history_hidden, futures.to(device) if unfreeze_right else future_hidden
        )
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
            lines = [
                f"step {step}: loss {record['train_loss']:.4f}  "
                f"E[K^2] {record['kernel_energy']:.4f}  "
                f"{record['seconds_per_step']:.3f}s/step  elapsed {now - started:.0f}s"
            ]
            for name, table in tables.items():
                record[name] = compare_with_exact_svd(
                    *eval_features(model, table, eval_chunk), table.table
                )
                lines.append(format_comparison(name, record[name]))
            if record["train_loss"] < -MEMORIZING_RATIO * record["kernel_energy"]:
                record["memorizing"] = True
                lines.append(
                    f"WARNING: loss {record['train_loss']:.4f} is below "
                    f"{-MEMORIZING_RATIO:.1f} x E[K^2]; the network is fitting which history "
                    "came with which future. The store is probably not being refreshed."
                )
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
