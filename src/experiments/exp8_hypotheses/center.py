"""Study B of exp8: NeuralSVD on the table minus its column means.

After g is removed (exp5-exp7), two thirds of what is left is still one mode
whose left function is nearly constant over the histories: the mean of each
column over the histories. Study A shows that the modes behind it are real (they
carry over to other histories and futures), but exp5-exp7 never reach them,
because every network first has to predict that mean from the future alone.

Here the mean never enters the target. Each step runs the frozen model on a
grid of histories x futures and subtracts from every column its mean over the
batch's histories; the left features lose their batch mean in the same way.
A constant left function is then worth exactly nothing, and g drops out as well
(it is constant over the histories).

Evaluation is on the two exp4 tables minus their own column means, against the
exact SVD of that matrix; the left features lose their mean over the table's
histories.

Study D is the same training with --layout landmark: the right functions are
mixes of the frozen model's entries behind fixed landmark histories
(model.LandmarkNetwork), so only the left side has to generalise. Its runs also
report minus_g_energy, the energy on the table minus g (exp5-exp7's measure) of

    1 x (mean of the landmark rows - g)  +  the learned centered modes,

next to the exact energy at one rank more (the mean counts as a mode).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from experiments.exp4_table_neural_svd import TABLE_NAMES
from experiments.exp4_table_neural_svd.train import (
    RANKS,
    Table,
    compare_with_exact_svd,
    format_comparison,
    load_table,
)
from utils.inference import CausalLMInference
from utils.nestedlora import NestedLoRA

from .grid import Teacher, Windows, step_seed
from .mean import load_mean_table
from .model import ENCODERS, LAYOUTS, build_network, parameter_groups
from .runs import RunFiles, lr_factor

NESTINGS = ("joint", "sequential")


def centered_loss(
    wrapper: NestedLoRA, left: torch.Tensor, right: torch.Tensor, entries: torch.Tensor
) -> torch.Tensor:
    """NestedLoRA loss of left (N, L), right (F, K, L) on a grid of entries (N, F, K).

    Entries and left features lose their mean over the N histories. Both terms of
    the loss then estimate the centered table's loss times (N - 1) / N.
    """
    kernel = entries - entries.mean(dim=0, keepdim=True)
    left = left - left.mean(dim=0, keepdim=True)
    return wrapper.compute_loss_kernel(left, right.flatten(0, 1), kernel.flatten(1))[0]


def load_centered_table(
    table_dir: Path, device: str | torch.device
) -> tuple[Table, torch.Tensor, torch.Tensor]:
    """An exp4 table minus its column means with its exact SVD, its futures and header ids."""
    table = load_table(table_dir, device, "column-mean")
    tokens = np.load(table_dir / "tokens.npz")
    return (
        table,
        torch.from_numpy(tokens["futures"]).to(device),
        torch.from_numpy(tokens["header_token_ids"]).long().to(device),
    )


@torch.no_grad()
def table_features(
    network: torch.nn.Module, table: Table, future_ids: torch.Tensor,
    header_ids: torch.Tensor, chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Features of every row (N, L), minus their mean, and of every column (C, L); float64."""
    network.eval()
    left = torch.cat([
        network.left_features(table.histories[start:start + chunk]).double()
        for start in range(0, len(table.histories), chunk)
    ])
    right = torch.cat([
        network.right_features(
            future_ids[start:start + chunk], header_ids[start:start + chunk]
        ).double().flatten(0, 1)
        for start in range(0, len(future_ids), chunk)
    ])
    network.train()
    return left - left.mean(dim=0, keepdim=True), right


@torch.no_grad()
def span_energy(
    left: torch.Tensor, right: torch.Tensor, matrix: torch.Tensor, ranks: list[int],
    chunk: int = 256,
) -> dict[str, list[float]]:
    """Energy within reach of each side's leading features, with the other side ideal.

    left_span_energy at rank k is ||Q_k^T A||^2 / ||A||^2 with Q_k an orthonormal
    basis of the first k left features: what those features capture with the best
    possible right functions. right_span_energy is the same for the right
    features. They separate a side that learned nothing from one whose partner
    failed.
    """
    left_basis, right_basis = torch.linalg.qr(left).Q, torch.linalg.qr(right).Q
    total = left.new_zeros(())
    # A table with fewer rows than features gives a basis with fewer columns.
    by_left = left.new_zeros(left_basis.shape[1])
    by_right = left.new_zeros(right_basis.shape[1])
    for start in range(0, matrix.shape[0], chunk):
        block = matrix[start:start + chunk].double()
        total += block.square().sum()
        by_right += (block @ right_basis).square().sum(dim=0)
    for start in range(0, matrix.shape[1], 8192):
        by_left += (left_basis.T @ matrix[:, start:start + 8192].double()).square().sum(dim=1)
    return {
        "left_span_energy": [(by_left[:rank].sum() / total).item() for rank in ranks],
        "right_span_energy": [(by_right[:rank].sum() / total).item() for rank in ranks],
    }


@torch.no_grad()
def landmark_mean(network: torch.nn.Module, table, chunk: int = 256) -> dict:
    """The landmark rows' mean as the first mode of a table minus g (a mean.MeanTable).

    mean_energy: 1 - ||A - 1 x (landmark mean - g)||^2 / ||A||^2; ceiling: the same
    with the table's own column means; centered_share: ||A - column means||^2 /
    ||A||^2, the weight of the centered modes in the table minus g; exact[k]: the
    exact rank-(k + 1) energy of the table minus g.
    """
    network.eval()
    estimate = torch.cat([
        network.entries(
            table.future_ids[start:start + chunk], table.header_ids[start:start + chunk]
        ).mean(dim=0)
        for start in range(0, len(table.future_ids), chunk)
    ]) - table.g
    network.train()
    miss = (estimate - table.column_mean).double().square().sum().item()
    return {
        "mean_energy": 1 - (table.centered_energy + table.histories * miss) / table.total_energy,
        "ceiling": 1 - table.centered_energy / table.total_energy,
        "centered_share": table.centered_energy / table.total_energy,
        "exact": table.exact_energy,
    }


def format_spans(comparison: dict) -> str:
    cells = ", ".join(
        f"{rank}: {left:.4f}/{right:.4f}"
        for rank, left, right in zip(
            comparison["ranks"], comparison["left_span_energy"],
            comparison["right_span_energy"]))
    return f"        span energy left/right by rank  {cells}"


def train_centered(
    *,
    inference_dir: Path,
    stream_path: Path,
    model_path: Path,
    output_dir: Path,
    device: str,
    layout: str,
    encoder: str,
    nesting: str,
    modes: int,
    history_batch: int,
    future_batch: int,
    steps: int,
    eval_every: int,
    lr: float,
    encoder_lr: float,
    seed: int,
    overwrite: bool,
    roberta_path: Path | None = None,
    landmarks: int = 64,
    eval_chunk: int = 256,
) -> dict:
    if layout not in LAYOUTS or encoder not in ENCODERS or nesting not in NESTINGS:
        raise ValueError(f"unknown layout {layout!r}, encoder {encoder!r} or nesting {nesting!r}")
    if history_batch < 2:
        raise ValueError("history_batch must be at least 2: a column's mean needs two histories")
    if not (inference_dir / "metadata.json").is_file():
        raise ValueError(f"no finished tables in {inference_dir}; run exp4 inference first")
    config = {
        "study": "center", "layout": layout, "encoder": encoder, "nesting": nesting,
        "modes": modes, "history_batch": history_batch, "future_batch": future_batch,
        "steps": steps, "eval_every": eval_every, "lr": lr, "encoder_lr": encoder_lr,
        "seed": seed, "model_path": str(model_path), "stream_path": str(stream_path),
        "roberta_path": str(roberta_path) if encoder == "roberta" else None,
        "landmarks": landmarks if layout == "landmark" else None,
    }
    files = RunFiles(output_dir, config, overwrite)
    saved = files.load()

    torch.manual_seed(seed)
    inference = CausalLMInference(model_path, device)
    teacher = Teacher(inference)
    tables = {name: load_centered_table(inference_dir / name, device) for name in TABLE_NAMES}
    reference, _, reference_headers = tables["val"]
    top_k = reference_headers.shape[1]
    history_tokens = reference.histories.shape[1]
    future_tokens = reference.columns.shape[1] - 1
    ranks = tuple(rank for rank in RANKS if rank <= modes)
    if modes not in ranks:
        ranks = ranks + (modes,)

    landmark_ids = None
    if layout == "landmark":
        # Fixed by the seed alone, so a resumed run has the same landmarks.
        landmark_ids = Windows(
            stream_path, history_tokens, future_tokens, inference.vocab_size,
            step_seed(seed, 0) + 1,
        ).draw(landmarks, 1)[0].to(device)
    network = build_network(
        inference, layout=layout, encoder=encoder, modes=modes, roberta_path=roberta_path,
        teacher=teacher, landmark_ids=landmark_ids,
    ).to(device)
    wrapper = NestedLoRA(model=network, neigs=modes, sequential=nesting == "sequential").to(device)
    optimizer = torch.optim.Adam(parameter_groups(network, encoder, lr, encoder_lr))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, steps))
    step = 0
    if saved is not None:
        network.load_state_dict(saved["network"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step = saved["step"]
        print(f"Resuming at step {step}", flush=True)
    files.write_metadata({
        **config, "device": device, "top_k": top_k,
        "target": "logit(token | history + future) - vocabulary mean, minus the mean of that "
                  "over the histories of the batch (training) or of the table (evaluation)",
        "sampling": "a fresh grid of history_batch x future_batch stream windows per step, "
                    "computed by the frozen model",
        "ranks": list(ranks),
        "table_shapes": {name: list(table.matrix.shape) for name, (table, _, _) in tables.items()},
        "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
    })
    windows = Windows(
        stream_path, history_tokens, future_tokens, inference.vocab_size, step_seed(seed, step))

    # Study D: how the landmark mean and the centered modes add up on the table minus g.
    minus_g: dict[str, dict] = {}
    if layout == "landmark":
        for name, (table, future_ids, header_ids) in tables.items():
            reference_table = load_mean_table(inference_dir / name, teacher, device)
            minus_g[name] = landmark_mean(network, reference_table, eval_chunk)
            print(
                f"{name}: the mean of {landmarks} landmark rows as the first mode captures "
                f"{minus_g[name]['mean_energy']:.4f} of the table minus g "
                f"(its own column means {minus_g[name]['ceiling']:.4f}, "
                f"exact rank 1 {minus_g[name]['exact'][0]:.4f})", flush=True)

    def evaluate() -> dict:
        record = {}
        for name, (table, future_ids, header_ids) in tables.items():
            left, right = table_features(network, table, future_ids, header_ids, eval_chunk)
            comparison = compare_with_exact_svd(left, right, table, ranks)
            comparison.update(span_energy(left, right, table.matrix, comparison["ranks"]))
            if name in minus_g:
                parts = minus_g[name]
                comparison["minus_g_energy"] = [
                    parts["mean_energy"] + parts["centered_share"] * energy
                    for energy in comparison["captured_energy"]]
                comparison["minus_g_exact_energy"] = [
                    parts["exact"][min(rank, len(parts["exact"]) - 1)]
                    for rank in comparison["ranks"]]
            record[name] = comparison
        return record

    network.train()
    losses: list[float] = []
    kernel_energy: list[float] = []
    started = time.perf_counter()
    while step < steps:
        history_ids, future_ids = windows.draw(history_batch, future_batch)
        history_ids, future_ids = history_ids.to(device), future_ids.to(device)
        header_ids, _ = teacher.headers(future_ids, top_k)
        entries = teacher.grid_entries(history_ids, future_ids, header_ids)
        loss = centered_loss(
            wrapper, network.left_features(history_ids),
            network.right_features(future_ids, header_ids), entries)
        if not torch.isfinite(loss):
            raise RuntimeError(f"the loss is not finite at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        losses.append(loss.item())
        kernel_energy.append((entries - entries.mean(dim=0)).square().mean().item())
        if step % eval_every == 0 or step == steps:
            seconds = (time.perf_counter() - started) / len(losses)
            record = {
                "step": step, "train_loss": float(np.mean(losses)),
                "kernel_energy": float(np.mean(kernel_energy)), "seconds_per_step": seconds,
                **evaluate(),
            }
            files.history.append(record)
            print(
                f"step {step}  loss {record['train_loss']:+.5f}  "
                f"E[K^2] {record['kernel_energy']:.5f}  {seconds:.3f} s/step", flush=True)
            for name in TABLE_NAMES:
                print(format_comparison(name, record[name]), flush=True)
                print(format_spans(record[name]), flush=True)
                if "minus_g_energy" in record[name]:
                    cells = ", ".join(
                        f"{rank}+1: {energy:.4f}/{exact:.4f}" for rank, energy, exact in zip(
                            record[name]["ranks"], record[name]["minus_g_energy"],
                            record[name]["minus_g_exact_energy"]))
                    print(f"        table minus g, mean + modes: energy/exact  {cells}", flush=True)
            files.save({
                "network": network.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step,
            })
            losses, kernel_energy, started = [], [], time.perf_counter()
    last = files.history[-1] if files.history else {}
    return {
        "output_dir": str(output_dir), "step": step,
        "val_energy": last.get("val", {}).get("captured_energy"),
    }
