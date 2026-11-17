"""Study C of exp8: the mode that exp5-exp7 get stuck on.

After g is removed, 65% of the table's energy is one mode: (nearly) the same
number for every history, times the mean of each column over the histories.
exp5-exp7 reach 0.37-0.47 of it on held-out futures, whatever the encoder.
Learning it is a regression: predict, from the future alone, the header-token
logits after the future averaged over all histories that could stand before it.

MeanStudent does only that. It is trained on independent (history, future)
pairs with a squared error, which has the average over the histories as its
minimum, and is compared with both exp4 tables minus g:

    energy   = 1 - ||A - 1 x prediction||^2 / ||A||^2
    ceiling  = the same with the table's own column means as the prediction
    exact    = the exact rank-1 energy (exp7's reference for its first mode)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.exp4_table_neural_svd import TABLE_NAMES
from utils.inference import CausalLMInference

from .grid import Teacher, Windows, step_seed
from experiments.exp7_independent_subnets.model import token_vectors

from .model import GenericMeanStudent, MeanStudent, build_encoder
from .runs import RunFiles, lr_factor
from .spectrum import thin_svd


@dataclass(frozen=True)
class MeanTable:
    future_ids: torch.Tensor  # (F, T)
    header_ids: torch.Tensor  # (F, K)
    g: torch.Tensor  # (F, K)
    column_mean: torch.Tensor  # (F, K): mean over the histories of the table minus g
    histories: int
    total_energy: float  # ||A||^2 of the table minus g
    centered_energy: float  # ||A - 1 x column_mean||^2
    exact_rank1: float
    exact_energy: list[float]  # entry k: exact energy of the leading k + 1 modes, k <= 256


@torch.no_grad()
def load_mean_table(table_dir: Path, teacher: Teacher, device: str | torch.device) -> MeanTable:
    tokens = np.load(table_dir / "tokens.npz")
    future_ids = torch.from_numpy(tokens["futures"]).to(device)
    header_ids = torch.from_numpy(tokens["header_token_ids"]).long().to(device)
    g = teacher.future_only(future_ids, header_ids)
    matrix = torch.from_numpy(np.load(table_dir / "matrix.npy")).to(device).double()
    matrix -= g.reshape(-1).double()
    column_mean = matrix.mean(dim=0)
    total = matrix.square().sum().item()
    centered = (matrix - column_mean).square().sum().item()
    values = thin_svd(matrix)[1].square()
    top = values[0].item()
    cumulative = (values[:257].cumsum(dim=0) / total).tolist()
    return MeanTable(
        future_ids=future_ids, header_ids=header_ids, g=g,
        column_mean=column_mean.float().reshape(g.shape), histories=matrix.shape[0],
        total_energy=total, centered_energy=centered, exact_rank1=top / total,
        exact_energy=cumulative,
    )


@torch.no_grad()
def mean_comparison(student: MeanStudent, table: MeanTable, chunk: int = 256) -> dict:
    student.eval()
    prediction = torch.cat([
        student(table.future_ids[start:start + chunk], table.header_ids[start:start + chunk])
        for start in range(0, len(table.future_ids), chunk)
    ]) - table.g
    student.train()
    miss = (prediction - table.column_mean).double().square().sum().item()
    return {
        "energy": 1 - (table.centered_energy + table.histories * miss) / table.total_energy,
        "ceiling": 1 - table.centered_energy / table.total_energy,
        "exact_rank1": table.exact_rank1,
        # Share of the column means' own variation that the prediction explains.
        "mean_r2": 1 - miss / table.column_mean.double().square().sum().item(),
    }


def train_mean(
    *,
    inference_dir: Path,
    stream_path: Path,
    model_path: Path,
    output_dir: Path,
    device: str,
    prefix_tokens: int,
    batch_size: int,
    steps: int,
    eval_every: int,
    lr: float,
    encoder_lr: float,
    seed: int,
    overwrite: bool,
    generic: str | None = None,
    roberta_path: Path | None = None,
) -> dict:
    """generic names an exp7 encoder kind: the student is then exp7's right sub-network
    (GenericMeanStudent) instead of the copy that keeps the model's unembedding."""
    if not (inference_dir / "metadata.json").is_file():
        raise ValueError(f"no finished tables in {inference_dir}; run exp4 inference first")
    config = {
        "study": "mean", "prefix_tokens": prefix_tokens, "batch_size": batch_size,
        "steps": steps, "eval_every": eval_every, "lr": lr, "encoder_lr": encoder_lr,
        "seed": seed, "model_path": str(model_path), "stream_path": str(stream_path),
    }
    if generic is not None:
        config["generic"] = generic
        config["roberta_path"] = str(roberta_path) if generic == "roberta" else None
    files = RunFiles(output_dir, config, overwrite)
    saved = files.load()

    torch.manual_seed(seed)
    inference = CausalLMInference(model_path, device)
    teacher = Teacher(inference)
    tables = {name: load_mean_table(inference_dir / name, teacher, device) for name in TABLE_NAMES}
    reference = tables["val"]
    top_k = reference.header_ids.shape[1]
    table_tokens = np.load(inference_dir / "val" / "tokens.npz")
    history_tokens = table_tokens["histories"].shape[1]
    future_tokens = reference.future_ids.shape[1]
    if prefix_tokens + future_tokens > inference.max_length:
        raise ValueError(
            f"a prefix of {prefix_tokens} tokens and a future of {future_tokens} do not fit "
            f"the model's {inference.max_length} positions")

    # The prefix starts as one real window of the stream, fixed by the seed.
    if generic is not None:
        student = GenericMeanStudent(
            build_encoder(inference, generic, roberta_path),
            token_vectors(inference.model.get_output_embeddings().weight), teacher,
        ).to(device)
        groups = [
            {"params": list(student.encoder.parameters()),
             "lr": lr if generic == "scratch" else encoder_lr},
            {"params": list(student.head.parameters()), "lr": lr},
        ]
    else:
        prefix_ids = None
        if prefix_tokens:
            prefix_ids = Windows(
                stream_path, prefix_tokens, future_tokens, inference.vocab_size,
                step_seed(seed, 0) + 1,
            ).draw(1, 1)[0][0].to(device)
        student = MeanStudent(inference, prefix_ids).to(device)
        groups = [{"params": list(student.transformer.parameters()), "lr": encoder_lr}]
        if student.prefix is not None:
            groups.append({"params": [student.prefix], "lr": lr})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, steps))
    step = 0
    if saved is not None:
        student.load_state_dict(saved["network"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step = saved["step"]
        print(f"Resuming at step {step}", flush=True)
    files.write_metadata({
        **config, "device": device, "top_k": top_k,
        "target": "logit(token | history + future) - vocabulary mean, of independent pairs; "
                  "the squared error is least at its mean over the histories",
        "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "tables": {
            name: {"ceiling": 1 - table.centered_energy / table.total_energy,
                   "exact_rank1": table.exact_rank1}
            for name, table in tables.items()},
    })
    windows = Windows(
        stream_path, history_tokens, future_tokens, inference.vocab_size, step_seed(seed, step))

    def report(record: dict) -> None:
        cells = "  ".join(
            f"{name}: energy {record[name]['energy']:.4f} (ceiling {record[name]['ceiling']:.4f}, "
            f"exact rank 1 {record[name]['exact_rank1']:.4f}), mean R2 {record[name]['mean_r2']:.4f}"
            for name in TABLE_NAMES)
        print(f"step {record['step']}  loss {record['train_loss']:.5f}  "
              f"{record['seconds_per_step']:.3f} s/step  {cells}", flush=True)

    if step == 0 and not files.history:
        record = {"step": 0, "train_loss": float("nan"), "seconds_per_step": 0.0,
                  **{name: mean_comparison(student, table) for name, table in tables.items()}}
        files.history.append(record)
        report(record)

    student.train()
    losses: list[float] = []
    started = time.perf_counter()
    while step < steps:
        history_ids, future_ids = windows.draw(batch_size, batch_size)
        history_ids, future_ids = history_ids.to(device), future_ids.to(device)
        header_ids, _ = teacher.headers(future_ids, top_k)
        entries = teacher.pair_entries(history_ids, future_ids, header_ids)
        loss = (student(future_ids, header_ids) - entries).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"the loss is not finite at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        losses.append(loss.item())
        if step % eval_every == 0 or step == steps:
            record = {
                "step": step, "train_loss": float(np.mean(losses)),
                "seconds_per_step": (time.perf_counter() - started) / len(losses),
                **{name: mean_comparison(student, table) for name, table in tables.items()},
            }
            files.history.append(record)
            report(record)
            files.save({
                "network": student.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step,
            })
            losses, started = [], time.perf_counter()
    last = files.history[-1] if files.history else {}
    return {"output_dir": str(output_dir), "step": step,
            "val_energy": last.get("val", {}).get("energy")}
