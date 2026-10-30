from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.exp11_stream_student.features import build_student
from utils.common import save_json, save_numpy, write_atomic
from utils.inference import CausalLMInference
from utils.neural_svd import cosine_lr
from utils.stream_neural_svd import TokenStream, centered_unembedding


def distill_student(
    *,
    inference: CausalLMInference,
    stream: TokenStream,
    stream_path: Path,
    model_path: Path,
    student_config: dict,
    device: str,
    batch: int,
    steps: int,
    lr: float,
    min_lr: float,
    probe_windows: int,
    probe_seed: int,
    eval_every: int,
    checkpoint_every: int,
    log_every: int,
    seed: int,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, int | float | str]:
    outputs = [
        output_dir / "student.pt",
        output_dir / "loss_history.npy",
        output_dir / "metadata.json",
    ]
    checkpoint_path = output_dir / "checkpoint.pt"
    window = student_config["max_positions"]
    probe_generator = torch.Generator().manual_seed(probe_seed)
    probe_starts = stream.sample_starts(probe_windows, window, probe_generator)
    probe_ids = stream.windows(probe_starts, window)

    if overwrite:
        for path in [*outputs, checkpoint_path]:
            path.unlink(missing_ok=True)
    elif not checkpoint_path.exists():
        for path in outputs:
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )

    inference.model.eval()
    token_matrix = centered_unembedding(inference)
    torch.manual_seed(seed)
    student = build_student(student_config, token_matrix).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    config = {
        "student": student_config,
        "stream_path": str(stream_path),
        "model_path": str(model_path),
        "stream_tokens": len(stream),
        "batch": batch,
        "steps": steps,
        "lr": lr,
        "min_lr": min_lr,
        "optimizer": "Adam",
        "seed": seed,
        "probe_windows": probe_windows,
        "probe_seed": probe_seed,
        "eval_every": eval_every,
    }
    step = 0
    loss_history: list[float] = []
    probe_history: list[dict] = []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        saved_config = checkpoint["config"]
        if saved_config != config:
            diffs = ", ".join(
                f"{key}: {saved_config.get(key)!r} != {config.get(key)!r}"
                for key in sorted(saved_config.keys() | config.keys())
                if saved_config.get(key) != config.get(key)
            )
            raise ValueError(
                f"checkpoint config mismatch ({diffs}); pass --overwrite to start over"
            )
        student.load_state_dict(checkpoint["student"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["generator"])
        step = checkpoint["step"]
        loss_history = checkpoint["loss_history"].tolist()
        probe_history = checkpoint["probe_history"]
        print(f"resuming from step {step}", flush=True)
        del checkpoint

    def log_probabilities(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            teacher_lp = F.log_softmax(
                inference.final_hidden_states(ids) @ token_matrix.T, dim=-1
            )
        student_lp = F.log_softmax(
            student.predicted_hidden(student(ids)) @ token_matrix.T, dim=-1
        )
        return teacher_lp, student_lp

    def evaluate_probe() -> dict:
        all_sum = 0.0
        last_sum = 0.0
        with torch.no_grad():
            for start in range(0, probe_windows, batch):
                ids = probe_ids[start : start + batch].to(device)
                teacher_lp, student_lp = log_probabilities(ids)
                vocab = teacher_lp.shape[-1]
                all_sum += F.kl_div(
                    student_lp.reshape(-1, vocab), teacher_lp.reshape(-1, vocab),
                    log_target=True, reduction="sum",
                ).item()
                last_sum += F.kl_div(
                    student_lp[:, -1], teacher_lp[:, -1],
                    log_target=True, reduction="sum",
                ).item()
        return {
            "step": step,
            "probe_kl_all": all_sum / (probe_windows * window),
            "probe_kl_last": last_sum / probe_windows,
        }

    def save_checkpoint() -> None:
        payload = {
            "config": config,
            "step": step,
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
            "loss_history": torch.tensor(loss_history, dtype=torch.float64),
            "probe_history": probe_history,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(checkpoint_path, lambda tmp: torch.save(payload, tmp))

    def write_outputs() -> None:
        metadata = {
            "student": student_config,
            "stream_path": str(stream_path),
            "model_path": str(model_path),
            "stream_metadata": stream.metadata,
            "batch": batch,
            "steps_requested": steps,
            "steps": step,
            "complete": step == steps,
            "lr": lr,
            "min_lr": min_lr,
            "seed": seed,
            "probe_windows": probe_windows,
            "probe_seed": probe_seed,
            "final_loss": float(loss_history[-1]),
            "probe_history": probe_history,
            "trainable_parameters": sum(
                p.numel() for p in student.parameters() if p.requires_grad
            ),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(outputs[0], lambda tmp: torch.save(student.state_dict(), tmp))
        write_atomic(
            outputs[1], lambda tmp: save_numpy(tmp, np.asarray(loss_history, dtype=np.float64))
        )
        write_atomic(outputs[2], lambda tmp: save_json(tmp, metadata))

    started = time.perf_counter()
    while step < steps:
        step += 1
        current_lr = cosine_lr(
            step, base_lr=lr, min_lr=min_lr, decay_start_step=0, decay_steps=steps
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        starts = stream.sample_starts(batch, window, generator)
        ids = stream.windows(starts, window).to(device)
        teacher_lp, student_lp = log_probabilities(ids)
        vocab = teacher_lp.shape[-1]
        loss = F.kl_div(
            student_lp.reshape(-1, vocab), teacher_lp.reshape(-1, vocab),
            log_target=True, reduction="batchmean",
        )
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"non-finite distillation loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss_value)
        if step % log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"step {step} loss {loss_value:.6g} lr {current_lr:.6g} elapsed {elapsed:.1f}s",
                flush=True,
            )
        if step % eval_every == 0 or step == steps:
            probe_history.append(evaluate_probe())
        if step < steps and step % checkpoint_every == 0:
            save_checkpoint()
            write_outputs()

    write_outputs()
    checkpoint_path.unlink(missing_ok=True)
    return {
        "output_dir": str(output_dir),
        "steps": step,
        "final_loss": float(loss_history[-1]),
        "probe_kl_last": probe_history[-1]["probe_kl_last"],
    }
