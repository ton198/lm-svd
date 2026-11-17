"""Train exp7 one mode at a time and compare it with the exact SVDs.

The pairs, the target (the table entry minus g), the evaluation tables and the
comparison are exp5's; see experiments.exp5_gpt2_mlp_table.train. Stage l
builds mode l (model.py) and trains it on the target minus the outputs of the
frozen modes 1..l-1 (stage.py). Every evaluation compares the l modes with the
exact SVD of both exp4 tables at ranks 1..l. exp4's train table is the
selection table: only it decides when a stage ends and which weights are
frozen. exp4's val table is the report table. At the end of a stage the mode
is frozen and saved, and all l modes are aligned on fresh store pairs
(align.py). The alignment changes no weights and no later stage.

checkpoint.pt records the progress: the frozen modes in modes/ are final and at
most one stage is in progress. The same command continues an interrupted run or
adds modes to a finished one.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from data_pipline.sample_store import SampleSampler
from experiments.exp4_table_neural_svd.train import compare_with_exact_svd, format_comparison
from experiments.exp5_gpt2_mlp_table.features import (
    future_only_logits,
    header_tokens,
    last_hidden,
    pair_targets,
)
from experiments.exp5_gpt2_mlp_table.tables import EvalTable
from experiments.exp5_gpt2_mlp_table.train import (
    FIRST_MODE,
    draw_pairs,
    prepare_run,
    run_metadata,
    sampler_seed,
)
from utils.common import save_json, write_atomic
from utils.inference import CausalLMInference

from . import PRETRAINED, SUBNETS, TABLE_KEYS, WARMUP_STEPS
from .align import Moments, alignment, max_correlation
from .model import Mode, build_mode, token_vectors
from .stage import Selection, residual_target, stage_loss

# Added to the stage number when seeding the weights of a new mode, so the seeds
# of the modes never meet the sampler's seeds, which use the global step.
MODE_SEED_OFFSET = 10**9
# metadata.json's description of each encoder of a mode, per kind.
ENCODERS = {
    "scratch": "a randomly initialised BERT transformer (bidirectional attention, no pooler, "
               "dropout 0) with its own token and position embeddings, reading the model's "
               "token ids; pooled vector: the mean of its last hidden state over all positions",
    "gpt2": "a trainable copy of the model's transformer (causal attention, dropout off); "
            "pooled vector: the hidden state of the last position",
}


def frozen_outputs(
    modes: list[Mode],
    vectors: torch.Tensor,
    histories: torch.Tensor,
    futures: torch.Tensor,
    header_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outputs of frozen modes without gradient: left (B, k) and right (B, K, k)."""
    batch, top_k = header_ids.shape
    if not modes:
        left = histories.new_zeros((batch, 0), dtype=torch.float32)
        return left, histories.new_zeros((batch, top_k, 0), dtype=torch.float32)
    with torch.no_grad():
        left = torch.stack([mode.left_features(histories) for mode in modes], dim=1)
        right = torch.stack(
            [mode.right_features(futures, header_ids, vectors) for mode in modes], dim=2)
    return left, right


def train_step(
    current: Mode,
    frozen: list[Mode],
    vectors: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    histories: torch.Tensor,
    futures: torch.Tensor,
    header_ids: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    """One optimizer step of current on the target minus the frozen modes.

    histories (B, H), futures (B, T), header_ids (B, K), target (B, K). Returns
    the loss and the mean square of the residual it was fitted to.
    """
    device = target.device
    histories, futures, header_ids = (x.to(device) for x in (histories, futures, header_ids))
    residual = residual_target(
        target, *frozen_outputs(frozen, vectors, histories, futures, header_ids))
    current.train()
    optimizer.zero_grad()
    loss = stage_loss(
        current.left_features(histories),
        current.right_features(futures, header_ids, vectors),
        residual,
    )
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss {loss.item()}")
    loss.backward()
    optimizer.step()
    return loss.item(), residual.square().mean().item()


@torch.no_grad()
def mode_table_features(
    mode: Mode, table: EvalTable, vectors: torch.Tensor, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Features of every row (N,) and every column (C,) of a table, in float64.

    Columns are future-major: column future * top_k + j. chunk is the sequences
    per encoder pass.
    """
    mode.eval()
    histories = table.table.histories
    futures = table.future_ids
    header_ids = table.table.columns[:, -1].reshape(len(futures), -1)
    left = torch.cat([
        mode.left_features(histories[start:start + chunk]).double()
        for start in range(0, len(histories), chunk)
    ])
    right = torch.cat([
        mode.right_features(
            futures[start:start + chunk], header_ids[start:start + chunk], vectors
        ).double().reshape(-1)
        for start in range(0, len(futures), chunk)
    ])
    return left, right


def selection_energy(record: dict) -> float:
    """The selection metric: captured energy of the selection table at the largest rank."""
    return record["selection"]["captured_energy"][-1]


def warmup_factor(step: int) -> float:
    """Learning-rate factor of a scratch stage: linear over its first WARMUP_STEPS steps."""
    return min(1.0, (step + 1) / WARMUP_STEPS)


def _write_lines(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _write_stage_line(path: Path, line: dict) -> None:
    """Add a stage's line to stages.jsonl, dropping lines of it and later stages.

    A run interrupted after the line was written finishes the stage again.
    """
    previous = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [record for record in map(json.loads, previous) if record["stage"] < line["stage"]]
    write_atomic(path, lambda tmp: _write_lines(tmp, [*kept, line]))


def _mode_path(output_dir: Path, stage: int) -> Path:
    """The file of the frozen mode of a stage."""
    return output_dir / "modes" / f"mode-{stage:02d}.pt"


def _read_checkpoint(output_dir: Path, config: dict, modes: int) -> dict | None:
    """checkpoint.pt of output_dir if it can be continued to `modes` modes.

    None without a checkpoint. Rejects other settings, fewer modes than are
    finished (or are being trained) and a missing mode file.
    """
    path = output_dir / "checkpoint.pt"
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    saved = checkpoint["config"]
    if saved != config:
        changed = {
            key: (saved.get(key), config.get(key))
            for key in sorted(set(saved) | set(config)) if saved.get(key) != config.get(key)}
        raise ValueError(
            f"settings differ from the checkpoint in {output_dir} (saved, now): {changed}; "
            "pass --overwrite to start over")
    completed = checkpoint["completed_modes"]
    if modes < completed:
        raise ValueError(
            f"{output_dir} holds {completed} finished modes; cannot reduce them to "
            f"--modes {modes} (pass --overwrite to start over)")
    if modes == completed and checkpoint["stage"] is not None:
        raise ValueError(
            f"stage {completed + 1} is in progress in {output_dir}; the {completed} finished "
            f"mode(s) are intact and --modes {completed + 1} or more continues the stage; "
            "--overwrite starts again from nothing")
    for stage in range(1, completed + 1):
        if not _mode_path(output_dir, stage).is_file():
            raise ValueError(
                f"checkpoint.pt counts {completed} finished modes but "
                f"{_mode_path(output_dir, stage)} is missing; pass --overwrite to start over")
    return checkpoint


@dataclass
class StageState:
    """A stage in progress: what checkpoint.pt stores besides the frozen modes."""

    stage: int  # 1-based; the number of the mode being trained
    mode: Mode
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR | None  # scratch warmup only
    selection: Selection
    stage_step: int = 0
    best_state_dict: dict[str, torch.Tensor] | None = None  # on the CPU


class StagedRun:
    """The frozen modes, their table features and the output files of one run.

    config holds every setting the run's results depend on; it is the config of
    checkpoint.pt.
    """

    def __init__(
        self,
        *,
        inference: CausalLMInference,
        tables: dict[str, EvalTable],
        top_k: int,
        vectors: torch.Tensor,
        sampler: SampleSampler,
        config: dict,
        output_dir: Path,
        device: str,
        encode_batch: int,
        eval_chunk: int,
    ) -> None:
        self.inference = inference
        self.tables = tables
        self.top_k = top_k
        self.vectors = vectors
        self.sampler = sampler
        self.config = config
        self.output_dir = output_dir
        self.device = device
        self.encode_batch = encode_batch
        self.eval_chunk = eval_chunk
        self.frozen: list[Mode] = []
        # Per exp4 table name, the (rows (N,), columns (C,)) features of each frozen mode.
        self.features: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
            name: [] for name in tables}
        self.step = 0
        self.history: list[dict] = []
        self.started = time.perf_counter()

    def new_mode(self) -> Mode:
        config = self.config
        return build_mode(
            self.inference, config["subnet"], scratch_layers=config["scratch_layers"],
            scratch_width=config["scratch_width"], scratch_heads=config["scratch_heads"],
        ).to(self.device)

    def start_stage(self, stage: int, saved: dict | None = None) -> StageState:
        """Mode `stage` with its own optimizer.

        A new stage is seeded by the stage number; with saved, the stage dict of
        checkpoint.pt, the stage continues where it was saved.
        """
        config = self.config
        if saved is None:
            torch.manual_seed(sampler_seed(config["seed"], MODE_SEED_OFFSET + stage))
        mode = self.new_mode()
        if config["subnet"] in PRETRAINED:
            optimizer = torch.optim.Adam([
                {"params": [*mode.left_head.parameters(), *mode.right_head.parameters()],
                 "lr": config["lr"]},
                {"params": [*mode.left_encoder.parameters(), *mode.right_encoder.parameters()],
                 "lr": config["encoder_lr"]},
            ])
            scheduler = None
        else:
            optimizer = torch.optim.Adam(mode.parameters(), lr=config["lr"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_factor)
        if saved is None:
            selection = Selection(
                max_steps=config["stage_steps"], min_steps=config["min_stage_steps"],
                patience=config["patience"])
            return StageState(stage, mode, optimizer, scheduler, selection)
        mode.load_state_dict(saved["mode_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler_state_dict"])
        return StageState(
            stage, mode, optimizer, scheduler, Selection(**saved["selection"]),
            stage_step=saved["stage_step"], best_state_dict=saved["best_state_dict"])

    def write_history(self) -> None:
        write_atomic(
            self.output_dir / "eval_history.jsonl", lambda tmp: _write_lines(tmp, self.history))

    def resume(self, checkpoint: dict) -> StageState:
        """Load the frozen modes and the progress of checkpoint.pt; the stage to run next.

        eval_history.jsonl is rewritten from the checkpoint: a run killed after the
        checkpoint and before the history was written left it a record short.
        """
        self.step, self.history = checkpoint["step"], checkpoint["history"]
        self.write_history()
        for stage in range(1, checkpoint["completed_modes"] + 1):
            mode = self.new_mode()
            mode.load_state_dict(torch.load(
                _mode_path(self.output_dir, stage), map_location="cpu",
                weights_only=True)["state_dict"])
            self.freeze(mode)
        return self.start_stage(len(self.frozen) + 1, checkpoint["stage"])

    def draw_batch(
        self, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Histories (B, H), futures (B, T), header ids (B, K) and targets (B, K) of fresh pairs."""
        histories, futures, pair_logits = draw_pairs(self.sampler, count)
        future_hidden = last_hidden(self.inference, futures, self.encode_batch)
        header_ids, g = header_tokens(
            future_only_logits(self.inference, future_hidden), self.top_k)
        target = pair_targets(pair_logits.to(self.device), header_ids, g)
        return histories.to(self.device), futures.to(self.device), header_ids, target

    def compare(
        self,
        current: Mode | None = None,
        transforms: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, dict]:
        """compare_with_exact_svd of the frozen modes (and current) on both tables.

        With transforms (T_f, T_g) the features are multiplied by them first.
        """
        result = {}
        for name, table in self.tables.items():
            pairs = list(self.features[name])
            if current is not None:
                pairs.append(mode_table_features(current, table, self.vectors, self.eval_chunk))
            left = torch.stack([rows for rows, _ in pairs], dim=1)
            right = torch.stack([columns for _, columns in pairs], dim=1)
            if transforms is not None:
                left, right = left @ transforms[0].to(left), right @ transforms[1].to(right)
            result[TABLE_KEYS[name]] = compare_with_exact_svd(
                left, right, table.table, ranks=tuple(range(1, left.shape[1] + 1)))
        return result

    def write_checkpoint(self, state: StageState | None) -> None:
        stage = None if state is None else {
            "stage_step": state.stage_step,
            "mode_state_dict": state.mode.state_dict(),
            "best_state_dict": state.best_state_dict,
            "optimizer_state_dict": state.optimizer.state_dict(),
            "scheduler_state_dict": (
                None if state.scheduler is None else state.scheduler.state_dict()),
            "selection": dataclasses.asdict(state.selection),
        }
        write_atomic(self.output_dir / "checkpoint.pt", lambda tmp: torch.save({
            "config": self.config,
            "completed_modes": len(self.frozen),
            "step": self.step,
            "history": self.history,
            "stage": stage,
        }, tmp))

    def run_stage(self, state: StageState) -> None:
        """Train state.mode from state.stage_step until the stage ends, then freeze it."""
        batch_size, eval_every = self.config["batch_size"], self.config["eval_every"]
        stage_steps = self.config["stage_steps"]
        if state.stage_step == 0:
            print(
                f"Stage {state.stage}: mode {state.stage} on the target minus "
                f"{len(self.frozen)} frozen mode(s)",
                flush=True,
            )
        else:
            print(
                f"Resuming stage {state.stage} at stage step {state.stage_step} "
                f"(step {self.step})",
                flush=True,
            )
            # Saved at its last evaluation: the run stopped while freezing the mode.
            reason = state.selection.reason(state.stage_step)
            if reason is not None:
                self.finish_stage(state, reason)
                return
        loss_sum = kernel_sum = residual_sum = 0.0
        window = 0
        window_started = time.perf_counter()
        while True:
            histories, futures, header_ids, target = self.draw_batch(batch_size)
            loss, residual_energy = train_step(
                state.mode, self.frozen, self.vectors, state.optimizer,
                histories, futures, header_ids, target)
            if state.scheduler is not None:
                state.scheduler.step()
            state.stage_step += 1
            self.step += 1
            loss_sum += loss
            kernel_sum += target.square().mean().item()
            residual_sum += residual_energy
            window += 1
            if state.stage_step % eval_every and state.stage_step != stage_steps:
                continue

            now = time.perf_counter()
            record = {
                "step": self.step,
                "stage": state.stage,
                "stage_step": state.stage_step,
                "train_loss": loss_sum / window,
                "kernel_energy": kernel_sum / window,
                "residual_energy": residual_sum / window,
                "seconds_per_step": (now - window_started) / window,
                **self.compare(state.mode),
            }
            metric = selection_energy(record)
            if not math.isfinite(metric):
                raise RuntimeError(
                    f"non-finite selection metric {metric} at stage {state.stage} step "
                    f"{state.stage_step} (step {self.step})")
            lines = [
                f"stage {state.stage} step {state.stage_step} (step {self.step}): "
                f"loss {record['train_loss']:.4f}  E[K^2] {record['kernel_energy']:.4f}  "
                f"E[r^2] {record['residual_energy']:.4f}  "
                f"{record['seconds_per_step']:.3f}s/step  elapsed {now - self.started:.0f}s"
            ]
            lines.extend(format_comparison(key, record[key]) for key in TABLE_KEYS.values())
            print("\n".join(lines), flush=True)
            self.history.append(record)
            if state.selection.record(metric, state.stage_step):
                state.best_state_dict = {
                    key: value.detach().to("cpu", copy=True)
                    for key, value in state.mode.state_dict().items()}
            # The checkpoint first: eval_history.jsonl never holds a record that the
            # checkpoint does not, and a resumed run rewrites it from the checkpoint.
            self.write_checkpoint(state)
            self.write_history()
            reason = state.selection.reason(state.stage_step)
            if reason is not None:
                self.finish_stage(state, reason)
                return
            loss_sum = kernel_sum = residual_sum = 0.0
            window = 0
            window_started = time.perf_counter()

    def freeze(self, mode: Mode) -> None:
        """Freeze a mode and keep its table features, which no longer change."""
        mode.zero_grad(set_to_none=True)
        mode.requires_grad_(False).eval()
        for name, table in self.tables.items():
            self.features[name].append(
                mode_table_features(mode, table, self.vectors, self.eval_chunk))
        self.frozen.append(mode)

    def align(self) -> tuple[Moments, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Moments of all frozen modes on align_pairs fresh pairs, and (T_f, T_g, S)."""
        moments = Moments(len(self.frozen))
        remaining = self.config["align_pairs"]
        while remaining > 0:
            count = min(self.config["batch_size"], remaining)
            histories, futures, header_ids, target = self.draw_batch(count)
            left, right = frozen_outputs(self.frozen, self.vectors, histories, futures, header_ids)
            # The target, not the residual: the alignment refits all modes together.
            moments.add(left, right, target)
            remaining -= count
        return moments, *alignment(moments.lam_f, moments.lam_g, moments.cross)

    def finish_stage(self, state: StageState, reason: str) -> None:
        """Freeze the stage's mode, save it, align all modes and record the stage."""
        stage, mode, selection = state.stage, state.mode, state.selection
        if selection.keeps_best:
            mode.load_state_dict(state.best_state_dict)
            best_step = selection.best_step
            best_energy = selection.best
        else:
            best_step = state.stage_step
            best_energy = selection_energy(self.history[-1])
        self.freeze(mode)
        path = _mode_path(self.output_dir, stage)
        path.parent.mkdir(exist_ok=True)
        weights = {key: value.detach().cpu() for key, value in mode.state_dict().items()}
        write_atomic(path, lambda tmp: torch.save(
            {"state_dict": weights, "stage": stage, "best_step": best_step}, tmp))

        moments, T_f, T_g, S = self.align()
        write_atomic(self.output_dir / "alignment.pt", lambda tmp: torch.save(
            {"T_f": T_f, "T_g": T_g, "S": S, "modes": len(self.frozen)}, tmp))
        aligned = self.compare(transforms=(T_f, T_g))
        line = {
            "stage": stage,
            "steps": state.stage_step,
            "best_step": best_step,
            "reason": reason,
            "selection_energy": best_energy,
            **self.compare(),
            "aligned": aligned,
            "singular_values": S.tolist(),
            "left_correlation": max_correlation(moments.lam_f),
            "right_correlation": max_correlation(moments.lam_g),
        }
        _write_stage_line(self.output_dir / "stages.jsonl", line)
        self.write_checkpoint(None)
        print(
            f"stage {stage} finished at stage step {state.stage_step} ({reason}), "
            f"froze stage step {best_step}; aligned energy at rank {len(self.frozen)}: "
            f"selection {aligned['selection']['captured_energy'][-1]:.5f}  "
            f"report {aligned['report']['captured_energy'][-1]:.5f}  "
            f"max correlation left {line['left_correlation']:.3f} "
            f"right {line['right_correlation']:.3f}",
            flush=True,
        )


def train_exp7(
    *,
    inference_dir: Path,
    store_dir: Path,
    model_path: Path,
    output_dir: Path,
    device: str,
    subnet: str,
    modes: int,
    stage_steps: int,
    min_stage_steps: int,
    patience: int,
    align_pairs: int,
    scratch_layers: int,
    scratch_width: int,
    scratch_heads: int,
    batch_size: int,
    eval_every: int,
    lr: float,
    encoder_lr: float,
    seed: int,
    overwrite: bool,
    encode_batch: int = 1024,
    eval_chunk: int = 256,
) -> dict:
    """Train `modes` modes one stage at a time on store pairs.

    A stage evaluates every eval_every of its steps and at its last step, and
    ends after stage_steps steps or, with patience > 0, once past
    min_stage_steps the selection metric has not improved for `patience`
    evaluations; it then freezes the weights of its best evaluation (its last
    with patience 0). Modes of a pretrained kind (PRETRAINED) train their
    encoders with encoder_lr and their heads with lr; scratch modes train
    everything with lr after a linear warmup.

    Writes eval_history.jsonl (one line per evaluation), stages.jsonl (one
    line per finished stage), modes/mode-XX.pt, alignment.pt (the latest
    alignment), checkpoint.pt and metadata.json to output_dir.

    With a checkpoint.pt of the same settings (modes may differ) the run
    continues: the stage in progress from its last evaluation, then the stages
    up to `modes`. With all `modes` modes finished it returns the summary and
    writes nothing. overwrite deletes the outputs and starts again; without a
    checkpoint an existing eval_history.jsonl needs it.
    """
    if subnet not in SUBNETS:
        raise ValueError(f"subnet must be one of {SUBNETS}, got {subnet!r}")
    for name, value, smallest in (
        ("modes", modes, 1), ("stage_steps", stage_steps, 1), ("eval_every", eval_every, 1),
        ("batch_size", batch_size, 1), ("align_pairs", align_pairs, 1),
        ("min_stage_steps", min_stage_steps, 0), ("patience", patience, 0),
    ):
        if value < smallest:
            raise ValueError(f"{name} must be at least {smallest}, got {value}")
    if subnet == "scratch":
        for name, value in (("scratch_layers", scratch_layers), ("scratch_width", scratch_width),
                            ("scratch_heads", scratch_heads)):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if scratch_width % scratch_heads:
            raise ValueError(
                f"scratch_width must be divisible by scratch_heads, got {scratch_width} "
                f"and {scratch_heads}")
    config = {
        "inference_dir": str(inference_dir), "store_dir": str(store_dir),
        "model_path": str(model_path), "subnet": subnet, "stage_steps": stage_steps,
        "min_stage_steps": min_stage_steps, "patience": patience, "align_pairs": align_pairs,
        "scratch_layers": scratch_layers, "scratch_width": scratch_width,
        "scratch_heads": scratch_heads, "batch_size": batch_size, "eval_every": eval_every,
        "lr": lr, "encoder_lr": encoder_lr, "seed": seed, "optimizer": "Adam",
        "first_mode": FIRST_MODE,
    }

    output_dir = Path(output_dir)
    history_path = output_dir / "eval_history.jsonl"
    checkpoint = None if overwrite else _read_checkpoint(output_dir, config, modes)
    if checkpoint is None and not overwrite and history_path.exists():
        raise FileExistsError(
            f"output already exists: {history_path}; pass --overwrite to replace it")
    if checkpoint is not None and checkpoint["completed_modes"] == modes:
        # Every mode is finished: no model, no tables, no file is touched.
        print(f"{output_dir} already holds {modes} finished modes", flush=True)
        return {
            "output_dir": str(output_dir), "step": checkpoint["step"], "modes": modes,
            "evaluations": len(checkpoint["history"]),
        }

    inference, tables, top_k, store_config = prepare_run(
        inference_dir=inference_dir, store_dir=store_dir, model_path=model_path,
        device=device, seed=seed, encode_batch=encode_batch,
    )
    # compare_with_exact_svd drops ranks above the exact singular vectors the table
    # holds, so beyond them the selection metric would not depend on the current mode.
    selection_name = next(name for name, key in TABLE_KEYS.items() if key == "selection")
    exact = tables[selection_name].table
    exact_modes = min(exact.left_vectors.shape[1], exact.right_vectors.shape[1])
    if modes > exact_modes:
        raise ValueError(
            f"--modes {modes} exceeds the {exact_modes} exact singular vectors of the "
            f"selection table (exp4 {selection_name!r} in {inference_dir})")
    if overwrite:
        for name in ("checkpoint.pt", "eval_history.jsonl", "stages.jsonl", "alignment.pt"):
            (output_dir / name).unlink(missing_ok=True)
        shutil.rmtree(output_dir / "modes", ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The frozen teacher's constant, shared by every mode and saved nowhere.
    vectors = token_vectors(inference.model.get_output_embeddings().weight).to(device)
    run = StagedRun(
        inference=inference, tables=tables, top_k=top_k, vectors=vectors,
        # Seeded by the global step the run starts from, as in exp5, so a continued
        # run does not replay the pairs it has seen.
        sampler=SampleSampler(
            store_dir, seed=sampler_seed(seed, 0 if checkpoint is None else checkpoint["step"])),
        config=config, output_dir=output_dir, device=device, encode_batch=encode_batch,
        eval_chunk=eval_chunk,
    )
    if checkpoint is None:
        state = run.start_stage(1)
    else:
        state = run.resume(checkpoint)
        # It holds CPU copies of the stage's weights and optimizer state.
        del checkpoint

    shared = run_metadata(
        device=device, top_k=top_k, tables=tables, model=state.mode, store_config=store_config)
    shared["ranks"] = "1..l at stage l"
    shared["table_shapes"] = {
        TABLE_KEYS[name]: list(table.table.matrix.shape) for name, table in tables.items()}
    roles = {"selection": "decides when a stage ends and which weights are frozen",
             "report": "reported only"}
    shared["tables"] = {
        TABLE_KEYS[name]: {"exp4_table": name, "role": roles[TABLE_KEYS[name]], **values}
        for name, values in shared["tables"].items()}
    write_atomic(output_dir / "metadata.json", lambda tmp: save_json(tmp, {
        **config,
        "modes": modes,
        "warmup_steps": 0 if subnet in PRETRAINED else WARMUP_STEPS,
        "encoder": ENCODERS[subnet],
        "left_side": "per mode: its own encoder reading the history, pooled to one vector; "
                     "LayerNorm + linear head to one number",
        "right_side": "per mode: its own encoder reading the future, pooled to one vector; "
                      "LayerNorm + linear head to w(future), the unembedding width E; "
                      "feature <w(future), v_z> / sqrt(E) with v_z the frozen model's "
                      "centered unembedding row of the header token over the root mean "
                      "square of all entries",
        "trainable_parameters_of": "one mode; no two modes share a trainable parameter",
        **shared,
    }))

    print(f"Waiting for the first sample block in {store_dir}", flush=True)
    run.sampler.wait_until_ready()
    if subnet == "gpt2":
        rates = (
            f"copies of the model's transformer read at the last token; heads lr {lr}, "
            f"encoders lr {encoder_lr}")
    else:
        rates = (
            f"bidirectional scratch transformers of {scratch_layers} layers, width "
            f"{scratch_width}, {scratch_heads} heads, mean-pooled; lr {lr} with a linear "
            f"warmup over the first {WARMUP_STEPS} steps of each stage")
    print(
        f"Training {modes} modes on {device}: {batch_size} pairs x {top_k} header tokens "
        f"per step, up to {stage_steps} steps per stage, {rates}",
        flush=True,
    )
    while True:
        run.run_stage(state)
        # The finished stage's optimizer state and best weights go before the next
        # mode is built: Adam's moments of a gpt2 mode are two copies of the model.
        # collect() frees what a reference cycle (the warmup scheduler) still holds.
        del state
        gc.collect()
        if len(run.frozen) == modes:
            break
        state = run.start_stage(len(run.frozen) + 1)
    return {
        "output_dir": str(output_dir), "step": run.step, "modes": len(run.frozen),
        "evaluations": len(run.history),
    }
