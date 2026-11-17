"""Files of one exp8 training run: metadata.json, eval_history.jsonl, checkpoint.pt.

A run is restarted by repeating its command: the checkpoint is loaded when its
settings equal the command's, so a requeued cluster job continues where its last
evaluation left it.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import torch

from utils.common import save_json, write_atomic

WARMUP_STEPS = 1000
# The learning rate ends at this share of its peak.
FINAL_LR_SHARE = 0.1


def lr_factor(step: int, steps: int) -> float:
    """Linear warm-up over WARMUP_STEPS, then a cosine decay to FINAL_LR_SHARE at `steps`."""
    warmup = min(WARMUP_STEPS, max(steps // 10, 1))
    if step < warmup:
        return (step + 1) / warmup
    progress = min((step - warmup) / max(steps - warmup, 1), 1.0)
    return FINAL_LR_SHARE + (1 - FINAL_LR_SHARE) * 0.5 * (1 + math.cos(math.pi * progress))


class RunFiles:
    def __init__(self, output_dir: Path, config: dict, overwrite: bool) -> None:
        self.output_dir = Path(output_dir)
        self.config = config
        if overwrite and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "checkpoint.pt"

    def load(self) -> dict | None:
        """The saved state, or None for a new run. Rejects a run with other settings."""
        if not self.checkpoint_path.is_file():
            leftovers = [path.name for path in self.output_dir.iterdir()
                         if path.name != "metadata.json"]
            if leftovers:
                raise FileExistsError(
                    f"{self.output_dir} holds {leftovers} but no checkpoint; "
                    "pass --overwrite to start over")
            return None
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if state["config"] != self.config:
            changed = sorted(
                key for key in set(state["config"]) | set(self.config)
                if state["config"].get(key) != self.config.get(key))
            raise ValueError(
                f"{self.output_dir} was trained with other settings ({', '.join(changed)}); "
                "pass --overwrite to start over")
        self.history = state["history"]
        return state

    def write_metadata(self, metadata: dict) -> None:
        write_atomic(self.output_dir / "metadata.json", lambda tmp: save_json(tmp, metadata))

    def save(self, state: dict) -> None:
        """Checkpoint first, then the history it contains."""
        state = {**state, "config": self.config, "history": self.history}
        write_atomic(self.checkpoint_path, lambda tmp: torch.save(state, tmp))
        text = "".join(json.dumps(record) + "\n" for record in self.history)
        write_atomic(
            self.output_dir / "eval_history.jsonl",
            lambda tmp: tmp.write_text(text, encoding="utf-8"))
