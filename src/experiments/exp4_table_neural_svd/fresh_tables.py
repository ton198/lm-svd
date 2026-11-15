"""Fresh training tables for exp4's table-switching training.

Each table draws its histories and futures from the sample file, leaving out
every sample of the validation table, and is built the same way as the stored
tables. Table `index` depends only on the seed and the index, so a resumed run
rebuilds the table it stopped in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from experiments.exp1_L_M_table.table import select_header_tokens
from experiments.svd_inference.finite_table import build_rows
from utils.common import load_samples
from utils.inference import CausalLMInference

from . import CENTERING_CHOICES
from .tables import token_tensor
from .train import Table, center_columns, table_columns


class FreshTables:
    def __init__(
        self,
        *,
        samples_path: Path,
        model_path: Path,
        inference_dir: Path,
        device: str,
        histories: int,
        futures: int,
        batch_size: int,
        seed: int,
        centering: str = "none",
    ) -> None:
        if centering not in CENTERING_CHOICES:
            raise ValueError(f"centering must be one of {CENTERING_CHOICES}, got {centering!r}")
        validation = np.load(inference_dir / "val" / "tokens.npz")
        self.top_k = validation["header_token_ids"].shape[1]
        pools = []
        for group, samples, requested in zip(
            ("history", "future"), load_samples(samples_path, None, None), (histories, futures)
        ):
            held_out = set(validation[f"{group}_sample_ids"].tolist())
            pool = [sample for sample in samples if sample.sample_id not in held_out]
            if requested > len(pool):
                raise ValueError(
                    f"requested {requested} {group} samples per table but only {len(pool)} "
                    "exist outside the validation table"
                )
            pools.append(pool)
        self._futures = pools[1]
        self._history_ids = token_tensor(pools[0], "history")
        self._future_ids = token_tensor(pools[1], "future")
        self._shape = (histories, futures)
        self._batch_size = batch_size
        self._seed = seed
        self._centering = centering
        self._device = device
        self._inference = CausalLMInference(model_path, device)

    @torch.no_grad()
    def build(self, index: int) -> Table:
        """Table `index`: distinct random histories and futures, without an SVD."""
        generator = torch.Generator().manual_seed(self._seed)
        # Replaying the earlier draws keeps every table a function of (seed, index).
        for _ in range(index + 1):
            rows = torch.randperm(len(self._history_ids), generator=generator)[: self._shape[0]]
            columns = torch.randperm(len(self._future_ids), generator=generator)[: self._shape[1]]
        history_ids, future_ids = self._history_ids[rows], self._future_ids[columns]
        header_token_ids, _ = select_header_tokens(
            self._inference, [self._futures[position] for position in columns.tolist()], self.top_k
        )
        matrix = build_rows(
            self._inference, history_ids, future_ids, header_token_ids, self._batch_size
        )
        if self._centering == "column-mean":
            matrix = center_columns(matrix)
        return Table(
            histories=history_ids.to(self._device),
            columns=table_columns(
                future_ids, torch.from_numpy(header_token_ids).long()
            ).to(self._device),
            matrix=torch.from_numpy(matrix).to(self._device),
        )
