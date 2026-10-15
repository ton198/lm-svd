from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Sample:
    sample_id: int
    token_ids: list[int]


def load_samples(
    samples_path: Path, history_count: int | None, future_count: int | None
) -> tuple[list[Sample], list[Sample]]:
    if not samples_path.is_file():
        raise ValueError(f"samples file does not exist: {samples_path}")
    table = pq.read_table(samples_path, columns=["sample_id", "group", "token_ids"])
    histories: list[Sample] = []
    futures: list[Sample] = []
    for group, requested, target in (
        ("history", history_count, histories),
        ("future", future_count, futures),
    ):
        rows = table.filter(pc.equal(table["group"], group))
        available = len(rows)
        if available == 0:
            raise ValueError(f"no {group} samples in {samples_path}")
        if requested is not None and available < requested:
            raise ValueError(
                f"requested {requested} {group} samples but only {available} exist"
            )
        selected = rows.slice(0, requested) if requested is not None else rows
        target.extend(
            Sample(int(row["sample_id"]), list(row["token_ids"]))
            for row in selected.to_pylist()
        )
    return histories, futures


def write_atomic(path: Path, writer: Callable[[Path], None]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        writer(temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_numpy(path: Path, matrix: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, matrix)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
