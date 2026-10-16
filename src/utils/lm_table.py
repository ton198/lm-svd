from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def build_row_table(history_sample_ids: Sequence[int]) -> pa.Table:
    return pa.table(
        {
            "row_index": pa.array(range(len(history_sample_ids)), type=pa.int32()),
            "history_sample_id": pa.array(history_sample_ids, type=pa.int32()),
        }
    )


def build_column_table(
    future_sample_ids: Sequence[int],
    header_token_ids: np.ndarray,
    future_only_values: np.ndarray,
) -> pa.Table:
    top_k = header_token_ids.shape[1]
    column_count = len(future_sample_ids) * top_k
    return pa.table(
        {
            "column_index": pa.array(range(column_count), type=pa.int32()),
            "future_sample_id": pa.array(
                [sample_id for sample_id in future_sample_ids for _ in range(top_k)],
                type=pa.int32(),
            ),
            "rank": pa.array(
                [rank for _ in future_sample_ids for rank in range(top_k)],
                type=pa.int32(),
            ),
            "token_id": pa.array(header_token_ids.reshape(-1), type=pa.int32()),
            "future_only_centered_logit": pa.array(
                future_only_values.reshape(-1), type=pa.float32()
            ),
        }
    )


def read_source_metadata(samples_path: Path) -> object | None:
    source_metadata = pq.read_schema(samples_path).metadata or {}
    raw_source_metadata = source_metadata.get(b"lm_svd")
    return json.loads(raw_source_metadata) if raw_source_metadata else None
