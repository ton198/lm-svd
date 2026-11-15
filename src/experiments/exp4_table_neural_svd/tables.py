"""Build the exp4 training and validation tables and their exact SVDs.

Each table is an exp1 matrix: rows are histories, columns are (future, header
token) pairs grouped by future, and entries are
logit(token | history + future) - mean over the vocabulary. The two tables
share no history and no future.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from experiments.exp1_L_M_table.table import select_header_tokens
from experiments.svd_inference.finite_table import build_rows
from utils.common import Sample, load_samples, save_json, save_numpy, write_atomic
from utils.inference import CausalLMInference
from utils.linalg import matrix_svd
from utils.lm_table import read_source_metadata

from . import TABLE_NAMES

# Leading singular vectors kept per table; training compares at most this many modes.
SVD_RANK = 256
_TABLE_FILES = ("matrix.npy", "tokens.npz", "svd.npz")


def _token_tensor(samples: list[Sample], group: str) -> torch.Tensor:
    lengths = sorted({len(sample.token_ids) for sample in samples})
    if len(lengths) != 1:
        raise ValueError(
            f"exp4 needs equal-length {group} windows, got lengths {lengths[0]} to {lengths[-1]}"
        )
    return torch.tensor([sample.token_ids for sample in samples], dtype=torch.long)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    # A file object keeps np.savez from appending ".npz" to the temporary name.
    with path.open("wb") as stream:
        np.savez(stream, **arrays)


@torch.no_grad()
def build_exp4_tables(
    *,
    samples_path: Path,
    model_path: Path,
    device: str,
    train_histories: int,
    train_futures: int,
    val_histories: int,
    val_futures: int,
    top_k: int,
    batch_size: int,
    output_dir: Path,
    overwrite: bool,
) -> dict:
    """Write {train,val}/{matrix.npy,tokens.npz,svd.npz} and metadata.json.

    The training table takes the first histories and futures of the sample
    file and the validation table takes the ones that follow. metadata.json is
    written last, so its presence marks a finished build.
    """
    outputs = [output_dir / "metadata.json"] + [
        output_dir / name / file for name in TABLE_NAMES for file in _TABLE_FILES
    ]
    if not overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )
    histories, futures = load_samples(
        samples_path, train_histories + val_histories, train_futures + val_futures
    )
    splits = {
        "train": (histories[:train_histories], futures[:train_futures]),
        "val": (histories[train_histories:], futures[train_futures:]),
    }
    token_tensors = {
        name: (_token_tensor(rows, "history"), _token_tensor(columns, "future"))
        for name, (rows, columns) in splits.items()
    }
    inference = CausalLMInference(model_path, device)

    # An unfinished rebuild must not look finished.
    outputs[0].unlink(missing_ok=True)
    tables = {}
    for name, (rows, columns) in splits.items():
        table_dir = output_dir / name
        table_dir.mkdir(parents=True, exist_ok=True)
        history_ids, future_ids = token_tensors[name]
        print(f"Building {name} table: {len(rows)} histories x {len(columns)} futures", flush=True)
        header_token_ids, _ = select_header_tokens(inference, columns, top_k)
        matrix = build_rows(inference, history_ids, future_ids, header_token_ids, batch_size)

        started = time.perf_counter()
        svd = matrix_svd(matrix, device=device)
        kept = min(SVD_RANK, len(svd.S))
        print(
            f"Exact SVD of the {name} table took {time.perf_counter() - started:.1f}s; "
            f"leading singular values {svd.S[:4].tolist()}",
            flush=True,
        )
        write_atomic(table_dir / "matrix.npy", lambda tmp: save_numpy(tmp, matrix))
        write_atomic(table_dir / "tokens.npz", lambda tmp: _save_npz(
            tmp,
            histories=history_ids.numpy(),
            futures=future_ids.numpy(),
            header_token_ids=header_token_ids,
            history_sample_ids=np.array([sample.sample_id for sample in rows]),
            future_sample_ids=np.array([sample.sample_id for sample in columns]),
        ))
        write_atomic(table_dir / "svd.npz", lambda tmp: _save_npz(
            tmp, S=svd.S, U=svd.U[:, :kept], Vt=svd.Vt[:kept],
        ))
        tables[name] = {
            "histories": len(rows),
            "futures": len(columns),
            "matrix_shape": list(matrix.shape),
            "singular_vectors_kept": kept,
            "numerical_rank": svd.rank,
            "frobenius_norm_sq": float(np.square(svd.S).sum()),
        }

    metadata = {
        "samples_path": str(samples_path),
        "model_path": str(model_path),
        "device": device,
        "top_k": top_k,
        "header_context": "future_only",
        "value": "logit(token | history + future) - mean(logits over full vocabulary)",
        "column_order": "future-major, then descending future-only logit rank",
        "svd": "float64 thin SVD of the stored matrix, no secondary centering",
        "split": "train takes the leading histories and futures; val takes the following ones",
        "tables": tables,
        "source_metadata": read_source_metadata(samples_path),
    }
    write_atomic(outputs[0], lambda tmp: save_json(tmp, metadata))
    return {"output_dir": str(output_dir), "tables": tables}
