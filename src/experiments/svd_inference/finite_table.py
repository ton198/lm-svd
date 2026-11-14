"""Finite-table LinGen baseline: fit history coefficients on an exp1-style table.

Rows are histories h and columns are (future f, header token z) pairs, with
entries logit(z | h + f) - mean over the vocabulary. A target history's row is
regressed on the basis rows, and the same coefficients are then applied to the
basis histories' own next-token logits:

    LM(ht)  vs  sum_i v_i LM(h_i)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from data_pipline.sample_store.store import SampleSampler
from experiments.exp1_L_M_table.table import select_header_tokens
from utils.common import Sample, save_json, save_numpy, write_atomic
from utils.inference import CausalLMInference

from .lingen import summarize_next_token_approximation

DEFAULT_STORE_DIR = Path("data/sample_stores/wikitext-103-train__gpt2-small__h64-f32")
DEFAULT_MODEL_PATH = Path("models/gpt2-small")
_RUN_DIR = Path(
    "data/experiment_results/svd_inference_finite_table/wikitext-103-train__gpt2-small"
)
DEFAULT_OUTPUT_DIR = _RUN_DIR / "analysis"
DEFAULT_TABLE_DIR = _RUN_DIR / "inference"
_FIELDS = {"h": 0, "f": 1}


def sample_distinct(
    sampler: SampleSampler, count: int, field: str, exclude: torch.Tensor | None = None
) -> torch.Tensor:
    """Draw `count` token windows that differ from each other and from `exclude`.

    SampleSampler draws with replacement from a small pool, so repeats are
    common. A target history that also sits in the basis would be fitted by a
    single coefficient of one, and the evaluation would measure nothing.
    """
    if field not in _FIELDS:
        raise ValueError(f"field must be one of {sorted(_FIELDS)}, got {field!r}")
    seen = set() if exclude is None else {tuple(row) for row in exclude.tolist()}
    rows: list[torch.Tensor] = []
    # Far more draws than a store with enough distinct rows needs.
    for _ in range(100 * (count + len(seen)) + 1000):
        if len(rows) == count:
            break
        row = sampler.sample()[_FIELDS[field]]
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            rows.append(row)
    if len(rows) < count:
        raise ValueError(
            f"found only {len(rows)} of {count} distinct {field!r} windows in {sampler.store_dir}"
        )
    return torch.stack(rows)


def build_rows(
    inference: CausalLMInference,
    histories: torch.Tensor,
    futures: torch.Tensor,
    header_token_ids: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Batched exp1 `build_matrix`: one (futures * top_k) row per history."""
    header = torch.as_tensor(header_token_ids, dtype=torch.long, device=inference.device)
    matrix = np.empty((len(histories), header.numel()), dtype=np.float32)
    started = time.perf_counter()
    for history_index, history in enumerate(histories):
        chunks = []
        for start in range(0, len(futures), batch_size):
            batch = futures[start : start + batch_size]
            sequences = torch.cat((history.expand(len(batch), -1), batch), dim=1)
            logits = inference.next_token_logits_batch(sequences.tolist())
            centered = logits - logits.mean(dim=-1, keepdim=True)
            chunks.append(centered.gather(1, header[start : start + batch_size]))
        matrix[history_index] = torch.cat(chunks).flatten().cpu().numpy()
        done = history_index + 1
        if done % 64 == 0 or done == len(histories):
            print(
                f"history {done}/{len(histories)} done, elapsed {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return matrix


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.savez(stream, **arrays)


def fit_coefficients(table, target_rows) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares coefficients of each target row on the basis rows.

    With basis table M of shape (N, C) and target rows R of shape (T, C), solve

        v_t = argmin_v ||v M - r_t||_2^2

    in float64. The minimum-norm solution is returned when M is rank deficient.
    Coefficients are unconstrained: they may be negative and need not sum to
    one. Also return each target's relative squared residual
    ||v_t M - r_t||^2 / ||r_t||^2.
    """
    table = torch.as_tensor(table, dtype=torch.float64)
    target_rows = torch.as_tensor(target_rows, dtype=torch.float64)
    coefficients = torch.linalg.lstsq(table.T, target_rows.T, driver="gelsd").solution.T
    residual = (coefficients @ table - target_rows).square().sum(dim=-1)
    return coefficients, residual / target_rows.square().sum(dim=-1)


@torch.no_grad()
def report_finite_table_baseline(
    *,
    store_dir: str | Path = DEFAULT_STORE_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    table_dir: str | Path | None = DEFAULT_TABLE_DIR,
    num_basis: int = 8192,
    num_futures: int = 2048,
    top_k: int = 50,
    num_targets: int = 50,
    batch_size: int = 1024,
    seed: int = 0,
    device: str | None = None,
) -> dict[str, float]:
    """Run the baseline; print and save mean/std metrics over target histories.

    The defaults are the paper's setup (40,000 histories, 10,000 futures, 50
    targets, top-50 tokens) with histories and futures each cut by about five
    to fit a few GPU hours. With table_dir set, the sampled tokens, the table
    and the evaluated logits are saved there before the fit, so coefficients
    can be refitted later without rerunning the model.
    """
    for name, value in (("num_basis", num_basis), ("num_futures", num_futures),
                        ("top_k", top_k), ("num_targets", num_targets),
                        ("batch_size", batch_size)):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    store_dir, model_path, output_dir = Path(store_dir), Path(model_path), Path(output_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_top_k = 10

    # Targets are drawn first, as in lingen.report_next_token_approximation.
    sampler = SampleSampler(store_dir, seed=seed)
    targets = sample_distinct(sampler, num_targets, "h")
    basis = sample_distinct(sampler, num_basis, "h", exclude=targets)
    futures = sample_distinct(sampler, num_futures, "f")

    gpt_model = CausalLMInference(model_path, device=device)
    header_token_ids, _ = select_header_tokens(
        gpt_model, [Sample(index, row) for index, row in enumerate(futures.tolist())], top_k
    )
    print(f"Building {num_basis} x {num_futures * top_k} basis table", flush=True)
    table = build_rows(gpt_model, basis, futures, header_token_ids, batch_size)
    print(f"Building {num_targets} target rows", flush=True)
    target_rows = build_rows(gpt_model, targets, futures, header_token_ids, batch_size)
    if table_dir is not None:
        table_dir = Path(table_dir)
        table_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(table_dir / "tokens.npz", lambda tmp: _save_npz(
            tmp, targets=targets.numpy(), basis=basis.numpy(), futures=futures.numpy(),
            header_token_ids=header_token_ids,
        ))
        write_atomic(table_dir / "table.npy", lambda tmp: save_numpy(tmp, table))
        write_atomic(table_dir / "target_rows.npy", lambda tmp: save_numpy(tmp, target_rows))
        print(f"Saved table to {table_dir}", flush=True)

    print("Fitting coefficients", flush=True)
    coefficients, _ = fit_coefficients(table, target_rows)

    # LM(x): full-vocabulary next-token logits after the history alone.
    basis_logits = torch.cat([
        gpt_model.next_token_logits_batch(basis[start : start + batch_size].tolist())
        for start in range(0, num_basis, batch_size)
    ]).to(device="cpu", dtype=torch.float64)
    true_logits = gpt_model.next_token_logits_batch(targets.tolist()).to(
        device="cpu", dtype=torch.float64
    )
    if table_dir is not None:
        for name, logits in (("basis_logits", basis_logits), ("target_logits", true_logits)):
            write_atomic(
                table_dir / f"{name}.npy",
                lambda tmp, logits=logits: save_numpy(tmp, logits.to(torch.float32).numpy()),
            )
    approx_logits = coefficients @ basis_logits

    print(
        f"Finite-table baseline: {num_targets} target histories, {num_basis} basis histories, "
        f"{num_futures} futures x top-{top_k} tokens"
    )
    metrics, per_target = summarize_next_token_approximation(
        approx_logits, true_logits, top_k=eval_top_k,
        output_path=output_dir / "next_token_approximation.png",
    )
    # The table is close to rank one, so a residual relative to ||r_t||^2 looks
    # small for any fit. The second ratio is relative to predicting the mean basis row.
    table64 = torch.as_tensor(table, dtype=torch.float64)
    target_rows64 = torch.as_tensor(target_rows, dtype=torch.float64)
    residual = (coefficients @ table64 - target_rows64).square().sum(dim=-1)
    for name, values in (
        ("table_relative_residual", residual / target_rows64.square().sum(dim=-1)),
        ("table_residual_vs_mean_row",
         residual / (target_rows64 - table64.mean(dim=0)).square().sum(dim=-1)),
        ("coefficient_sum", coefficients.sum(dim=-1)),
        ("coefficient_l1", coefficients.abs().sum(dim=-1)),
    ):
        per_target[name] = values
        metrics[name] = values.mean().item()
        metrics[f"{name}_std"] = values.std(unbiased=False).item()
        print(f"  {name}: mean={metrics[name]:.6g}, std={metrics[f'{name}_std']:.6g}")

    # Reference on the same targets and basis: no fitting, every basis history
    # weighted equally.
    print("Reference uniform_mean (unweighted mean of the basis logits):")
    uniform_mean, _ = summarize_next_token_approximation(
        basis_logits.mean(dim=0).expand(num_targets, -1), true_logits,
        top_k=eval_top_k, output_path=None,
    )
    references = {"uniform_mean": uniform_mean}

    result = {
        "config": {
            "store_dir": str(store_dir), "model_path": str(model_path), "device": device,
            "seed": seed, "num_basis": num_basis, "num_futures": num_futures,
            "top_k": top_k, "num_targets": num_targets, "eval_top_k": eval_top_k,
            "table_shape": list(table.shape),
            "table_dir": None if table_dir is None else str(table_dir),
            "table_value": "logit(token | history + future) - mean(logits over full vocabulary)",
            "header_context": "future_only",
            "fit": "float64 minimum-norm least squares of each target row on the basis rows",
            "evaluation": "next-token logits of the target history vs. "
                          "coefficient-weighted next-token logits of the basis histories",
        },
        "summary": metrics,
        "references": references,
        "per_target": {name: values.tolist() for name, values in per_target.items()},
    }
    metrics_path = output_dir / "metrics.json"
    write_atomic(metrics_path, lambda tmp: save_json(tmp, result))
    print(f"Saved metrics to {metrics_path}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finite_table", description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--table-dir", type=Path, default=DEFAULT_TABLE_DIR,
        help="Where the sampled tokens, table and evaluated logits are saved",
    )
    parser.add_argument("--num-basis", type=int, default=8192)
    parser.add_argument("--num-futures", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-targets", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Sequences per GPT-2 forward pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Defaults to cuda when available")
    report_finite_table_baseline(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
