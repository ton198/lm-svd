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
from utils.common import Sample, save_json, write_atomic
from utils.inference import CausalLMInference

from .lingen import summarize_next_token_approximation

DEFAULT_STORE_DIR = Path("data/sample_stores/wikitext-103-train__gpt2-small__h64-f32")
DEFAULT_MODEL_PATH = Path("models/gpt2-small")
DEFAULT_OUTPUT_DIR = Path(
    "data/experiment_results/svd_inference_finite_table/wikitext-103-train__gpt2-small/analysis"
)
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


def spectral_fit(table, target_rows) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the least-squares fit into one term per singular direction of the table.

    With the thin SVD M = U diag(S) Vh, the minimum-norm solution of
    `fit_coefficients` is

        v_t = sum_k weights[t, k] * U[:, k],   weights[t, k] = (r_t . Vh[k]) / S[k]

    Return U of shape (N, K) and weights of shape (T, K). Summing the first k
    terms fits the rank-k truncation of the table. Summing all but the first
    fits the deflate1 table, so deflating and then restoring the leading term
    gives back the ordinary fit unchanged. Directions at rounding level get
    zero weight.
    """
    table = torch.as_tensor(table, dtype=torch.float64)
    target_rows = torch.as_tensor(target_rows, dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(table, full_matrices=False)
    tolerance = S.max() * max(table.shape) * torch.finfo(torch.float64).eps
    weights = (target_rows @ Vh.T) / S
    return U, weights.masked_fill(S <= tolerance, 0.0)


def deflated_relative_error(coefficients, basis_logits, target_logits) -> torch.Tensor:
    """Relative squared logit error with the leading basis direction removed.

    Center the logits over the vocabulary, take the leading right singular
    direction b of the centered basis logits, and project it out of both sides:

        ||P (v L - l_t)||^2 / ||P l_t||^2,   P = I - b b^T

    This is the evaluation-side counterpart of deflate1. Zero coefficients
    score exactly one. Coefficients of shape (..., T, N) give errors of shape
    (..., T), so several fits can share one SVD.
    """
    coefficients = torch.as_tensor(coefficients, dtype=torch.float64)
    basis = torch.as_tensor(basis_logits, dtype=torch.float64)
    target = torch.as_tensor(target_logits, dtype=torch.float64)
    basis = basis - basis.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    direction = torch.linalg.svd(basis, full_matrices=False).Vh[0]

    def deflate(rows: torch.Tensor) -> torch.Tensor:
        return rows - (rows @ direction).unsqueeze(-1) * direction

    error = deflate(coefficients @ basis - target).square().sum(dim=-1)
    return error / deflate(target).square().sum(dim=-1)


@torch.no_grad()
def report_finite_table_baseline(
    *,
    store_dir: str | Path = DEFAULT_STORE_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    num_basis: int = 1024,
    num_futures: int = 1024,
    top_k: int = 50,
    num_targets: int = 32,
    batch_size: int = 1024,
    seed: int = 0,
    device: str | None = None,
) -> dict[str, float]:
    """Run the baseline; print and save mean/std metrics over target histories."""
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

    coefficients, _ = fit_coefficients(table, target_rows)

    # LM(x): full-vocabulary next-token logits after the history alone.
    basis_logits = torch.cat([
        gpt_model.next_token_logits_batch(basis[start : start + batch_size].tolist())
        for start in range(0, num_basis, batch_size)
    ]).to(device="cpu", dtype=torch.float64)
    true_logits = gpt_model.next_token_logits_batch(targets.tolist()).to(
        device="cpu", dtype=torch.float64
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
    mean_row_error = (target_rows64 - table64.mean(dim=0)).square().sum(dim=-1)

    def fit_diagnostics(fitted: torch.Tensor) -> dict[str, torch.Tensor]:
        residual = (fitted @ table64 - target_rows64).square().sum(dim=-1)
        return {
            "table_relative_residual": residual / target_rows64.square().sum(dim=-1),
            "table_residual_vs_mean_row": residual / mean_row_error,
            "coefficient_sum": fitted.sum(dim=-1),
            "coefficient_l1": fitted.abs().sum(dim=-1),
        }

    for name, values in fit_diagnostics(coefficients).items():
        per_target[name] = values
        metrics[name] = values.mean().item()
        metrics[f"{name}_std"] = values.std(unbiased=False).item()
        print(f"  {name}: mean={metrics[name]:.6g}, std={metrics[f'{name}_std']:.6g}")

    # Two reference points on the same targets and basis. Uniform mean uses no
    # fitting at all. The oracle fits the coefficients on the very logits being
    # evaluated, so it bounds what any coefficients on this basis can reach.
    centered_basis = basis_logits - basis_logits.mean(dim=-1, keepdim=True)
    centered_true = true_logits - true_logits.mean(dim=-1, keepdim=True)
    oracle_coefficients, _ = fit_coefficients(centered_basis, centered_true)
    references = {}
    for name, description, reference_logits in (
        ("uniform_mean", "unweighted mean of the basis logits",
         basis_logits.mean(dim=0).expand(num_targets, -1)),
        ("oracle_full_vocab_fit", "coefficients fitted on the evaluated logits themselves",
         oracle_coefficients @ basis_logits),
    ):
        print(f"Reference {name} ({description}):")
        references[name], _ = summarize_next_token_approximation(
            reference_logits, true_logits, top_k=eval_top_k, output_path=None
        )

    # Rank sweep: coefficients from only the first k singular directions of the
    # table, i.e. the least-squares fit to its rank-k truncation. k = num_basis
    # is the ordinary fit above.
    basis_directions, weights = spectral_fit(table64, target_rows64)
    ranks = [rank for rank in (2**power for power in range(32)) if rank < num_basis]
    rank_sweep = {}
    for rank in (*ranks, num_basis):
        fitted = weights[:, :rank] @ basis_directions[:, :rank].T
        print(f"Rank {rank} of {num_basis}:")
        rank_sweep[rank], _ = summarize_next_token_approximation(
            fitted @ basis_logits, true_logits, top_k=eval_top_k, output_path=None
        )
        for name, values in fit_diagnostics(fitted).items():
            rank_sweep[rank][name] = values.mean().item()
    print("Rank sweep (means over targets):")
    print("  rank  KL(true||approx)  centered_rmse  top1  top10  table_res_vs_mean  coef_l1")
    for rank, row in rank_sweep.items():
        print(
            f"  {rank:4d}  {row['kl_true_to_approx']:16.4f}  {row['centered_logit_rmse']:13.4f}"
            f"  {row['top1_agreement']:4.2f}  {row[f'top{eval_top_k}_overlap']:5.2f}"
            f"  {row['table_residual_vs_mean_row']:17.4f}  {row['coefficient_l1']:7.2f}"
        )

    # deflate1: least-squares fit on M - S[0] * outer(U[:, 0], Vt[0]), which is
    # the ordinary fit without its leading-direction term. These coefficients
    # sum to about zero, so on their own they describe only how a target
    # deviates from what all histories share.
    leading = weights[:, :1] @ basis_directions[:, :1].T
    deflated = weights[:, 1:] @ basis_directions[:, 1:].T
    deflate1 = {
        "fit": "least squares on M - S[0] * outer(U[:, 0], Vt[0]), M = U @ diag(S) @ Vt",
        "restore_max_abs_coefficient_difference": (
            deflated + leading - coefficients
        ).abs().max().item(),
        "leading_term_coefficient_l1": leading.abs().sum(dim=-1).mean().item(),
    }
    for name, values in fit_diagnostics(deflated).items():
        if name.startswith("coefficient"):
            per_target[f"deflate1_{name}"] = values
            deflate1[name] = values.mean().item()
    for name, description, deflate1_logits in (
        ("fit_only", "sum_i v'_i LM(h_i)", deflated @ basis_logits),
        ("fit_plus_basis_mean", "mean_i LM(h_i) + sum_i v'_i LM(h_i)",
         basis_logits.mean(dim=0) + deflated @ basis_logits),
    ):
        print(f"deflate1 {name} ({description}):")
        deflate1[name], _ = summarize_next_token_approximation(
            deflate1_logits, true_logits, top_k=eval_top_k, output_path=None
        )
    uniform = torch.full_like(coefficients, 1 / num_basis)
    deflated_errors = deflated_relative_error(
        torch.stack((deflated, coefficients, uniform)), basis_logits, true_logits
    ).mean(dim=-1)
    deflate1["deflated_logit_relative_error"] = dict(zip(
        ("deflate1_fit", "ordinary_fit", "uniform_mean"), deflated_errors.tolist()
    ))
    print(
        "deflate1: relative squared logit error with the leading basis-logit direction "
        "removed from both sides (zero coefficients score 1):"
    )
    for name, value in deflate1["deflated_logit_relative_error"].items():
        print(f"  {name}: {value:.6g}")
    print(
        "deflate1 fit plus leading term vs ordinary fit, max |coefficient difference|: "
        f"{deflate1['restore_max_abs_coefficient_difference']:.3g}"
    )

    result = {
        "config": {
            "store_dir": str(store_dir), "model_path": str(model_path), "device": device,
            "seed": seed, "num_basis": num_basis, "num_futures": num_futures,
            "top_k": top_k, "num_targets": num_targets, "eval_top_k": eval_top_k,
            "table_shape": list(table.shape),
            "table_value": "logit(token | history + future) - mean(logits over full vocabulary)",
            "header_context": "future_only",
            "fit": "float64 minimum-norm least squares of each target row on the basis rows",
            "evaluation": "next-token logits of the target history vs. "
                          "coefficient-weighted next-token logits of the basis histories",
        },
        "summary": metrics,
        "references": references,
        "rank_sweep": rank_sweep,
        "deflate1": deflate1,
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
    parser.add_argument("--num-basis", type=int, default=1024)
    parser.add_argument("--num-futures", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-targets", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Sequences per GPT-2 forward pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Defaults to cuda when available")
    report_finite_table_baseline(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
