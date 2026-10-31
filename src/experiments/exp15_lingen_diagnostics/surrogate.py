from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.exp14_paper_lingen.lingen import neural_svd_coefficients, neural_svd_system
from experiments.exp6_stream_gpt2_mlp.features import build_features
from utils.stream_neural_svd import centered_unembedding, load_trained_features

from .diagnostics import (
    accumulate_metrics,
    finalize_metrics,
    new_metric_totals,
    prepare_output_dir,
    teacher_hidden_grid,
    write_outputs,
)

FUTURE_LENGTHS = (0, 1, 4, 32)
ANCHOR_LENGTHS = (1, 4, 32)
VARIANTS = ("finite_saved", "single_saved", "nsvd_saved", "nsvd_current", "anchor_full", "anchor_nsvd")
DECOMPOSED = ("nsvd_saved", "nsvd_current", "anchor_nsvd", "finite_saved")


def anchored_constrained_solve(A: np.ndarray, B: np.ndarray, constraint: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """Minimize anchor error over the affine space v0 + null(constraint)."""
    _, singular, vh = np.linalg.svd(constraint, full_matrices=True)
    tolerance = singular[0] * max(constraint.shape) * np.finfo(np.float64).eps if singular.size else 0.0
    rank = int(np.count_nonzero(singular > tolerance))
    null = vh[rank:].T
    if null.shape[1] == 0:
        return v0.copy()
    z = np.linalg.lstsq(A @ null, B - A @ v0.T, rcond=None)[0]
    return v0 + (null @ z).T


def _root(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * values.clamp_min(0).sqrt()) @ vectors.T, values


def _squared(x: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    return ((x @ gram) * x).sum(dim=-1)


def _feature_summary(v: torch.Tensor, u_h: torch.Tensor, u_t: torch.Tensor, gram: torch.Tensor) -> dict:
    residual = u_t - v @ u_h
    ratio = _squared(residual, gram) / _squared(u_t, gram)
    return {"mean": ratio.mean().item(), "max": ratio.max().item()}


@torch.no_grad()
def run_exp14_surrogate(*, fit_dir: Path, nsvd_train_dir: Path, device: str, pair_batch: int, gram_batch: int, eval_futures: int, anchor_futures: int, output_dir: Path, overwrite: bool) -> str:
    started = time.monotonic()
    fit_dir, nsvd_train_dir, output_dir = map(Path, (fit_dir, nsvd_train_dir, output_dir))
    prepare_output_dir(output_dir, ("summary.json", "residuals.png", "metadata.json"), overwrite)
    for name in ("metadata.json", "sets.npz", "coefficients.npz"):
        path = fit_dir / name
        if not path.is_file():
            raise ValueError(f"fit output does not exist: {path}")
    fit_meta = json.loads((fit_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(fit_dir / "sets.npz") as archive:
        sets = {key: archive[key] for key in archive.files}
    with np.load(fit_dir / "coefficients.npz") as archive:
        saved = {key: archive[key] for key in archive.files if key in {f"{method}_{opt}" for method in ("finite", "single", "nsvd") for opt in ("opt1", "opt2")}}
    train_meta, inference, features = load_trained_features(nsvd_train_dir, build_features, device)
    if fit_meta["nsvd"]["features_config"] != train_meta["features"]:
        raise ValueError("current NeuralSVD features differ from the exp14 fit")
    n_fit = len(sets["futures"])
    if len(sets["gram_futures"]) < n_fit + eval_futures:
        raise ValueError(f"need {n_fit + eval_futures} gram futures but the fit has {len(sets['gram_futures'])}")
    if anchor_futures > n_fit:
        raise ValueError(f"need {anchor_futures} anchor futures but the fit has {n_fit}")
    token_matrix = centered_unembedding(inference).double()
    token_gram = token_matrix.T @ token_matrix
    gram_root, _ = _root(token_gram)
    target_ids = torch.as_tensor(sets["targets"], dtype=torch.long, device=inference.device)
    eval_ids = torch.as_tensor(sets["gram_futures"][n_fit:n_fit + eval_futures], dtype=torch.long, device=inference.device)
    summary = {}

    for option in ("opt1", "opt2"):
        suffix = "" if option == "opt1" else "_nonsense"
        basis_ids = torch.as_tensor(sets["basis" + suffix], dtype=torch.long, device=inference.device)
        anchor_ids = torch.as_tensor(sets["futures" + suffix][:anchor_futures], dtype=torch.long, device=inference.device)
        gram_ids = sets["gram_futures" + suffix]
        u_h, u_t, gram_w = neural_svd_system(features, token_matrix, basis_ids, target_ids, gram_ids, batch=gram_batch)
        root_w, eig = _root(gram_w)
        constraint = (root_w @ u_h.T).cpu().numpy()
        c_rank = int(np.linalg.matrix_rank(constraint))
        singular_h = torch.linalg.svdvals(u_h).cpu().numpy()
        eig_np = eig.cpu().numpy()
        gram_rank = int(np.count_nonzero(eig_np > max(float(eig_np[-1]), 0.0) * len(eig_np) * np.finfo(np.float64).eps))
        h_rank = int(np.count_nonzero(singular_h > singular_h[0] * max(u_h.shape) * np.finfo(np.float64).eps))
        geometry = {
            "gram_w": {"min": float(eig_np[0]), "max": float(eig_np[-1]), "numerical_rank": gram_rank},
            "u_h": {"min": float(singular_h[-1]), "max": float(singular_h[0]), "numerical_rank": h_rank},
            "constraint_numerical_rank": c_rank,
            "basis_histories": len(basis_ids),
            "null_dimension": len(basis_ids) - c_rank,
        }
        current, _ = neural_svd_coefficients(features, token_matrix, basis_ids, target_ids, gram_ids, batch=gram_batch)
        # Materialize the dense anchor design only on GPU before transferring it to CPU.
        rows = len(ANCHOR_LENGTHS) * anchor_futures * token_gram.shape[0]
        A_gpu = torch.empty((rows, len(basis_ids)), dtype=torch.float64, device=inference.device)
        B_gpu = torch.empty((rows, len(target_ids)), dtype=torch.float64, device=inference.device)
        position = 0
        for length in ANCHOR_LENGTHS:
            basis_states = teacher_hidden_grid(inference, basis_ids, anchor_ids[:, :length], pair_batch=pair_batch)
            target_states = teacher_hidden_grid(inference, target_ids, anchor_ids[:, :length], pair_batch=pair_batch)
            for xb, xt in zip(basis_states, target_states):
                end = position + token_gram.shape[0]
                A_gpu[position:end] = gram_root @ xb.double().T
                B_gpu[position:end] = gram_root @ xt.double().T
                position = end
            del basis_states, target_states
        A, B = A_gpu.cpu().numpy(), B_gpu.cpu().numpy()
        del A_gpu, B_gpu
        full = np.linalg.lstsq(A, B, rcond=None)[0].T
        anchored = anchored_constrained_solve(A, B, constraint, current)
        variants = {f"{method}_saved": saved[f"{method}_{option}"] for method in ("finite", "single", "nsvd")}
        variants.update(nsvd_current=current, anchor_full=full, anchor_nsvd=anchored)
        result_variants = {}
        for name, value in variants.items():
            v = torch.as_tensor(value, dtype=torch.float64, device=inference.device)
            result_variants[name] = {"feature_residual": _feature_summary(v, u_h, u_t, gram_w), "evaluation": {}}
            if name in ("anchor_full", "anchor_nsvd", "nsvd_current"):
                residual = A @ value.T - B
                result_variants[name]["anchor_residual"] = {"mean": float(((residual ** 2).sum(axis=0) / (B ** 2).sum(axis=0)).mean())}
        del A, B, full, anchored, current, constraint
        v_tensors = {name: torch.as_tensor(value, dtype=torch.float64, device=inference.device) for name, value in variants.items()}
        decomposition = {}
        for length in FUTURE_LENGTHS:
            futures = eval_ids[:, :length]
            basis_states = teacher_hidden_grid(inference, basis_ids, futures, pair_batch=pair_batch)
            target_states = teacher_hidden_grid(inference, target_ids, futures, pair_batch=pair_batch)
            totals = {name: new_metric_totals() for name in VARIANTS}
            denom = torch.zeros(len(target_ids), dtype=torch.float64, device=inference.device)
            errors = {name: torch.zeros_like(denom) for name in VARIANTS}
            surrogate_errors = {name: torch.zeros_like(denom) for name in DECOMPOSED}
            model_errors = {name: torch.zeros_like(denom) for name in DECOMPOSED}
            target_recon = torch.zeros_like(denom)
            basis_num = torch.zeros(len(basis_ids), dtype=torch.float64, device=inference.device)
            basis_denom = torch.zeros_like(basis_num)
            for index in range(eval_futures):
                j = index if length else 0
                xb, xt = basis_states[j].double(), target_states[j].double()
                denom += _squared(xt, token_gram)
                if length:
                    weights = features.column_weights(futures[index:index + 1]).double()[0]
                    target_recon += _squared(xt - u_t @ weights, token_gram)
                    basis_num += _squared(xb - u_h @ weights, token_gram)
                    basis_denom += _squared(xb, token_gram)
                for name, v in v_tensors.items():
                    prediction = v @ xb
                    residual = xt - prediction
                    errors[name] += _squared(residual, token_gram)
                    for start in range(0, len(target_ids), pair_batch):
                        accumulate_metrics(totals[name], xt[start:start + pair_batch], prediction[start:start + pair_batch], token_matrix)
                    if length and name in DECOMPOSED:
                        surrogate = (u_t - v @ u_h) @ weights
                        surrogate_errors[name] += _squared(surrogate, token_gram)
                        model_errors[name] += _squared(residual - surrogate, token_gram)
            for name in VARIANTS:
                relative = (errors[name] / denom).cpu().numpy()
                result_variants[name]["evaluation"][str(length)] = {
                    "relative_residual": {"mean": float(relative.mean()), "median": float(np.median(relative))},
                    "metrics": finalize_metrics(totals[name]),
                }
            if length:
                decomposition[str(length)] = {
                    "target_reconstruction": float((target_recon / denom).mean().item()),
                    "basis_reconstruction": float((basis_num / basis_denom).mean().item()),
                    "variants": {name: {"surrogate_part": float((surrogate_errors[name] / denom).mean().item()), "model_error_part": float((model_errors[name] / denom).mean().item())} for name in DECOMPOSED},
                }
            del basis_states, target_states
        summary[option] = {"geometry": geometry, "variants": result_variants, "decomposition": decomposition}
        del v_tensors, variants, u_h, u_t, gram_w

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, option in zip(axes, ("opt1", "opt2")):
        for name in VARIANTS:
            ax.plot(FUTURE_LENGTHS, [summary[option]["variants"][name]["evaluation"][str(length)]["relative_residual"]["mean"] for length in FUTURE_LENGTHS], marker="o", label=name)
        ax.set_yscale("log")
        ax.set_xticks(FUTURE_LENGTHS)
        ax.set_xlabel("Future length")
        ax.set_ylabel("Relative residual (mean)")
        ax.set_title(option)
        ax.legend(fontsize="small")
    metadata = {
        "fit_dir": str(fit_dir), "nsvd_train_dir": str(nsvd_train_dir), "device": device,
        "pair_batch": pair_batch, "gram_batch": gram_batch, "eval_futures": eval_futures,
        "anchor_futures": anchor_futures, "output_dir": str(output_dir), "overwrite": overwrite,
        "fit_snapshot_steps": fit_meta["nsvd"]["train_steps"], "current_snapshot_steps": train_meta["steps"],
        "FUTURE_LENGTHS": list(FUTURE_LENGTHS), "ANCHOR_LENGTHS": list(ANCHOR_LENGTHS),
        "definitions": {
            "token_gram": "G = E_centered.T @ E_centered",
            "constraint": "C = sqrt(gram_w) @ u_h.T; C @ v.T = sqrt(gram_w) @ u_t.T",
            "feature_residual": "(u_t - v @ u_h) gram_w (u_t - v @ u_h).T / (u_t gram_w u_t.T)",
            "anchor_residual": "||A @ v.T - B||^2 / ||B||^2 per target; A = stack(sqrt(G) @ X_a.T), B = stack(sqrt(G) @ x_a.T)",
            "relative_residual": "sum_f ||x_t,f - v @ X_f||_G^2 / sum_f ||x_t,f||_G^2 per target",
            "surrogate_part": "sum_f ||(u_t - v @ u_h) @ W_f||_G^2 / sum_f ||x_t,f||_G^2",
            "model_error_part": "sum_f ||(x_t,f - v @ X_f) - (u_t - v @ u_h) @ W_f||_G^2 / sum_f ||x_t,f||_G^2",
            "reconstruction": "sum_f ||x_f - u @ W_f||_G^2 / sum_f ||x_f||_G^2 per history",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_outputs(output_dir, summary, figure, metadata, plot_name="residuals.png")
    lines = ["| option | variant | feature residual | anchor residual | L0 | L1 | L4 | L32 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for option in ("opt1", "opt2"):
        for name in VARIANTS:
            item = summary[option]["variants"][name]
            anchor = f"{item['anchor_residual']['mean']:.4g}" if "anchor_residual" in item else "-"
            lengths = " | ".join(f"{item['evaluation'][str(length)]['relative_residual']['mean']:.4g}" for length in FUTURE_LENGTHS)
            lines.append(f"| {option} | {name} | {item['feature_residual']['mean']:.4g} | {anchor} | {lengths} |")
    return "\n".join(lines)
