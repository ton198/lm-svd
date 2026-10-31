from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.experiment_paths import token_stream_path
from utils.inference import CausalLMInference
from utils.stream_neural_svd import TokenStream, centered_unembedding, load_trained_features

from .diagnostics import (
    accumulate_metrics,
    finalize_metrics,
    fit_diagonal_scale,
    fit_temperature,
    history_gram,
    new_metric_totals,
    optimal_left,
    orthonormal_basis,
    prepare_output_dir,
    teacher_hidden_grid,
    write_outputs,
)


# Distribution comparisons create several vocabulary-sized arrays per pair.
_METRIC_BATCH = 128
_VALIDATION_BATCH = 1024
_LAYER_NAMES = (
    "oracle_at_own_rank", "row_span_refit", "column_refit", "model",
    "calibrated_scale", "calibrated_scale_temperature",
)


def _feature_rows(features, ids: torch.Tensor, device: torch.device, batch: int) -> torch.Tensor:
    return torch.cat([
        features.row_features(ids[start:start + batch].to(device)).double()
        for start in range(0, len(ids), batch)
    ])


def _feature_weights(features, ids: torch.Tensor, device: torch.device, batch: int) -> torch.Tensor:
    return torch.cat([
        features.column_weights(ids[start:start + batch].to(device)).float()
        for start in range(0, len(ids), batch)
    ])


def _accumulate_batched(totals, teacher, approx, tokens, *, temperature=1.0):
    for start in range(0, len(teacher), _METRIC_BATCH):
        end = start + _METRIC_BATCH
        accumulate_metrics(totals, teacher[start:end], approx[start:end], tokens, temperature=temperature)


def _validation_data(stream, teacher, history_tokens, future_tokens, count, seed, pair_batch):
    result = {}
    generator = torch.Generator().manual_seed(seed)
    histories = stream.windows(stream.sample_starts(count, history_tokens, generator), history_tokens)
    futures = stream.windows(stream.sample_starts(count, future_tokens, generator), future_tokens)
    result["independent"] = (histories, futures)
    generator = torch.Generator().manual_seed(seed + 1)
    windows = stream.windows(
        stream.sample_starts(count, history_tokens + future_tokens, generator),
        history_tokens + future_tokens,
    )
    result["contiguous"] = (windows[:, :history_tokens], windows[:, history_tokens:])
    for condition, (histories, futures) in result.items():
        states = torch.cat([
            teacher.final_hidden_states(torch.cat((histories[start:start + pair_batch], futures[start:start + pair_batch]), dim=1).to(teacher.device))[:, -1]
            for start in range(0, count, pair_batch)
        ])
        result[condition] = (histories, futures, states)
    return result


def _plot(summary, experiments):
    figure, axes = plt.subplots(1, 2, figsize=(max(10, len(experiments) * 1.8), 5), constrained_layout=True)
    x = np.arange(len(experiments))
    width = 0.8 / len(_LAYER_NAMES)
    for index, layer in enumerate(_LAYER_NAMES):
        values = []
        for name in experiments:
            experiment = summary["probe"]["experiments"][name]
            values.append(summary["probe"]["oracle"][str(experiment["meta"]["rank"])] if index == 0 else experiment["layers"][layer])
        positions = x + (index - (len(_LAYER_NAMES) - 1) / 2) * width
        axes[0].bar(positions, [value["kl_pq"] for value in values], width, label=layer)
        axes[1].bar(positions, [value["logit_relative_error_aggregate"] for value in values], width, label=layer)
    for axis, label in zip(axes, ("KL(p || q)", "Aggregate relative logit error")):
        axis.set_xticks(x, experiments, rotation=45, ha="right")
        axis.set_ylabel(label)
    axes[1].legend(fontsize="small")
    return figure


def _table(summary, experiments):
    headers = ("experiment", "rank", "steps", "oracle KL", "row_span KL", "column KL", "model KL", "scale KL", "scale+τ KL", "val-indep model KL", "val-contig model KL")
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for name in experiments:
        experiment = summary["probe"]["experiments"][name]
        meta, layers = experiment["meta"], experiment["layers"]
        scores = [summary["probe"]["oracle"][str(meta["rank"])]["kl_pq"]]
        scores.extend(layers[layer]["kl_pq"] for layer in _LAYER_NAMES[1:])
        scores.extend(summary["validation"][condition][name]["model"]["kl_pq"] for condition in ("independent", "contiguous"))
        lines.append(" | ".join([name, str(meta["rank"]), str(meta["train_steps"]), *(f"{score:.4f}" for score in scores)]))
    return "\n".join(lines)


@torch.no_grad()
def run_direct(
    *, runs: dict[str, tuple[Path, str]], reference_metadata: Path | None,
    stream: str, eval_stream: str, model_path: Path, device: str, pair_batch: int,
    calibration_histories: int, calibration_futures: int, calibration_seed: int,
    tau_futures: int, validation_pairs: int, validation_seed: int,
    output_dir: Path, overwrite: bool,
) -> str:
    started = time.monotonic()
    if tau_futures > calibration_futures:
        raise ValueError(f"tau futures {tau_futures} exceed calibration futures {calibration_futures}")
    experiments = list(runs)
    for name in experiments:
        train_dir, _ = runs[name]
        for filename in ("metadata.json", "features.pt", "probe.npz"):
            path = train_dir / filename
            if not path.is_file():
                raise ValueError(f"training input does not exist: {path}")
    prepare_output_dir(output_dir, ("summary.json", "layers.png", "metadata.json"), overwrite)

    metadata_by_name = {}
    probe_history_starts = probe_future_starts = None
    for name in experiments:
        train_dir, _ = runs[name]
        meta = json.loads((train_dir / "metadata.json").read_text(encoding="utf-8"))
        metadata_by_name[name] = meta
        with np.load(train_dir / "probe.npz") as probe:
            history_starts, future_starts = probe["history_starts"].copy(), probe["future_starts"].copy()
        if name == experiments[0]:
            probe_history_starts, probe_future_starts = history_starts, future_starts
        else:
            if not (np.array_equal(history_starts, probe_history_starts) and np.array_equal(future_starts, probe_future_starts)):
                raise ValueError(f"probe.npz differs between {experiments[0]} and {name}; diagnostics require a shared probe")
        base = metadata_by_name[experiments[0]]
        for key in ("history_tokens", "future_tokens", "stream_path"):
            if meta[key] != base[key]:
                raise ValueError(f"{name} training metadata is incompatible with {experiments[0]}: {key}")
        if Path(meta["model_path"]).name != model_path.name:
            raise ValueError(f"{name} training metadata is incompatible with {experiments[0]}: model_path")

    base = metadata_by_name[experiments[0]]
    teacher = CausalLMInference(Path(base["model_path"]), device)
    tokens = centered_unembedding(teacher).double()
    token_gram = tokens.T @ tokens
    train_stream = TokenStream(token_stream_path(stream))
    eval_data = TokenStream(token_stream_path(eval_stream))
    hlen, flen = base["history_tokens"], base["future_tokens"]
    histories = train_stream.windows(torch.from_numpy(probe_history_starts), hlen)
    futures = train_stream.windows(torch.from_numpy(probe_future_starts), flen)
    states = teacher_hidden_grid(teacher, histories, futures, pair_batch=pair_batch)
    generator = torch.Generator().manual_seed(calibration_seed)
    cal_histories = train_stream.windows(train_stream.sample_starts(calibration_histories, hlen, generator), hlen)
    cal_futures = train_stream.windows(train_stream.sample_starts(calibration_futures, flen, generator), flen)
    cal_states = teacher_hidden_grid(teacher, cal_histories, cal_futures, pair_batch=pair_batch)
    validation = _validation_data(eval_data, teacher, hlen, flen, validation_pairs, validation_seed, pair_batch)

    eigenvalues, eigenvectors = torch.linalg.eigh(history_gram(states, token_gram))
    eigenvalues, eigenvectors = eigenvalues.flip(0), eigenvectors.flip(1)
    singular_values = (eigenvalues.clamp_min(0) / (len(histories) * len(futures) * tokens.shape[0])).sqrt()
    reference_difference = None
    if reference_metadata is not None and reference_metadata.is_file():
        analysis = json.loads(reference_metadata.read_text(encoding="utf-8"))
        reference = np.asarray(analysis["reference"]["singular_values"], dtype=np.float64)
        count = min(len(reference), len(singular_values))
        actual = singular_values[:count].cpu().numpy()
        reference_difference = float(np.max(np.abs(actual - reference[:count]) / reference[:count]))
    ranks = sorted({meta["rank"] for meta in metadata_by_name.values()})
    summary = {"probe": {"oracle": {}, "experiments": {}}, "validation": {"independent": {}, "contiguous": {}}}
    for rank in ranks:
        basis = eigenvectors[:, :rank]
        totals = new_metric_totals()
        for state in states:
            state = state.double()
            _accumulate_batched(totals, state, basis @ (basis.T @ state), tokens)
        summary["probe"]["oracle"][str(rank)] = finalize_metrics(totals)

    for name in experiments:
        train_dir, features_module = runs[name]
        train_meta, feature_inference, features = load_trained_features(
            train_dir, importlib.import_module(features_module).build_features, device,
        )
        rows = _feature_rows(features, histories, teacher.device, pair_batch)
        weights = _feature_weights(features, futures, teacher.device, pair_batch)
        basis = torch.as_tensor(orthonormal_basis(rows.cpu().numpy()), device=teacher.device)
        left = torch.as_tensor(optimal_left(weights, states, token_gram), device=teacher.device)
        cal_rows = _feature_rows(features, cal_histories, teacher.device, pair_batch)
        cal_weights = _feature_weights(features, cal_futures, teacher.device, pair_batch)
        scale = torch.as_tensor(fit_diagonal_scale(cal_rows, cal_weights, cal_states, token_gram), device=teacher.device)
        scaled_cal_rows = cal_rows * scale

        def teacher_logits_blocks():
            for state in cal_states[:tau_futures]:
                for start in range(0, len(state), _METRIC_BATCH):
                    yield state[start:start + _METRIC_BATCH].double() @ tokens.T

        def approx_logits_blocks():
            for weight in cal_weights[:tau_futures]:
                weight = weight.double()
                for start in range(0, len(cal_rows), _METRIC_BATCH):
                    yield (scaled_cal_rows[start:start + _METRIC_BATCH] @ weight) @ tokens.T

        temperature = fit_temperature(teacher_logits_blocks, approx_logits_blocks)
        layer_totals = {layer: new_metric_totals() for layer in _LAYER_NAMES[1:]}
        for state, weight in zip(states, weights):
            state, weight = state.double(), weight.double()
            model = rows @ weight
            scaled = (rows * scale) @ weight
            approximations = {
                "row_span_refit": basis @ (basis.T @ state),
                "column_refit": left @ weight,
                "model": model,
                "calibrated_scale": scaled,
                "calibrated_scale_temperature": scaled,
            }
            for layer, approx in approximations.items():
                _accumulate_batched(layer_totals[layer], state, approx, tokens, temperature=temperature if layer == "calibrated_scale_temperature" else 1.0)
        summary["probe"]["experiments"][name] = {
            "meta": {
                "run_dir": str(train_dir.parent), "train_steps": train_meta["steps"],
                "converged": train_meta["converged"], "rank": train_meta["rank"],
                "features": train_meta["features"], "row_span_rank": basis.shape[1],
                "scale": {"min": scale.min().item(), "max": scale.max().item(), "mean": scale.mean().item()},
                "temperature": temperature,
            },
            "layers": {layer: finalize_metrics(totals) for layer, totals in layer_totals.items()},
        }
        for condition, (val_histories, val_futures, val_states) in validation.items():
            totals = {layer: new_metric_totals() for layer in ("model", "calibrated_scale", "calibrated_scale_temperature")}
            for start in range(0, validation_pairs, _VALIDATION_BATCH):
                end = start + _VALIDATION_BATCH
                val_rows = _feature_rows(features, val_histories[start:end], teacher.device, pair_batch)
                val_weights = _feature_weights(features, val_futures[start:end], teacher.device, pair_batch).double()
                model = torch.einsum("br,brd->bd", val_rows, val_weights)
                scaled = torch.einsum("br,brd->bd", val_rows * scale, val_weights)
                for layer, approx in (("model", model), ("calibrated_scale", scaled), ("calibrated_scale_temperature", scaled)):
                    _accumulate_batched(totals[layer], val_states[start:end], approx, tokens, temperature=temperature if layer == "calibrated_scale_temperature" else 1.0)
            summary["validation"][condition][name] = {layer: finalize_metrics(value) for layer, value in totals.items()}
        del features, feature_inference
        if teacher.device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "train_dirs": {name: str(train_dir) for name, (train_dir, _) in runs.items()},
        "reference_metadata": str(reference_metadata) if reference_metadata is not None else None,
        "stream": stream, "eval_stream": eval_stream,
        "model_path": str(model_path), "experiments": experiments, "device": device,
        "pair_batch": pair_batch, "calibration_histories": calibration_histories,
        "calibration_futures": calibration_futures, "calibration_seed": calibration_seed,
        "tau_futures": tau_futures, "validation_pairs": validation_pairs,
        "validation_seed": validation_seed, "output_dir": str(output_dir), "overwrite": overwrite,
        "probe_shape": [len(histories), len(futures)],
        "calibration_shape": [calibration_histories, calibration_futures], "ranks": ranks,
        "reference_check_max_relative_difference": reference_difference,
        "definitions": {
            "token_matrix": "Etilde = centered_unembedding(teacher); G = Etilde.T @ Etilde",
            "teacher_hidden": "x(c) = teacher.final_hidden_states(c)[:, -1]",
            "teacher_logits": "x @ Etilde.T",
            "approx_logits": "x_hat @ Etilde.T",
            "squared_logit_error": "(x - x_hat) @ G @ (x - x_hat).T",
            "model": "x_hat_f = R @ W_f; R = row_features(histories), W_f = column_weights(f)",
            "oracle": "x_hat_f = U_r @ U_r.T @ S_f; U_r are leading eigenvectors of sum_f S_f @ G @ S_f.T",
            "row_span_refit": "x_hat_f = Q @ Q.T @ S_f; Q is an orthonormal basis of R",
            "column_refit": "x_hat_f = U_star @ W_f; U_star minimizes summed G-norm error",
            "calibrated_scale": "x_hat_f = R @ diag(c) @ W_f; c minimizes calibration G-norm error",
            "calibrated_scale_temperature": "softmax(tau * x_hat @ Etilde.T), with tau fitted on calibration KL(p || q)",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_outputs(output_dir, summary, _plot(summary, experiments), metadata, plot_name="layers.png")
    return _table(summary, experiments)
