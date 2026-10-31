from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from experiments.exp1_L_M_table.table import select_header_tokens
from experiments.exp6_stream_gpt2_mlp.features import build_features
from utils.common import Sample, load_samples, save_json, save_numpy, write_atomic
from utils.inference import CausalLMInference
from utils.lingen import compare_distributions
from utils.stream_neural_svd import centered_unembedding, load_trained_features


VARIANTS = tuple(f"{method}_{option}" for method in ("finite", "nsvd", "single") for option in ("opt1", "opt2")) + ("short_context",)
METRICS = ("kl_pq", "kl_qp", "topk_overlap", "logit_sq_error", "logit_sq_norm")


def build_sets(histories: list[Sample], futures: list[Sample], *, basis_histories: int, targets: int, future_count: int, nonsense_seed: int) -> dict[str, np.ndarray]:
    if basis_histories + targets > len(histories):
        raise ValueError(f"need {basis_histories + targets} histories but the dataset has {len(histories)}")
    if future_count > len(futures):
        raise ValueError(f"need {future_count} futures but the dataset has {len(futures)}")
    basis_samples = histories[:basis_histories]
    target_samples = histories[basis_histories:basis_histories + targets]
    future_samples = futures[:future_count]
    basis = np.asarray([s.token_ids for s in basis_samples], dtype=np.int64)
    target_ids = np.asarray([s.token_ids for s in target_samples], dtype=np.int64)
    future_ids = np.asarray([s.token_ids for s in future_samples], dtype=np.int64)
    gram_futures = np.asarray([s.token_ids for s in futures], dtype=np.int64)
    rng = np.random.default_rng(nonsense_seed)

    def permute(ids: np.ndarray) -> np.ndarray:
        return ids.ravel()[rng.permutation(ids.size)].reshape(ids.shape)

    return {
        "basis": basis, "targets": target_ids, "futures": future_ids,
        "gram_futures": gram_futures,
        "basis_nonsense": permute(basis), "futures_nonsense": permute(future_ids),
        "gram_futures_nonsense": permute(gram_futures),
        "basis_sample_ids": np.asarray([s.sample_id for s in basis_samples], dtype=np.int64),
        "target_sample_ids": np.asarray([s.sample_id for s in target_samples], dtype=np.int64),
        "future_sample_ids": np.asarray([s.sample_id for s in future_samples], dtype=np.int64),
    }


@torch.no_grad()
def selected_centered_logits(inference, token_matrix, history_ids, future_ids, header_ids, *, pair_batch: int) -> torch.Tensor:
    device = token_matrix.device
    history_ids = torch.as_tensor(history_ids, dtype=torch.long, device=device)
    future_ids = torch.as_tensor(future_ids, dtype=torch.long, device=device)
    header_ids = torch.as_tensor(header_ids, dtype=torch.long, device=device)
    n_h, n_f = len(history_ids), len(future_ids)
    k = header_ids.shape[1]
    result = torch.empty((n_h, n_f * k), dtype=torch.float32, device="cpu")
    bf = min(n_f, pair_batch)
    bh = max(1, pair_batch // bf)
    for f in range(0, n_f, bf):
        future = future_ids[f:f + bf]
        selected = token_matrix[header_ids[f:f + bf]]
        for h in range(0, n_h, bh):
            history = history_ids[h:h + bh]
            pairs = torch.cat((history[:, None, :].expand(-1, len(future), -1), future[None, :, :].expand(len(history), -1, -1)), dim=-1).reshape(len(history) * len(future), -1)
            hidden = inference.final_hidden_states(pairs)[:, -1].reshape(len(history), len(future), -1)
            logits = torch.einsum("hfd,fkd->hfk", hidden, selected)
            result[h:h + len(history), f * k:(f + len(future)) * k] = logits.reshape(len(history), -1).cpu()
    return result


def _relative_residual(basis: np.ndarray, targets: np.ndarray, v: np.ndarray) -> np.ndarray:
    result = np.empty(len(targets), dtype=np.float64)
    for start in range(0, len(targets), 8):
        rows = targets[start:start + 8].astype(np.float64, copy=False)
        diff = rows - v[start:start + 8] @ basis
        result[start:start + len(rows)] = np.einsum("ij,ij->i", diff, diff) / np.einsum("ij,ij->i", rows, rows)
    return result


def regress(basis_rows: np.ndarray, target_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = np.asarray(basis_rows).astype(np.float64, copy=False)
    targets = np.asarray(target_rows).astype(np.float64, copy=False)
    solution, _, _, singular_values = np.linalg.lstsq(basis.T, targets.T, rcond=None)
    v = solution.T
    return v, _relative_residual(basis, targets, v), singular_values


@torch.no_grad()
def neural_svd_coefficients(features, token_matrix, basis_ids, target_ids, gram_future_ids, *, batch: int) -> tuple[np.ndarray, np.ndarray]:
    device = token_matrix.device
    basis_ids = torch.as_tensor(basis_ids, dtype=torch.long, device=device)
    target_ids = torch.as_tensor(target_ids, dtype=torch.long, device=device)
    gram_future_ids = torch.as_tensor(gram_future_ids, dtype=torch.long, device=device)

    def rows(ids):
        return torch.cat([features.row_features(ids[i:i + batch]).double() for i in range(0, len(ids), batch)])

    u_h, u_t = rows(basis_ids), rows(target_ids)
    embedding = token_matrix.double()
    token_gram = embedding.T @ embedding
    gram = torch.zeros((u_h.shape[1], u_h.shape[1]), dtype=torch.float64, device=device)
    for i in range(0, len(gram_future_ids), batch):
        weights = features.column_weights(gram_future_ids[i:i + batch]).double()
        gram += torch.einsum("brd,de,bse->rs", weights, token_gram, weights)
    gram /= len(gram_future_ids)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    root = (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T
    a = (root @ u_h.T).cpu().numpy()
    b = (root @ u_t.T).cpu().numpy()
    v = np.linalg.lstsq(a, b, rcond=None)[0].T
    residual = u_t - torch.as_tensor(v, device=device) @ u_h
    numerator = torch.einsum("tr,rs,ts->t", residual, gram, residual)
    denominator = torch.einsum("tr,rs,ts->t", u_t, gram, u_t)
    return v, (numerator / denominator).cpu().numpy()


@torch.no_grad()
def weighted_basis_hidden(inference, basis_ids: torch.Tensor, continuation: torch.Tensor | None, weights: torch.Tensor, *, chunk: int) -> torch.Tensor:
    device = inference.device
    basis_ids = torch.as_tensor(basis_ids, dtype=torch.long, device=device)
    weights = torch.as_tensor(weights, dtype=torch.float64, device=device)
    if continuation is not None:
        continuation = torch.as_tensor(continuation, dtype=torch.long, device=device)
    generations = weights.shape[0]
    result = None
    for start in range(0, len(basis_ids), chunk):
        ids = basis_ids[start:start + chunk]
        out = inference.model.base_model(input_ids=ids, use_cache=True)
        if continuation is None:
            hidden = out.last_hidden_state[:, -1, :][:, None, :].expand(-1, generations, -1)
        else:
            cache = out.past_key_values
            cache.batch_repeat_interleave(generations)
            hidden = inference.model.base_model(input_ids=continuation.repeat(len(ids), 1), past_key_values=cache, use_cache=True).last_hidden_state[:, -1, :].reshape(len(ids), generations, -1)
        contribution = torch.einsum("cgd,gc->gd", hidden.double(), weights[:, start:start + len(ids)])
        result = contribution if result is None else result + contribution
    return result


def _generation_arrays(generations: int, steps: int) -> dict[str, np.ndarray]:
    arrays = {key: np.empty((generations, steps, 4) if key == "topk_overlap" else (generations, steps), dtype=np.float64) for key in METRICS}
    arrays["tokens"] = np.empty((generations, steps), dtype=np.int64)
    return arrays


def _record_and_sample(arrays, step, p_logits, q_logits, generator):
    metrics = compare_distributions(p_logits, q_logits)
    for key in METRICS:
        arrays[key][:, step] = metrics[key].cpu().numpy()
    token = torch.multinomial(metrics["log_q"].exp(), 1, generator=generator)
    arrays["tokens"][:, step] = token[:, 0].cpu().numpy()
    return token


@torch.no_grad()
def run_lingen_variant(inference, token_matrix, *, basis_ids, target_ids, coefficients, samples_per_target, steps, chunk, generator) -> dict[str, np.ndarray]:
    device = token_matrix.device
    basis = torch.as_tensor(basis_ids, dtype=torch.long, device=device)
    targets = torch.as_tensor(target_ids, dtype=torch.long, device=device).repeat_interleave(samples_per_target, dim=0)
    weights = torch.as_tensor(coefficients, dtype=torch.float64, device=device).repeat_interleave(samples_per_target, dim=0)
    embedding_t = token_matrix.double().T
    z = torch.empty((len(targets), 0), dtype=torch.long, device=device)
    arrays = _generation_arrays(len(targets), steps)
    for step in range(steps):
        hidden = weighted_basis_hidden(inference, basis, z if step else None, weights, chunk=chunk)
        q_logits = hidden @ embedding_t
        p_logits = inference.final_hidden_states(torch.cat((targets, z), dim=1))[:, -1].double() @ embedding_t
        z = torch.cat((z, _record_and_sample(arrays, step, p_logits, q_logits, generator)), dim=1)
    return arrays


@torch.no_grad()
def run_short_context(inference, token_matrix, *, target_ids, samples_per_target, steps, context_tokens, generator) -> dict[str, np.ndarray]:
    device = token_matrix.device
    targets = torch.as_tensor(target_ids, dtype=torch.long, device=device).repeat_interleave(samples_per_target, dim=0)
    embedding_t = token_matrix.double().T
    z = torch.empty((len(targets), 0), dtype=torch.long, device=device)
    arrays = _generation_arrays(len(targets), steps)
    for step in range(steps):
        context = torch.cat((targets, z), dim=1)
        p_logits = inference.final_hidden_states(context)[:, -1].double() @ embedding_t
        q_logits = inference.final_hidden_states(context[:, -context_tokens:])[:, -1].double() @ embedding_t
        z = torch.cat((z, _record_and_sample(arrays, step, p_logits, q_logits, generator)), dim=1)
    return arrays


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as stream:
        np.savez(stream, **arrays)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    write_atomic(path, lambda tmp: _save_npz(tmp, arrays))


def _check_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}; pass --overwrite to replace it")


def _single_rows(inference, token_matrix, ids, batch):
    embedding_t = token_matrix.double().T
    result = np.empty((len(ids), token_matrix.shape[0]), dtype=np.float64)
    for i in range(0, len(ids), batch):
        hidden = inference.final_hidden_states(torch.as_tensor(ids[i:i + batch], dtype=torch.long, device=token_matrix.device))[:, -1].double()
        result[i:i + len(hidden)] = (hidden @ embedding_t).cpu().numpy()
    return result


def fit_coefficients(*, samples_path, model_path, device, basis_histories, targets, future_count, top_k, nonsense_seed, pair_batch, nsvd_train_dir, gram_batch, output_dir, overwrite) -> dict:
    samples_path, model_path, nsvd_train_dir, output_dir = map(Path, (samples_path, model_path, nsvd_train_dir, output_dir))
    metadata_path = output_dir / "metadata.json"
    if not overwrite:
        _check_output(metadata_path)
    else:
        for name in ("metadata.json", "sets.npz", "coefficients.npz", *(f"{prefix}_{option}.npy" for option in ("opt1", "opt2") for prefix in ("matrix", "target_rows", "header"))):
            (output_dir / name).unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    inference = CausalLMInference(model_path, device)
    token_matrix = centered_unembedding(inference)
    histories, futures = load_samples(samples_path, None, None)
    sets = build_sets(histories, futures, basis_histories=basis_histories, targets=targets, future_count=future_count, nonsense_seed=nonsense_seed)
    _atomic_npz(output_dir / "sets.npz", sets)
    coefficients = {}
    summaries = {}
    target_single = _single_rows(inference, token_matrix, sets["targets"], pair_batch)
    for option in ("opt1", "opt2"):
        print(f"fitting {option}", flush=True)
        suffix = "" if option == "opt1" else "_nonsense"
        basis = sets["basis" + suffix]
        future = sets["futures" + suffix]
        header, _ = select_header_tokens(inference, [Sample(i, ids.tolist()) for i, ids in enumerate(future)], top_k)
        matrix = selected_centered_logits(inference, token_matrix, basis, future, header, pair_batch=pair_batch).numpy()
        target_rows = selected_centered_logits(inference, token_matrix, sets["targets"], future, header, pair_batch=pair_batch).numpy()
        finite, _, singular = regress(matrix, target_rows)
        single_rows = _single_rows(inference, token_matrix, basis, pair_batch)
        single, single_residual, single_singular = regress(single_rows, target_single)
        coefficients[f"finite_{option}"] = finite
        coefficients[f"finite_{option}_singular_values"] = singular
        coefficients[f"single_{option}"] = single
        coefficients[f"single_{option}_residual"] = single_residual
        coefficients[f"single_{option}_singular_values"] = single_singular
        for name, value in (("matrix", matrix), ("target_rows", target_rows), ("header", header)):
            write_atomic(output_dir / f"{name}_{option}.npy", lambda tmp, value=value: save_numpy(tmp, value))
        del matrix, target_rows, single_rows
    train_metadata, teacher, features = load_trained_features(nsvd_train_dir, build_features, device)
    if Path(train_metadata["model_path"]).name != model_path.name:
        raise ValueError(f"NeuralSVD teacher {train_metadata['model_path']} differs from regression model {model_path}")
    for option in ("opt1", "opt2"):
        print(f"fitting NeuralSVD {option}", flush=True)
        suffix = "" if option == "opt1" else "_nonsense"
        v, residual = neural_svd_coefficients(features, token_matrix, sets["basis" + suffix], sets["targets"], sets["gram_futures" + suffix], batch=gram_batch)
        coefficients[f"nsvd_{option}"] = v
        coefficients[f"nsvd_{option}_population_residual"] = residual
        matrix = np.load(output_dir / f"matrix_{option}.npy", mmap_mode="r")
        target_rows = np.load(output_dir / f"target_rows_{option}.npy", mmap_mode="r")
        matrix64 = matrix.astype(np.float64)
        for method in ("finite", "nsvd", "single"):
            name = f"{method}_{option}"
            v = coefficients[name]
            residual = _relative_residual(matrix64, target_rows, v)
            coefficients[f"{name}_finite_residual"] = residual
            summaries[name] = {"finite_residual_mean": float(residual.mean()), "finite_residual_median": float(np.median(residual)), "v_norm_mean": float(np.linalg.norm(v, axis=1).mean())}
            if method == "nsvd":
                summaries[name]["population_residual_mean"] = float(coefficients[f"{name}_population_residual"].mean())
            if method == "single":
                summaries[name]["residual_mean"] = float(coefficients[f"{name}_residual"].mean())
        del matrix64, matrix, target_rows
    _atomic_npz(output_dir / "coefficients.npz", coefficients)
    config = {"samples_path": str(samples_path), "model_path": str(model_path), "device": str(device), "basis_histories": basis_histories, "targets": targets, "future_count": future_count, "top_k": top_k, "nonsense_seed": nonsense_seed, "pair_batch": pair_batch, "nsvd_train_dir": str(nsvd_train_dir), "gram_batch": gram_batch, "output_dir": str(output_dir), "overwrite": overwrite}
    dataset_metadata = json.loads(pq.read_schema(samples_path).metadata[b"lm_svd"].decode("utf-8"))
    metadata = {"config": config, "dataset_metadata": dataset_metadata, "vocab_size": inference.vocab_size, "history_tokens": int(sets["targets"].shape[1]), "nsvd": {"train_dir": str(nsvd_train_dir), "train_steps": train_metadata["steps"], "features_config": train_metadata["features"]}, "coefficients": summaries}
    write_atomic(metadata_path, lambda tmp: save_json(tmp, metadata))
    return metadata


def _variant_summary(arrays):
    pq_by_position = arrays["kl_pq"].mean(axis=0)
    qp_by_position = arrays["kl_qp"].mean(axis=0)
    return {"kl_pq_by_position": pq_by_position.tolist(), "kl_pq_total": float(pq_by_position.sum()), "kl_pq_median": float(np.median(arrays["kl_pq"])), "kl_qp_by_position": qp_by_position.tolist(), "kl_qp_total": float(qp_by_position.sum()), "top1": float(arrays["topk_overlap"][..., 0].mean()), "top10": float(arrays["topk_overlap"][..., 2].mean()), "logit_relative_error_aggregate": float(arrays["logit_sq_error"].sum() / arrays["logit_sq_norm"].sum())}


def generate(*, fit_dir, eval_model_path, device, samples_per_target, steps, short_context, basis_chunk, seed, output_dir, overwrite) -> str:
    from transformers import AutoTokenizer

    fit_dir, eval_model_path, output_dir = map(Path, (fit_dir, eval_model_path, output_dir))
    fit_metadata_path = fit_dir / "metadata.json"
    if not fit_metadata_path.is_file():
        raise ValueError(f"fit output does not exist: {fit_metadata_path}")
    fit_bytes = fit_metadata_path.read_bytes()
    fit_metadata = json.loads(fit_bytes)
    inference = CausalLMInference(eval_model_path, device)
    if inference.vocab_size != fit_metadata["vocab_size"]:
        raise ValueError(f"evaluation vocabulary size {inference.vocab_size} differs from fit vocabulary size {fit_metadata['vocab_size']}")
    if fit_metadata["history_tokens"] + steps - 1 > inference.max_length:
        raise ValueError("history and generation exceed evaluation model maximum length")
    config = {"fit_metadata_sha256": hashlib.sha256(fit_bytes).hexdigest(), "eval_model_path": str(eval_model_path), "samples_per_target": samples_per_target, "steps": steps, "short_context": short_context, "seed": seed}
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in (*output_dir.glob("*.npz"), output_dir / "samples.jsonl", output_dir / "metadata.json"):
            path.unlink(missing_ok=True)
    with np.load(fit_dir / "sets.npz") as data:
        sets = {key: data[key] for key in data.files}
    with np.load(fit_dir / "coefficients.npz") as data:
        coefficients = {key: data[key] for key in data.files}
    token_matrix = centered_unembedding(inference)
    summaries = {}
    token_outputs = {}
    for i, name in enumerate(VARIANTS):
        print(f"processing variant {name}", flush=True)
        path = output_dir / f"{name}.npz"
        if path.exists() and not overwrite:
            with np.load(path, allow_pickle=False) as saved:
                try:
                    saved_config = json.loads(saved["config"].item())
                except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                    saved_config = None
                if saved_config != config:
                    raise ValueError(f"existing variant output {path} was produced with different settings; pass --overwrite to replace it")
                arrays = {key: saved[key] for key in saved.files if key != "config"}
            print(f"skipping completed variant {name}", flush=True)
        else:
            generator = torch.Generator(device=device).manual_seed(seed + i)
            if name == "short_context":
                arrays = run_short_context(inference, token_matrix, target_ids=sets["targets"], samples_per_target=samples_per_target, steps=steps, context_tokens=short_context, generator=generator)
            else:
                basis = sets["basis_nonsense" if name.endswith("opt2") else "basis"]
                arrays = run_lingen_variant(inference, token_matrix, basis_ids=basis, target_ids=sets["targets"], coefficients=coefficients[name], samples_per_target=samples_per_target, steps=steps, chunk=basis_chunk, generator=generator)
            _atomic_npz(path, {**arrays, "config": np.asarray(json.dumps(config, sort_keys=True))})
        summaries[name] = _variant_summary(arrays)
        token_outputs[name] = arrays["tokens"]
    tokenizer = AutoTokenizer.from_pretrained(eval_model_path, local_files_only=True)

    def write_samples(path):
        with path.open("w", encoding="utf-8") as stream:
            for name in VARIANTS:
                for g, tokens in enumerate(token_outputs[name]):
                    target_index = g // samples_per_target
                    item = {"variant": name, "target_index": target_index, "sample": g % samples_per_target, "context": tokenizer.decode(sets["targets"][target_index].tolist(), clean_up_tokenization_spaces=False), "continuation": tokenizer.decode(tokens.tolist(), clean_up_tokenization_spaces=False)}
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_atomic(output_dir / "samples.jsonl", write_samples)
    write_atomic(output_dir / "metadata.json", lambda tmp: save_json(tmp, {"config": config, "variants": summaries}))
    return str(output_dir)


def report(*, lingen_dir, output_dir, overwrite) -> str:
    import matplotlib.pyplot as plt

    lingen_dir, output_dir = Path(lingen_dir), Path(output_dir)
    metadata_paths = sorted(lingen_dir.glob("*/metadata.json"))
    if not metadata_paths:
        raise ValueError(f"no generate outputs under {lingen_dir}")
    outputs = (output_dir / "summary.json", output_dir / "kl_by_position.png")
    if not overwrite:
        for path in outputs:
            _check_output(path)
    summaries = {path.parent.name: json.loads(path.read_text(encoding="utf-8"))["variants"] for path in metadata_paths}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(outputs[0], lambda tmp: save_json(tmp, summaries))
    fig, axes = plt.subplots(1, len(summaries), figsize=(7 * len(summaries), 5), squeeze=False)
    for ax, (model, variants) in zip(axes[0], summaries.items()):
        for name, metrics in variants.items():
            values = metrics["kl_pq_by_position"]
            ax.plot(range(1, len(values) + 1), values, label=f"{name}; total={metrics['kl_pq_total']:.2f}")
        ax.set_title(model)
        ax.set_xlabel("Position")
        ax.set_ylabel("KL(p‖q)")
        ax.legend()
    fig.tight_layout()
    try:
        write_atomic(outputs[1], lambda tmp: fig.savefig(tmp, format="png"))
    finally:
        plt.close(fig)
    lines = ["| model | variant | kl_pq_total | mean kl_pq | kl_qp_total | top1 | top10 | logit_relative_error_aggregate |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, variants in summaries.items():
        for name, metrics in variants.items():
            values = (metrics["kl_pq_total"], np.mean(metrics["kl_pq_by_position"]), metrics["kl_qp_total"], metrics["top1"], metrics["top10"], metrics["logit_relative_error_aggregate"])
            lines.append(f"| {model} | {name} | " + " | ".join(f"{value:.6g}" for value in values) + " |")
    return "\n".join(lines)
