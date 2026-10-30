from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer

from utils.common import save_json, write_atomic
from utils.inference import CausalLMInference
from utils.stream_neural_svd import TokenStream, centered_unembedding, load_trained_features


TOPK = (1, 5, 10, 50)


@torch.no_grad()
def teacher_logits(
    inference: CausalLMInference, token_matrix: torch.Tensor, context_ids: torch.Tensor
) -> torch.Tensor:
    return inference.final_hidden_states(context_ids)[:, -1] @ token_matrix.T


@torch.no_grad()
def reconstructed_logits(
    features: nn.Module,
    token_matrix: torch.Tensor,
    history_ids: torch.Tensor,
    future_ids: torch.Tensor,
) -> torch.Tensor:
    left = features.row_features(history_ids.to(token_matrix.device))
    weights = features.column_weights(future_ids.to(token_matrix.device))
    hidden = torch.einsum("br,brd->bd", left, weights)
    return hidden @ token_matrix.T


def compare_distributions(teacher: torch.Tensor, learned: torch.Tensor) -> dict[str, torch.Tensor]:
    teacher = teacher.double()
    learned = learned.double()
    log_p = torch.log_softmax(teacher, dim=-1)
    log_q = torch.log_softmax(learned, dim=-1)
    kmax = min(max(TOPK), teacher.shape[-1])
    top_p = teacher.topk(kmax, dim=-1).indices
    top_q = learned.topk(kmax, dim=-1).indices
    overlaps = []
    for k in TOPK:
        count = min(k, teacher.shape[-1])
        overlaps.append(
            (top_p[:, :count, None] == top_q[:, None, :count])
            .any(dim=-1).sum(dim=-1).double() / count
        )
    centered_teacher = teacher - teacher.mean(dim=-1, keepdim=True)
    centered_learned = learned - learned.mean(dim=-1, keepdim=True)
    return {
        "kl_pq": (log_p.exp() * (log_p - log_q)).sum(dim=-1),
        "kl_qp": (log_q.exp() * (log_q - log_p)).sum(dim=-1),
        "topk_overlap": torch.stack(overlaps, dim=-1),
        "logit_sq_error": (centered_teacher - centered_learned).square().sum(dim=-1),
        "logit_sq_norm": centered_teacher.square().sum(dim=-1),
        "log_p": log_p,
        "log_q": log_q,
    }


@torch.no_grad()
def _check_outputs(paths: tuple[Path, ...], overwrite: bool) -> None:
    if not overwrite:
        for path in paths:
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )


@torch.no_grad()
def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


@torch.no_grad()
def _load(
    train_dir: Path,
    build_features: Callable[[dict, CausalLMInference], nn.Module],
    device: str,
    eval_stream_path: Path,
) -> tuple[dict, CausalLMInference, nn.Module, TokenStream, torch.Tensor]:
    metadata, inference, features = load_trained_features(train_dir, build_features, device)
    stream = TokenStream(eval_stream_path)
    vocab_size = stream.metadata["vocab_size"]
    if vocab_size != inference.vocab_size:
        raise ValueError(
            f"eval stream vocab_size {vocab_size} does not match teacher vocab_size {inference.vocab_size}"
        )
    return metadata, inference, features, stream, centered_unembedding(inference)


@torch.no_grad()
def _input_metadata(train_dir: Path, metadata: dict, stream_path: Path, stream: TokenStream) -> dict:
    return {
        "train_dir": str(train_dir),
        "train_steps": metadata["steps"],
        "train_converged": metadata["converged"],
        "train_metadata": metadata,
        "eval_stream_path": str(stream_path),
        "eval_stream_metadata": stream.metadata,
    }


DEFINITIONS = {
    "p": "softmax(GPT-2 logits at last position of history+future)",
    "q": "softmax(u(h)^T W(f) Etilde^T)",
    "kl_pq": "sum_z p log(p/q)",
    "kl_qp": "sum_z q log(q/p)",
    "topk_overlap": "|top_k(p) ∩ top_k(q)| / k",
    "logit_relative_error": "||A - Ahat||^2 / ||A||^2 per pair, centered logits",
    "logit_relative_error_aggregate": "sum ||A - Ahat||^2 / sum ||A||^2",
}


@torch.no_grad()
def run_lingen_onestep(
    *, train_dir: Path, build_features: Callable[[dict, CausalLMInference], nn.Module],
    device: str, eval_stream_path: Path, output_dir: Path, pairs: int,
    batch_size: int, seed: int, overwrite: bool,
) -> str:
    metrics_path = output_dir / "metrics.npz"
    metadata_path = output_dir / "metadata.json"
    _check_outputs((metrics_path, metadata_path), overwrite)
    train_metadata, inference, features, stream, token_matrix = _load(
        train_dir, build_features, device, eval_stream_path
    )
    h_len, f_len = train_metadata["history_tokens"], train_metadata["future_tokens"]
    gen = torch.Generator().manual_seed(seed)
    history_starts = stream.sample_starts(pairs, h_len, gen)
    future_starts = stream.sample_starts(pairs, f_len, gen)
    names = ("kl_pq", "kl_qp", "topk_overlap", "logit_sq_error", "logit_sq_norm")
    chunks = {name: [] for name in names}
    for start in range(0, pairs, batch_size):
        h = stream.windows(history_starts[start:start + batch_size], h_len).to(device)
        f = stream.windows(future_starts[start:start + batch_size], f_len).to(device)
        result = compare_distributions(
            teacher_logits(inference, token_matrix, torch.cat((h, f), dim=1)),
            reconstructed_logits(features, token_matrix, h, f),
        )
        for name in names:
            chunks[name].append(result[name].cpu().numpy())
    arrays = {name: np.concatenate(chunks[name], axis=0) for name in names}
    arrays.update({
        "history_starts": history_starts.numpy(),
        "future_starts": future_starts.numpy(),
        "topk": np.array(TOPK),
        "logit_relative_error": arrays["logit_sq_error"] / arrays["logit_sq_norm"],
    })
    aggregate = float(arrays["logit_sq_error"].sum() / arrays["logit_sq_norm"].sum())
    summary = {
        name: {"mean": float(arrays[name].mean()), "median": float(np.median(arrays[name]))}
        for name in ("kl_pq", "kl_qp", "logit_relative_error")
    }
    summary["topk_overlap"] = {
        str(k): float(arrays["topk_overlap"][:, i].mean()) for i, k in enumerate(TOPK)
    }
    summary["logit_relative_error_aggregate"] = aggregate
    metadata = {
        "mode": "onestep",
        "input": _input_metadata(train_dir, train_metadata, eval_stream_path, stream),
        "config": {"pairs": pairs, "batch_size": batch_size, "seed": seed,
                   "history_tokens": h_len, "future_tokens": f_len, "topk": list(TOPK)},
        "definitions": DEFINITIONS,
        "summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(metrics_path, lambda tmp: _save_npz(tmp, arrays))
    write_atomic(metadata_path, lambda tmp: save_json(tmp, metadata))
    return "\n".join((
        f"output_dir: {output_dir}", f"training step: {train_metadata['steps']}",
        f"KL(p||q) mean: {summary['kl_pq']['mean']:.6g}",
        f"KL(q||p) mean: {summary['kl_qp']['mean']:.6g}",
        f"top-1 overlap: {summary['topk_overlap']['1']:.6g}",
        f"top-10 overlap: {summary['topk_overlap']['10']:.6g}",
        f"aggregate relative error: {aggregate:.6g}", f"metadata: {metadata_path}",
    ))


@torch.no_grad()
def run_lingen_autoregressive(
    *, train_dir: Path, build_features: Callable[[dict, CausalLMInference], nn.Module],
    device: str, eval_stream_path: Path, output_dir: Path, sequences: int,
    steps: int, seed: int, overwrite: bool,
) -> str:
    metrics_path = output_dir / "metrics.npz"
    samples_path = output_dir / "samples.jsonl"
    plot_path = output_dir / "autoregressive.png"
    metadata_path = output_dir / "metadata.json"
    _check_outputs((metrics_path, samples_path, plot_path, metadata_path), overwrite)
    train_metadata, inference, features, stream, token_matrix = _load(
        train_dir, build_features, device, eval_stream_path
    )
    h_len, f_len = train_metadata["history_tokens"], train_metadata["future_tokens"]
    gen = torch.Generator().manual_seed(seed)
    history_starts = stream.sample_starts(sequences, h_len, gen)
    future_starts = stream.sample_starts(sequences, f_len, gen)
    context0 = torch.cat((stream.windows(history_starts, h_len),
                          stream.windows(future_starts, f_len)), dim=1).to(device)
    lingen_gen = torch.Generator(device=token_matrix.device).manual_seed(seed + 1)
    gpt2_gen = torch.Generator(device=token_matrix.device).manual_seed(seed + 2)
    metric_names = ("kl_pq", "kl_qp", "logit_relative_error", "gpt2_nll_lingen_tokens",
                    "lingen_nll_lingen_tokens", "gpt2_nll_gpt2_tokens")
    arrays = {name: np.empty((sequences, steps), dtype=np.float64) for name in metric_names}
    arrays.update({
        "history_starts": history_starts.numpy(), "future_starts": future_starts.numpy(),
        "initial_context": context0.cpu().numpy(), "topk": np.array(TOPK),
        "topk_overlap": np.empty((sequences, steps, len(TOPK)), dtype=np.float64),
        "lingen_tokens": np.empty((sequences, steps), dtype=np.int64),
        "gpt2_tokens": np.empty((sequences, steps), dtype=np.int64),
    })
    context = context0.clone()
    for t in range(steps):
        h, f = context[:, :h_len], context[:, h_len:]
        result = compare_distributions(
            teacher_logits(inference, token_matrix, context),
            reconstructed_logits(features, token_matrix, h, f),
        )
        z = torch.multinomial(result["log_q"].exp(), 1, generator=lingen_gen)
        for name in ("kl_pq", "kl_qp", "topk_overlap"):
            arrays[name][:, t] = result[name].cpu().numpy()
        arrays["logit_relative_error"][:, t] = (
            result["logit_sq_error"] / result["logit_sq_norm"]
        ).cpu().numpy()
        arrays["gpt2_nll_lingen_tokens"][:, t] = -result["log_p"].gather(1, z).squeeze(1).cpu().numpy()
        arrays["lingen_nll_lingen_tokens"][:, t] = -result["log_q"].gather(1, z).squeeze(1).cpu().numpy()
        arrays["lingen_tokens"][:, t] = z.squeeze(1).cpu().numpy()
        context = torch.cat((context[:, 1:], z), dim=1)
    context = context0.clone()
    for t in range(steps):
        log_p = torch.log_softmax(teacher_logits(inference, token_matrix, context).double(), dim=-1)
        z = torch.multinomial(log_p.exp(), 1, generator=gpt2_gen)
        arrays["gpt2_nll_gpt2_tokens"][:, t] = -log_p.gather(1, z).squeeze(1).cpu().numpy()
        arrays["gpt2_tokens"][:, t] = z.squeeze(1).cpu().numpy()
        context = torch.cat((context[:, 1:], z), dim=1)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(inference.model_path), use_fast=True, local_files_only=True
        )
    except OSError as error:
        raise ValueError(f"cannot load local tokenizer from {inference.model_path}: {error}") from error

    per_step = {name: arrays[name].mean(axis=0).tolist() for name in metric_names}
    overall = {name: float(arrays[name].mean()) for name in metric_names}
    per_step["topk_overlap"] = {
        str(k): arrays["topk_overlap"][:, :, i].mean(axis=0).tolist()
        for i, k in enumerate(TOPK)
    }
    overall["topk_overlap"] = {
        str(k): float(arrays["topk_overlap"][:, :, i].mean()) for i, k in enumerate(TOPK)
    }
    overall["gpt2_perplexity_lingen_tokens"] = float(np.exp(overall["gpt2_nll_lingen_tokens"]))
    overall["gpt2_perplexity_gpt2_tokens"] = float(np.exp(overall["gpt2_nll_gpt2_tokens"]))
    metadata = {
        "mode": "autoregressive",
        "input": _input_metadata(train_dir, train_metadata, eval_stream_path, stream),
        "config": {"sequences": sequences, "steps": steps, "seed": seed,
                   "lingen_sampling_seed": seed + 1, "gpt2_sampling_seed": seed + 2,
                   "temperature": 1.0, "history_tokens": h_len, "future_tokens": f_len,
                   "topk": list(TOPK),
                   "window_update": "context = (h+f)[1:] + z; h = context[:H], f = context[H:]"},
        "definitions": DEFINITIONS,
        "summary": {"per_step": per_step, "overall": overall},
    }

    @torch.no_grad()
    def save_samples(path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for i in range(sequences):
                sample = {
                    "index": i, "history_start": int(history_starts[i]),
                    "future_start": int(future_starts[i]),
                    "context": tokenizer.decode(arrays["initial_context"][i].tolist(), clean_up_tokenization_spaces=False),
                    "lingen": tokenizer.decode(arrays["lingen_tokens"][i].tolist(), clean_up_tokenization_spaces=False),
                    "gpt2": tokenizer.decode(arrays["gpt2_tokens"][i].tolist(), clean_up_tokenization_spaces=False),
                }
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    try:
        x = np.arange(1, steps + 1)
        for name, label in (("kl_pq", "KL(p||q)"), ("kl_qp", "KL(q||p)")):
            axes[0, 0].plot(x, per_step[name], label=label)
        for k in TOPK:
            axes[0, 1].plot(x, per_step["topk_overlap"][str(k)], label=f"top-{k}")
        axes[1, 0].plot(x, per_step["logit_relative_error"], label="relative error")
        for name, label in (("gpt2_nll_lingen_tokens", "GPT-2 NLL on lingen"),
                            ("lingen_nll_lingen_tokens", "lingen NLL on lingen"),
                            ("gpt2_nll_gpt2_tokens", "GPT-2 NLL on GPT-2")):
            axes[1, 1].plot(x, per_step[name], label=label)
        for ax, title in zip(axes.flat, ("KL divergence", "Top-k overlap", "Centered logit relative error", "Token NLL")):
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.legend()
        fig.suptitle(f"lingen autoregressive: {sequences} sequences x {steps} steps")
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(metrics_path, lambda tmp: _save_npz(tmp, arrays))
        write_atomic(samples_path, save_samples)
        write_atomic(plot_path, lambda tmp: fig.savefig(tmp, format="png", dpi=150))
    finally:
        plt.close(fig)
    write_atomic(metadata_path, lambda tmp: save_json(tmp, metadata))
    return "\n".join((
        f"output_dir: {output_dir}", f"training step: {train_metadata['steps']}",
        f"KL(p||q) mean: {overall['kl_pq']:.6g}",
        f"KL(q||p) mean: {overall['kl_qp']:.6g}",
        f"top-1 overlap: {overall['topk_overlap']['1']:.6g}",
        f"top-10 overlap: {overall['topk_overlap']['10']:.6g}",
        f"relative error mean: {overall['logit_relative_error']:.6g}",
        f"GPT-2 perplexity on lingen tokens: {overall['gpt2_perplexity_lingen_tokens']:.6g}",
        f"GPT-2 perplexity on GPT-2 tokens: {overall['gpt2_perplexity_gpt2_tokens']:.6g}",
        f"samples: {samples_path}", f"metadata: {metadata_path}",
    ))
