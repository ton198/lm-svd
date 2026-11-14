
from pathlib import Path

import torch

from data_pipline.sample_store.store import SampleSampler
from utils.inference import CausalLMInference

from .main import load_model_from_checkpoint


def get_one_sample():
    """
    Returns a single sample from the dataset.
    """
    store_dir = Path(__file__).parent / "data" / "sample_store"
    sampler = SampleSampler(store_dir, seed=0)
    h, f, logits = sampler.sample()
    logits = logits - logits.mean()
    token_ids = torch.randint(logits.shape[0], (32,), generator=torch.Generator().manual_seed(0))
    f_concat_z = torch.cat(
        (f.unsqueeze(0).expand(32, -1), token_ids.unsqueeze(-1)),
        dim=-1,
    )
    return h, f_concat_z, logits[token_ids]

def get_ramdom_h_samples(num_samples: int):
    """
    Returns a batch of random h's.
    """
    store_dir = Path(__file__).parent / "data" / "sample_store"
    sampler = SampleSampler(store_dir, seed=0)
    hs = torch.empty((num_samples, sampler.config["history_tokens"]), dtype=torch.long)
    for index in range(num_samples):
        hs[index] = sampler.sample()[0]
    return hs

def load_model(checkpoint_path: str | Path = "trained_model.pth"):
    model = load_model_from_checkpoint(checkpoint_path, neigs=128)
    model.eval()
    return model

@torch.no_grad()
def calculate_phi_h(model, h):
    f = h.new_zeros((*h.shape[:-1], 1))
    phi_h, _ = model(h, f)
    return phi_h

def calculate_h_coefficients(model, hn, ht):
    """
    Return coefficients that best approximate phi(ht) using phi(hn[i]).

    With A = [phi(hn[0]), ..., phi(hn[N-1])] of shape (K, N) and
    b = phi(ht) of shape (K,), solve:

        c* = argmin_c ||A c - b||_2^2
           = argmin_c ||sum_i c_i phi(hn[i]) - phi(ht)||_2^2

    The pseudoinverse gives c* = A^+ b, the minimum-norm least-squares
    solution, with shape (N,). Coefficients are unconstrained: they may
    be negative and need not sum to one.

    For a target batch ht of shape (B, H), return coefficients of shape
    (B, N), sharing the reference pseudoinverse across all targets.
    """
    phi_hn = calculate_phi_h(model, hn)
    phi_ht = calculate_phi_h(model, ht if ht.ndim == 2 else ht.unsqueeze(0))
    coefficients = phi_ht @ torch.linalg.pinv(phi_hn)
    return coefficients if ht.ndim == 2 else coefficients.squeeze(0)

def approximate_next_token_logits(gpt_model: CausalLMInference, hn, coefficients):
    logits = gpt_model.next_token_logits_batch(hn.tolist())
    return coefficients.to(device=logits.device, dtype=logits.dtype) @ logits

def evaluate_next_token_approximation(gpt_model: CausalLMInference, model, hn, ht):
    coefficients = calculate_h_coefficients(model, hn, ht)
    approx_logits = approximate_next_token_logits(gpt_model, hn, coefficients)
    true_logits = (
        gpt_model.next_token_logits_batch(ht.tolist())
        if ht.ndim == 2 else gpt_model.next_token_logits(ht.tolist())
    )
    return approx_logits, true_logits

@torch.no_grad()
def report_next_token_approximation(num_target_histories: int = 32) -> dict[str, float]:
    """Print batch diagnostics, save a comparison figure, and return mean/std metrics.

    Logit errors use vocabulary-centered logits, matching the training target
    and ignoring constant offsets that do not affect next-token probabilities.
    """
    from matplotlib.figure import Figure

    if num_target_histories < 1:
        raise ValueError("num_target_histories must be positive")

    model_path = Path("models/gpt2-small")
    checkpoint_path = Path("trained_model.pth")
    store_dir = Path("data/sample_stores/wikitext-103-train__gpt2-small__h64-f32")
    output_path = Path("next_token_approximation.png")
    top_k = 10
    num_reference_histories = 4096
    seed = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gpt_model = CausalLMInference(model_path, device=device)
    model = load_model(checkpoint_path)
    sampler = SampleSampler(store_dir, seed=seed)
    ht = torch.stack([sampler.sample()[0] for _ in range(num_target_histories)])
    hn = torch.stack([sampler.sample()[0] for _ in range(num_reference_histories)])

    if top_k < 1:
        raise ValueError("top_k must be positive")

    approx_logits, true_logits = evaluate_next_token_approximation(
        gpt_model, model, hn, ht
    )
    approx_logits = approx_logits.detach().to(device="cpu", dtype=torch.float64)
    true_logits = true_logits.detach().to(device="cpu", dtype=torch.float64)
    if true_logits.ndim != 2 or approx_logits.shape != true_logits.shape or not true_logits.numel():
        raise ValueError("expected matching, nonempty (target, vocabulary) logit matrices")
    if not torch.isfinite(approx_logits).all() or not torch.isfinite(true_logits).all():
        raise ValueError("evaluation logits must be finite")

    approx_centered = approx_logits - approx_logits.mean(dim=-1, keepdim=True)
    true_centered = true_logits - true_logits.mean(dim=-1, keepdim=True)
    error = approx_centered - true_centered
    approx_log_probs = approx_logits.log_softmax(dim=-1)
    true_log_probs = true_logits.log_softmax(dim=-1)
    approx_probs, true_probs = approx_log_probs.exp(), true_log_probs.exp()
    k = min(top_k, true_logits.shape[-1])
    true_top = true_logits.topk(k).indices
    approx_top = approx_logits.topk(k).indices
    per_history_metrics = {
        "centered_logit_rmse": error.square().mean(dim=-1).sqrt(),
        "centered_logit_mae": error.abs().mean(dim=-1),
        "mean_logit_bias": approx_logits.mean(dim=-1) - true_logits.mean(dim=-1),
        "kl_true_to_approx": (true_probs * (true_log_probs - approx_log_probs)).sum(dim=-1),
        "total_variation": 0.5 * (true_probs - approx_probs).abs().sum(dim=-1),
        "top1_agreement": (true_top[:, 0] == approx_top[:, 0]).double(),
        f"top{k}_overlap": (
            true_top.unsqueeze(-1) == approx_top.unsqueeze(-2)
        ).any(dim=-1).double().mean(dim=-1),
    }
    metrics = {}
    print(f"Next-token approximation: {ht.shape[0]} target histories, {hn.shape[0]} reference histories")
    for name, values in per_history_metrics.items():
        metrics[name] = values.mean().item()
        metrics[f"{name}_std"] = values.std(unbiased=False).item()
        print(f"  {name}: mean={metrics[name]:.6g}, std={metrics[f'{name}_std']:.6g}")

    figure = Figure(figsize=(13, 5), constrained_layout=True)
    logit_axis, error_axis = figure.subplots(1, 2)
    logit_axis.hexbin(
        true_centered.flatten().numpy(), approx_centered.flatten().numpy(), gridsize=60, mincnt=1,
        bins="log", cmap="viridis",
    )
    lower = min(true_centered.min().item(), approx_centered.min().item())
    upper = max(true_centered.max().item(), approx_centered.max().item())
    logit_axis.plot([lower, upper], [lower, upper], "r--", label="Perfect approximation")
    logit_axis.set(
        xlabel="True centered logit", ylabel="Approximated centered logit",
        title=f"All target logits (mean RMSE = {metrics['centered_logit_rmse']:.3g})",
    )
    logit_axis.legend()

    error_axis.hist(per_history_metrics["centered_logit_rmse"].numpy(), bins="auto")
    error_axis.set(
        xlabel="Centered logit RMSE", ylabel="Target histories",
        title="Error distribution across target histories",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    print(f"Saved evaluation plots to {output_path}")
    return metrics

if __name__ == "__main__":
    report_next_token_approximation()
