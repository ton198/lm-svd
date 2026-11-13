
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
    """
    phi_hn = calculate_phi_h(model, hn)
    phi_ht = calculate_phi_h(model, ht).reshape(-1)
    coefficients = torch.linalg.pinv(phi_hn.T) @ phi_ht
    return coefficients

def approximate_next_token_logits(gpt_model: CausalLMInference, hn, coefficients):
    logits = gpt_model.next_token_logits_batch(hn.tolist())
    return coefficients.to(device=logits.device, dtype=logits.dtype) @ logits

def evaluate_next_token_approximation(gpt_model: CausalLMInference, model, hn, ht):
    coefficients = calculate_h_coefficients(model, hn, ht)
    approx_logits = approximate_next_token_logits(gpt_model, hn, coefficients)
    true_logits = gpt_model.next_token_logits(ht.tolist())
    return approx_logits, true_logits

@torch.no_grad()
def report_next_token_approximation() -> dict[str, float]:
    """Print single-history diagnostics, save a comparison figure, and return metrics.

    Logit errors use vocabulary-centered logits, matching the training target
    and ignoring constant offsets that do not affect next-token probabilities.
    """
    from matplotlib.figure import Figure

    model_path = Path("models/gpt2-small")
    checkpoint_path = Path("trained_model.pth")
    store_dir = Path("data/sample_stores/wikitext-103-train__gpt2-small__h64-f32")
    output_path = Path("next_token_approximation.png")
    top_k = 10
    num_reference_histories = 128
    seed = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gpt_model = CausalLMInference(model_path, device=device)
    model = load_model(checkpoint_path)
    sampler = SampleSampler(store_dir, seed=seed)
    ht = sampler.sample()[0]
    hn = torch.stack([sampler.sample()[0] for _ in range(num_reference_histories)])

    if top_k < 1:
        raise ValueError("top_k must be positive")
    approx_logits, true_logits = evaluate_next_token_approximation(
        gpt_model, model, hn, ht
    )
    approx_logits = approx_logits.detach().to(device="cpu", dtype=torch.float64)
    true_logits = true_logits.detach().to(device="cpu", dtype=torch.float64)
    if true_logits.ndim != 1 or approx_logits.shape != true_logits.shape or not true_logits.numel():
        raise ValueError("expected matching, nonempty vocabulary logit vectors")
    if not torch.isfinite(approx_logits).all() or not torch.isfinite(true_logits).all():
        raise ValueError("evaluation logits must be finite")

    approx_centered = approx_logits - approx_logits.mean()
    true_centered = true_logits - true_logits.mean()
    error = approx_centered - true_centered
    approx_log_probs = approx_logits.log_softmax(dim=0)
    true_log_probs = true_logits.log_softmax(dim=0)
    approx_probs, true_probs = approx_log_probs.exp(), true_log_probs.exp()
    k = min(top_k, true_logits.numel())
    true_top = true_logits.topk(k).indices
    approx_top = approx_logits.topk(k).indices
    metrics = {
        "centered_logit_rmse": error.square().mean().sqrt().item(),
        "centered_logit_mae": error.abs().mean().item(),
        "mean_logit_bias": (approx_logits.mean() - true_logits.mean()).item(),
        "kl_true_to_approx": (true_probs * (true_log_probs - approx_log_probs)).sum().item(),
        "total_variation": (0.5 * (true_probs - approx_probs).abs().sum()).item(),
        "top1_agreement": float(true_top[0] == approx_top[0]),
        f"top{k}_overlap": torch.isin(true_top, approx_top).double().mean().item(),
    }
    print(f"Next-token approximation: 1 target history, {hn.shape[0]} reference histories")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6g}")

    figure = Figure(figsize=(13, 5), constrained_layout=True)
    logit_axis, probability_axis = figure.subplots(1, 2)
    logit_axis.hexbin(
        true_centered.numpy(), approx_centered.numpy(), gridsize=60, mincnt=1,
        bins="log", cmap="viridis",
    )
    lower = min(true_centered.min().item(), approx_centered.min().item())
    upper = max(true_centered.max().item(), approx_centered.max().item())
    logit_axis.plot([lower, upper], [lower, upper], "r--", label="Perfect approximation")
    logit_axis.set(
        xlabel="True centered logit", ylabel="Approximated centered logit",
        title=f"Vocabulary logits (RMSE = {metrics['centered_logit_rmse']:.3g})",
    )
    logit_axis.legend()

    # Include tokens preferred by either distribution so false positives are visible.
    token_ids = torch.unique(torch.cat((true_top, approx_top)))
    token_ids = token_ids[true_probs[token_ids].argsort(descending=True)]
    positions = torch.arange(token_ids.numel()).numpy()
    probability_axis.bar(positions - 0.2, true_probs[token_ids].numpy(), 0.4, label="True")
    probability_axis.bar(positions + 0.2, approx_probs[token_ids].numpy(), 0.4, label="Approximation")
    probability_axis.set_xticks(positions, [str(token.item()) for token in token_ids], rotation=90)
    probability_axis.set(
        xlabel="Token ID", ylabel="Probability (full vocabulary softmax)",
        title=f"Union of top-{k} tokens",
    )
    probability_axis.legend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    print(f"Saved evaluation plots to {output_path}")
    return metrics

if __name__ == "__main__":
    report_next_token_approximation()
