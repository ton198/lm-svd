import math
from dataclasses import dataclass

import numpy as np
import torch



def learned_feature_moments(
    history_weight: torch.Tensor,
    future_weight: torch.Tensor,
    token_gram: torch.Tensor,
    *,
    rank: int,
    hidden_size: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return float64 f_rms, g_rms, and right_gram on token_gram.device."""
    with torch.no_grad():
        device = token_gram.device
        history_features_t = history_weight.to(device=device, dtype=torch.float64)
        future_count = future_weight.shape[0]
        future_weights = future_weight.view(future_count, rank, hidden_size)
        f_rms_t = history_features_t.pow(2).mean(dim=0).sqrt()
        g_sq = torch.zeros(rank, dtype=torch.float64, device=device)
        right_gram_t = torch.zeros((rank, rank), dtype=torch.float64, device=device)
        for start in range(0, future_count, 1024):
            weights = future_weights[start : start + 1024].to(
                device=device, dtype=torch.float64
            )
            weighted_gram = (
                weights.reshape(-1, hidden_size) @ token_gram
            ).view_as(weights)
            g_sq += (weighted_gram * weights).sum(dim=(0, 2))
            right_gram_t += torch.einsum("cld,cmd->lm", weighted_gram, weights)
        g_rms = (g_sq / columns).sqrt()
        right_gram = right_gram_t / columns
        return f_rms_t, g_rms, right_gram


def learned_singular_values(
    f_rms: torch.Tensor, g_rms: torch.Tensor, *, history_count: int, columns: int
) -> torch.Tensor:
    """Matrix singular values in learned order from feature RMS moments."""
    return f_rms * g_rms * math.sqrt(history_count * columns)


@dataclass(frozen=True)
class LearnedSpectrum:
    singular_values: np.ndarray        # float64, (rank,), learned order
    left_singular_vectors: np.ndarray  # float64, (history_count, rank), unit-norm columns
    left_orthogonality_error: float
    right_orthogonality_error: float


def learned_spectrum(
    state: dict[str, torch.Tensor],
    token_gram: np.ndarray,
    *,
    history_count: int,
    columns: int,
    rank: int,
    hidden_size: int,
    device: str | torch.device,
) -> LearnedSpectrum:
    try:
        device = torch.device(device)
    except RuntimeError as error:
        raise ValueError(f"invalid device: {device}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use device='cpu'")
    token_gram = torch.from_numpy(token_gram).to(device=device, dtype=torch.float64)
    f_rms_t, g_rms_t, right_gram_t = learned_feature_moments(
        state["history.weight"],
        state["future.weight"],
        token_gram,
        rank=rank,
        hidden_size=hidden_size,
        columns=columns,
    )
    history_features = state["history.weight"].to(dtype=torch.float64).cpu().numpy()
    f_rms = f_rms_t.cpu().numpy()
    g_rms = g_rms_t.cpu().numpy()
    right_gram = right_gram_t.cpu().numpy()
    if np.any(f_rms == 0) or np.any(g_rms == 0):
        raise ValueError("every feature column must have nonzero norm")

    # Keep learned order, matching neural_svd_spectrum rather than sorting.
    singular_values = learned_singular_values(
        f_rms_t, g_rms_t, history_count=history_count, columns=columns
    ).cpu().numpy()
    left_singular_vectors = history_features / (f_rms * math.sqrt(history_count))
    left_gram = left_singular_vectors.T @ left_singular_vectors
    normalized_right_gram = right_gram / np.outer(g_rms, g_rms)
    np.fill_diagonal(left_gram, 0)
    np.fill_diagonal(normalized_right_gram, 0)
    left_orthogonality_error = float(np.max(np.abs(left_gram)))
    right_orthogonality_error = float(np.max(np.abs(normalized_right_gram)))
    return LearnedSpectrum(
        singular_values=singular_values,
        left_singular_vectors=left_singular_vectors,
        left_orthogonality_error=left_orthogonality_error,
        right_orthogonality_error=right_orthogonality_error,
    )
