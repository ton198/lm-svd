"""Quantities exp5 takes from the frozen language model.

The networks see a history or a future only through its final hidden state at
the last token. The same pass over a future alone gives its header tokens and
their centered logits g, which exp5 uses as the fixed first mode.
"""

from __future__ import annotations

import torch

from utils.inference import CausalLMInference


@torch.no_grad()
def last_hidden(
    inference: CausalLMInference, ids: torch.Tensor, batch_size: int = 1024
) -> torch.Tensor:
    """Final hidden state (after ln_f) at the last token of every row of ids (N, T)."""
    return torch.cat([
        inference.final_hidden_states(ids[start:start + batch_size])[:, -1]
        for start in range(0, len(ids), batch_size)
    ])


@torch.no_grad()
def future_only_logits(inference: CausalLMInference, future_hidden: torch.Tensor) -> torch.Tensor:
    """Next-token logits (N, V) after each future alone, minus their vocabulary mean."""
    logits = inference.model.get_output_embeddings()(future_hidden).float()
    return logits - logits.mean(dim=-1, keepdim=True)


def header_tokens(centered: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The top_k tokens of every row and their values, ties going to the lower token id."""
    if not 1 <= top_k <= centered.shape[-1]:
        raise ValueError(f"top_k must be in [1, {centered.shape[-1]}]")
    # A stable sort keeps tied tokens in id order, as exp1's selection does.
    ids = torch.sort(centered, dim=-1, descending=True, stable=True).indices[..., :top_k]
    return ids, centered.gather(-1, ids).float()


def pair_targets(
    pair_logits: torch.Tensor, header_ids: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """Table entries of each (history, future) pair on its header tokens, minus g.

    pair_logits (B, V) are the raw logits after history + future.
    """
    centered = pair_logits - pair_logits.mean(dim=-1, keepdim=True)
    return centered.gather(-1, header_ids) - g
