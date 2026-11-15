"""NestedLoRA loss for independent (history, future) pairs.

A pair gives the table entries of one history on the header slots of one
future, so the operator term can only use those entries. The metric term needs
no entries and uses every history and every column of the batch.
"""

from __future__ import annotations

import torch

from utils.nestedlora import NestedLoRA


def paired_loss(
    model: NestedLoRA, left: torch.Tensor, right: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Loss for left (B, L), right (B * top_k, L) and entries target (B, top_k).

    Row i * top_k + j of right is slot j of pair i's future.
    """
    batch, top_k = target.shape
    if left.shape[0] != batch or right.shape[0] != target.numel():
        raise ValueError(
            f"features {tuple(left.shape)} and {tuple(right.shape)} do not match "
            f"{batch} pairs of {top_k} header slots"
        )
    # Zero outside each pair's own columns; the factor B undoes the average over
    # the B histories, of which only one has entries in a given column.
    kernel = target.new_zeros(batch, batch, top_k)
    pairs = torch.arange(batch, device=target.device)
    kernel[pairs, pairs] = batch * target
    return model.compute_loss_kernel(left, right, kernel.reshape(batch, batch * top_k))[0]
