from typing import Tuple

import torch
from torch import nn
import numpy as np


def get_joint_nesting_masks(weights: np.ndarray, set_first_mode_const: bool = False):
    vector_mask = list(np.cumsum(list(weights)[::-1])[::-1])
    if set_first_mode_const:
        vector_mask = [vector_mask[0]] + vector_mask
    vector_mask = torch.tensor(np.array(vector_mask)).float()
    matrix_mask = torch.minimum(vector_mask.unsqueeze(1), vector_mask.unsqueeze(1).T).float()
    return vector_mask, matrix_mask


def compute_lambda(f):
    return torch.einsum('bl...,bm...->lm', f, f) / f.shape[0]  # (L, L)

def get_sequential_nesting_masks(L, set_first_mode_const: bool = False):
    if set_first_mode_const:
        L += 1
    vector_mask = torch.ones(L)
    matrix_mask = torch.triu(torch.ones(L, L))
    return vector_mask, matrix_mask

def compute_loss_metric(f, g, matrix_mask):
    lam_f = compute_lambda(f)
    lam_g = compute_lambda(g)
    # compute loss_metric = E_{p(x)p(y)}[(f^T(x) g(y))^2]
    # f: (B1, L)
    # g: (B2, L)
    # lam_f, lam_g: (L, L)
    return (matrix_mask * lam_f * lam_g).sum(), lam_f, lam_g  # O(L ** 2)

def off_diagonal(x, start_dim=0):
    # return a flattened view of the off-diagonal elements of a batch of square matrices
    # (the last two dimensions must form square matrices)
    batch_shape = x.shape[:start_dim]
    n, m = x.shape[start_dim:]
    assert n == m
    return x.flatten(start_dim=start_dim)[..., :-1].view(*batch_shape, n - 1, n + 1)[..., 1:].flatten(start_dim=start_dim)


class NestedLoRALossFunctionSVD(torch.autograd.Function):
    # TODO: implement the matrix-valued kernel version
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
            ctx: torch.autograd.function.FunctionCtx,
            f,
            Tg,
            g,
            Tadjf,
            vector_mask,
            matrix_mask,
    ):
        """
        the reduction assumed here is `mean` (i.e., we take average over batch)
            f: (B1, L)
            Tg: (B1, L)
            g: (B2, L)
            Tadjf: (B2, L)
        """
        ctx.vector_mask = vector_mask
        ctx.matrix_mask = matrix_mask
        loss_metric, lam_f, lam_g = compute_loss_metric(f, g, matrix_mask)
        ctx.save_for_backward(f, Tg, g, Tadjf, lam_f, lam_g)
        # O(B^2 * L) version only
        # compute loss_operator = -2 * E_{p(x)}[f^T(x) (Tg)(x)]
        loss_operator = - 2 * torch.einsum('l,bl,bl->b', vector_mask, f, Tg).mean()  # O(B1 * L)
        loss = loss_operator + loss_metric
        return loss

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            ctx: The context object to retrieve saved tensors
            grad_output: The gradient of the loss with respect to the output
        """
        f, Tg, g, Tadjf, lam_f, lam_g = ctx.saved_tensors
        # for grad(f)
        operator_f = - (2 / f.shape[0]) * torch.einsum('l,bl->bl', ctx.vector_mask, Tg)
        metric_f = (2 / f.shape[0]) * torch.einsum('bi,il,il->bl', f, ctx.matrix_mask, lam_g)
        grad_f = operator_f + metric_f
        # for grad(g)
        operator_g = - (2 / g.shape[0]) * torch.einsum('l,bl->bl', ctx.vector_mask, Tadjf)
        metric_g = (2 / g.shape[0]) * torch.einsum('bi,il,il->bl', g, ctx.matrix_mask, lam_f)
        grad_g = operator_g + metric_g
        return grad_output * grad_f, None, grad_output * grad_g, None, None, None


class NestedLoRA(nn.Module):
    def __init__(
            self,
            model,
            neigs,
            step=1,
            sequential=False,
    ):
        self.name = 'nestedlora'
        super().__init__()
        self.neigs = neigs
        self.eigvals = None
        self.sort_indices = None
        self.sequential = sequential
        if sequential:
            vector_mask, matrix_mask = get_sequential_nesting_masks(self.neigs)
        else:
            end_indices = list(range(step, neigs + 1, step))
            if neigs not in end_indices:
                end_indices.append(neigs)
            step_weights = np.zeros(neigs)
            step_weights[np.array(end_indices) - 1] = 1.
            step_weights = torch.tensor(step_weights / sum(step_weights))
            vector_mask, matrix_mask = get_joint_nesting_masks(step_weights)
        self.register_buffer('vector_mask', vector_mask, persistent=False)
        self.register_buffer('matrix_mask', matrix_mask, persistent=False)
        self.model = model

    def forward(self, *args):
        output = self.model(*args)
        if self.sort_indices is not None and self.training:
            return output[:, self.sort_indices, ...]
        else:
            return output

    def _compute_loss(
            self,
            *args,
    ) -> torch.Tensor:
        return NestedLoRALossFunctionSVD.apply(
            *args,
            self.vector_mask,
            self.matrix_mask,
        )

    def compute_loss_kernel(
            self,
            f,
            g,
            K
    ):
        """
        Args:
            f: (B1, L)
            g: (B2, L)
            K: (B1, B2)
        """
        Tg = K @ g / g.shape[0]  # (B1, L)
        Tadjf = K.T @ f / f.shape[0]  # (B2, L)
        loss = self._compute_loss(f, Tg, g, Tadjf)
        return loss, dict(f=f, Tg=Tg, g=g, Tadjf=Tadjf)
