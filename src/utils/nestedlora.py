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


class NestedLoRALossFunctionEVD(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(
            ctx: torch.autograd.function.FunctionCtx,
            f,
            Tf,
            f1,
            f2,
            vector_mask,
            matrix_mask,
    ):
        """
        the reduction assumed here is `mean` (i.e., we take average over batch)
            f: (B, L) or (B, L, O)
            Tf: (B, L) or (B, L, O)
            f1: (B1, L) or (B1, L, O)
            f2: (B2, L) or (B2, L, O)
        warning: f1 and f2 must be independent
        """
        ctx.vector_mask = vector_mask = vector_mask.to(f.device)
        ctx.matrix_mask = matrix_mask = matrix_mask.to(f.device)
        loss_metric, lam_f1, lam_f2 = compute_loss_metric(f1, f2, matrix_mask)
        ctx.save_for_backward(f, Tf, f1, f2, lam_f1, lam_f2)
        # compute loss_operator = -2 * E_{p(x)}[\sum_{l=1}^L f_l^T(x) (Tf_l)(x)]
        loss_operator = - 2 * torch.einsum('l,bl...,bl...->b', vector_mask, f, Tf).mean()  # O(B1 * L * O)
        loss = loss_operator + loss_metric
        return loss

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor | None, ...]:
        """
        Args:
            ctx: The context object to retrieve saved tensors
            grad_outputs: The gradient of the loss with respect to the output
        """
        f, Tf, f1, f2, lam_f1, lam_f2 = ctx.saved_tensors
        operator_f = - (4 / f.shape[0]) * torch.einsum('l,bl...->bl...', ctx.vector_mask, Tf)
        metric_f1 = (2 / f1.shape[0]) * torch.einsum('lm,lm,bl...->bm...', ctx.matrix_mask, lam_f2, f1)
        metric_f2 = (2 / f2.shape[0]) * torch.einsum('lm,lm,bl...->bm...', ctx.matrix_mask, lam_f1, f2)
        return grad_outputs * operator_f, None, grad_outputs * metric_f1, grad_outputs * metric_f2, None, None


class NestedLoRALossFunctionSVD(torch.autograd.Function):
    # TODO: implement the matrix-valued kernel version
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
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
        ctx.vector_mask = vector_mask = vector_mask.to(f.device)
        ctx.matrix_mask = matrix_mask = matrix_mask.to(f.device)
        loss_metric, lam_f, lam_g = compute_loss_metric(f, g, matrix_mask)
        ctx.save_for_backward(f, Tg, g, Tadjf, lam_f, lam_g)
        # O(B^2 * L) version only
        # compute loss_operator = -2 * E_{p(x)}[f^T(x) (Tg)(x)]
        loss_operator = - 2 * torch.einsum('l,bl,bl->b', vector_mask, f, Tg).mean()  # O(B1 * L)
        loss = loss_operator + loss_metric
        return loss

class NestedLoRA(nn.Module):
    def __init__(
            self,
            model : nn.Module,
            neigs : int,
            step : int =1,
            sequential : bool =False,
    ):
        self.name = 'nestedlora'
        super().__init__()
        self.neigs = neigs
        self.eigvals = None
        self.sort_indices = None
        self.sequential = sequential
        if sequential:
            self.vector_mask, self.matrix_mask = get_sequential_nesting_masks(self.neigs)
        else:
            end_indices = list(range(step, neigs + 1, step))
            if neigs not in end_indices:
                end_indices.append(neigs)
            step_weights = np.zeros(neigs)
            step_weights[np.array(end_indices) - 1] = 1.
            step_weights = torch.tensor(step_weights / sum(step_weights))
            self.vector_mask, self.matrix_mask = get_joint_nesting_masks(step_weights)
        self.model = model

    def forward(self, *args):
        output = self.model(*args)
        if self.sort_indices is not None and self.training:
            return output[:, self.sort_indices, ...]
        else:
            return output

    def register_eigvals(self, eigvals):
        print("NOTE: eigenvalues have been registered!")
        self.eigvals = torch.Tensor(eigvals)  # (L, )
        self.sort_indices = torch.sort(self.eigvals)[1].flip(0)

    def reset_eigvals(self):
        print("NOTE: eigenvalues have been reset!")
        self.eigvals = None
        self.sort_indices = None

    def _compute_loss(
            self,
            *args,
            evd=True,
    ) -> torch.Tensor:
        if evd:
            return NestedLoRALossFunctionEVD.apply(
                *args,
                self.vector_mask,
                self.matrix_mask,
            )
        else:
            return NestedLoRALossFunctionSVD.apply(
                *args,
                self.vector_mask,
                self.matrix_mask,
            )

    def compute_loss_kernel(
            self,
            get_approx_kernel_op,
            x,
            importance,
            split_batch: bool,
            evd: bool = True,
    ):
        if evd:
            if split_batch:
                x1, x2 = torch.chunk(x, 2)
                Kf1, f1 = get_approx_kernel_op(x2)(self, x1, importance=importance)
                f2 = self.model(x2)
                loss = self._compute_loss(f1, Kf1, f1, f2, evd=True)
                f = f1
                Kf = Kf1
            else:
                Kf, f = get_approx_kernel_op(x)(self, x, importance=importance)
                f1, f2 = torch.chunk(f, 2)
                loss = self._compute_loss(f, Kf, f1, f2, evd=True)
            return loss, dict(f=f, Tf=Kf, eigvals=None)
        else:
            raise NotImplementedError

    def compute_loss_operator(
            self,
            operator,
            x,
            importance=None,
            evd: bool = True,
    ):
        if evd:
            Tf, f = operator(self, x, importance=importance)
            f1, f2 = torch.chunk(f, 2)
            loss = self._compute_loss(f, Tf, f1, f2, evd=True)
            return loss, dict(f=f, Tf=Tf, eigvals=None)
        else:
            raise NotImplementedError
