"""The trainable part of exp6.

What exp5 trains on is bilinear in the header token. With s(x) the final hidden
state after x and e_z the unembedding row of token z minus the mean row,

    A[history, (future, z)] - g(future, z) = <s(history + future) - s(future), e_z>

so every right singular function is <w_l(future), e_z> for some vector
w_l(future). Exp6Network gives its right side that form: a trainable copy of
the transformer reads the future, and a head turns its last hidden state into
the vectors w_l(future), which meet the header token only in the inner product.

The head would need rank x hidden outputs. It is factorised instead:

    w_l(future) = B_l(future) P        right_l(future, z) = <B_l(future), P e_z> / sqrt(d)

P (d x hidden) is one learned projection of the unembedding rows, shared by all
modes and futures, and the head only gives B(future) (rank x d). A future has
top_k header tokens, so with d >= top_k the projection can still reach every
value on them.

The left side is exp5's finetuned one: a second trainable copy that reads the
history, with a linear head on its last hidden state.
"""

from __future__ import annotations

import torch
from torch import nn

from experiments.exp5_gpt2_mlp_table.model import linear_head, trainable_encoder
from utils.inference import CausalLMInference
from utils.nestedlora import NestedLoRA


class Exp6Network(nn.Module):
    def __init__(
        self,
        left_encoder: nn.Module,
        right_encoder: nn.Module,
        unembedding: torch.Tensor,
        hidden_size: int,
        rank: int,
        head_rank: int,
    ) -> None:
        """unembedding (V, E) is the frozen model's output embedding matrix."""
        super().__init__()
        self.rank = rank
        self.head_rank = head_rank
        self.left_encoder = left_encoder
        self.right_encoder = right_encoder
        self.left_head = linear_head(hidden_size, rank)
        self.right_head = linear_head(hidden_size, rank * head_rank)
        self.token_projection = nn.Linear(unembedding.shape[1], head_rank, bias=False)
        centered = unembedding.detach().float()
        centered = centered - centered.mean(dim=0)
        # Divided by one number, so the features stay linear in the rows. Not saved:
        # it is the frozen model's.
        self.register_buffer(
            "token_vectors", centered / centered.square().mean().sqrt(), persistent=False)

    def left_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        """(B, H) token ids -> (B, rank)."""
        hidden = self.left_encoder(input_ids=history_ids, use_cache=False).last_hidden_state
        return self.left_head(hidden[:, -1].float())

    def coefficients(self, future_ids: torch.Tensor) -> torch.Tensor:
        """B(future): (F, T) token ids -> (F, rank, head_rank)."""
        hidden = self.right_encoder(input_ids=future_ids, use_cache=False).last_hidden_state
        return self.right_head(hidden[:, -1].float()).reshape(-1, self.rank, self.head_rank)

    def right_features(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        """Features (F * top_k, rank); row f * top_k + j is header_ids[f, j] after future f."""
        tokens = self.token_projection(self.token_vectors[header_ids.to(future_ids.device)])
        features = torch.einsum("fkd,fld->fkl", tokens, self.coefficients(future_ids))
        return features.reshape(-1, self.rank) / self.head_rank ** 0.5

    def forward(
        self, history_ids: torch.Tensor, future_ids: torch.Tensor, header_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.left_features(history_ids), self.right_features(future_ids, header_ids)


def build_model(inference: CausalLMInference, rank: int, head_rank: int) -> NestedLoRA:
    """Exp6Network on two trainable copies of the model, under NestedLoRA."""
    return NestedLoRA(
        model=Exp6Network(
            trainable_encoder(inference), trainable_encoder(inference),
            inference.model.get_output_embeddings().weight,
            inference.model.config.hidden_size, rank, head_rank,
        ),
        neigs=rank,
    )
