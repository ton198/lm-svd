"""Fresh table entries from the frozen model, for studies B and C of exp8.

exp5-exp7 train on stored independent (history, future) pairs. A pair gives one
history per future, so the mean of a column over the histories is not known and
cannot be removed. Here the model is run on a full grid of histories x futures
at every step: each column then has an entry for every history of the batch, and
its mean over them can be subtracted exactly.

Windows are drawn like the sample store's: uniformly from the token stream, with
replacement, histories and futures independently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data_pipline.token_stream import TokenStream
from experiments.exp5_gpt2_mlp_table.features import future_only_logits, header_tokens
from utils.inference import CausalLMInference


def step_seed(seed: int, step: int) -> int:
    """Seed of the window draws of a run that starts or resumes at `step`."""
    return int(np.random.SeedSequence([seed, step]).generate_state(1)[0])


class Windows:
    """Random history and future windows of a token stream."""

    def __init__(self, stream_path: Path, history_tokens: int, future_tokens: int,
                 vocab_size: int, seed: int) -> None:
        self.stream = TokenStream(Path(stream_path))
        if self.stream.metadata.get("vocab_size") != vocab_size:
            raise ValueError(
                f"token stream {stream_path} has vocabulary "
                f"{self.stream.metadata.get('vocab_size')}, the model has {vocab_size}")
        self.history_tokens = history_tokens
        self.future_tokens = future_tokens
        self.generator = torch.Generator().manual_seed(seed)

    def draw(self, histories: int, futures: int) -> tuple[torch.Tensor, torch.Tensor]:
        """History ids (histories, H) and future ids (futures, F), on the CPU."""
        starts = self.stream.sample_starts(histories, self.history_tokens, self.generator)
        history_ids = self.stream.windows(starts, self.history_tokens)
        starts = self.stream.sample_starts(futures, self.future_tokens, self.generator)
        return history_ids, self.stream.windows(starts, self.future_tokens)


class Teacher:
    """Table entries computed by the frozen model.

    The entry of (history, future, token z) is the logit of z after
    history + future minus the mean logit over the vocabulary, which is
    <s(history + future), W_z - mean row of W> for the unembedding W.
    """

    def __init__(self, inference: CausalLMInference, chunk: int = 512) -> None:
        output = inference.model.get_output_embeddings()
        if output.bias is not None:
            raise ValueError("exp8 expects an unembedding without a bias, as GPT-2's is")
        weight = output.weight.detach().float()
        self.inference = inference
        self.rows = weight - weight.mean(dim=0)  # (V, E)
        self.chunk = chunk

    @torch.no_grad()
    def headers(self, future_ids: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Header tokens (F, K) of every future and their future-only entries g (F, K)."""
        hidden = torch.cat([
            self.inference.final_hidden_states(future_ids[start:start + self.chunk])[:, -1]
            for start in range(0, len(future_ids), self.chunk)
        ])
        return header_tokens(future_only_logits(self.inference, hidden), top_k)

    @torch.no_grad()
    def future_only(self, future_ids: torch.Tensor, header_ids: torch.Tensor) -> torch.Tensor:
        """g (F, K) at given header tokens."""
        hidden = torch.cat([
            self.inference.final_hidden_states(future_ids[start:start + self.chunk])[:, -1]
            for start in range(0, len(future_ids), self.chunk)
        ])
        return torch.einsum("fe,fke->fk", hidden, self.rows[header_ids.to(hidden.device)])

    @torch.no_grad()
    def pair_entries(
        self, history_ids: torch.Tensor, future_ids: torch.Tensor, header_ids: torch.Tensor
    ) -> torch.Tensor:
        """Entries (B, K) of pair b = (history b, future b) on future b's header tokens."""
        device = self.inference.device
        ids = torch.cat((history_ids, future_ids), dim=1)
        rows = self.rows[header_ids.to(device)]
        return torch.cat([
            torch.einsum(
                "be,bke->bk",
                self.inference.final_hidden_states(ids[start:start + self.chunk])[:, -1],
                rows[start:start + self.chunk])
            for start in range(0, len(ids), self.chunk)
        ])

    @torch.no_grad()
    def grid_entries(
        self, history_ids: torch.Tensor, future_ids: torch.Tensor, header_ids: torch.Tensor
    ) -> torch.Tensor:
        """Entries (N, F, K): every history with every future, on the future's header tokens.

        Each history is read once and its attention cache is shared by the F
        futures behind it. On a GPU the pass runs in half precision; on a
        32 x 16 grid of the stored table that changes the entries by under 0.03
        (they are about 11 in size) and the centered grid's energy by 2e-5.
        """
        device = self.inference.device
        base = self.inference.model.base_model
        history_ids, future_ids = history_ids.to(device), future_ids.to(device)
        histories, futures = len(history_ids), len(future_ids)
        rows = self.rows[header_ids.to(device)]
        group = max(self.chunk // futures, 1)
        blocks = []
        with torch.autocast("cuda", dtype=torch.float16, enabled=torch.device(device).type == "cuda"):
            for start in range(0, histories, group):
                part = history_ids[start:start + group]
                cache = base(input_ids=part, use_cache=True).past_key_values
                cache.batch_repeat_interleave(futures)
                hidden = base(
                    input_ids=future_ids.repeat(len(part), 1), past_key_values=cache,
                    use_cache=True,
                ).last_hidden_state[:, -1].float()
                blocks.append(torch.einsum(
                    "nfe,fke->nfk", hidden.unflatten(0, (len(part), futures)), rows.float()))
        return torch.cat(blocks).float()
