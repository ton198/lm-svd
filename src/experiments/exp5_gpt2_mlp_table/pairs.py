"""Training pairs made in the step, without a sample store.

Every draw takes fresh history windows and fresh future windows from the token
stream, independently of each other, and runs the frozen language model on each
history + future. The target of a pair is its table entry on the future's
header tokens minus g, as in the evaluation tables. No pair is stored, so a
run never sees one twice except by chance.
"""

from __future__ import annotations

from pathlib import Path

import torch

from data_pipline.token_stream import TokenStream
from utils.inference import CausalLMInference

from .features import future_only_logits, header_tokens, last_hidden, pair_targets


def open_stream(stream_path: Path, inference: CausalLMInference) -> TokenStream:
    """The token stream at stream_path; rejects a missing one and another vocabulary."""
    stream_path = Path(stream_path)
    if not stream_path.is_file():
        raise ValueError(
            f"no token stream at {stream_path}; build it with `data-pipline build-token-stream`")
    stream = TokenStream(stream_path)
    if stream.metadata.get("vocab_size") != inference.vocab_size:
        raise ValueError(
            f"token stream {stream_path} has vocabulary {stream.metadata.get('vocab_size')}, "
            f"the model has {inference.vocab_size}")
    return stream


class StreamPairs:
    """Independent (history, future) windows of the stream and their entries minus g."""

    def __init__(
        self, inference: CausalLMInference, stream_path: Path, *, history_tokens: int,
        future_tokens: int, top_k: int, seed: int, encode_batch: int,
    ) -> None:
        self.inference = inference
        self.stream = open_stream(stream_path, inference)
        self.history_tokens = history_tokens
        self.future_tokens = future_tokens
        self.top_k = top_k
        self.encode_batch = encode_batch
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def draw(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Histories (B, H), futures (B, T), header ids (B, K) and targets (B, K).

        The header tokens are the top-k tokens after the future alone and g their
        centered logits there; the target is the centered logit of each header
        token after history + future, minus g.
        """
        device = self.inference.device
        histories = self.stream.windows(
            self.stream.sample_starts(count, self.history_tokens, self.generator),
            self.history_tokens)
        futures = self.stream.windows(
            self.stream.sample_starts(count, self.future_tokens, self.generator),
            self.future_tokens)
        future_hidden = last_hidden(self.inference, futures, self.encode_batch)
        header_ids, g = header_tokens(
            future_only_logits(self.inference, future_hidden), self.top_k)
        state = last_hidden(
            self.inference, torch.cat([histories, futures], dim=1), self.encode_batch)
        logits = self.inference.model.get_output_embeddings()(state).float()
        return histories.to(device), futures.to(device), header_ids, pair_targets(
            logits, header_ids, g)
