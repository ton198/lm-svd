from __future__ import annotations

import operator
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel


class CausalLMInference:
    def __init__(
        self,
        model_path: str | Path,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise ValueError(f"model directory does not exist: {self.model_path}")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available; use device='cpu'")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), local_files_only=True, dtype=torch.float32
            )
        except OSError as error:
            raise ValueError(
                f"cannot load local model from {self.model_path}: {error}"
            ) from error

        self.model: PreTrainedModel = model.to(self.device).eval()

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    @property
    def max_length(self) -> int:
        return self.model.config.max_position_embeddings

    def _validate_ids(self, token_ids: Sequence[int]) -> list[int]:
        ids: list[int] = []
        for token_id in token_ids:
            if isinstance(token_id, bool):
                raise ValueError("token_ids must be integers")
            try:
                ids.append(operator.index(token_id))
            except TypeError as error:
                raise ValueError("token_ids must be integers") from error

        if not ids:
            raise ValueError("token_ids must not be empty")
        if len(ids) > self.max_length:
            raise ValueError(
                f"token_ids length {len(ids)} exceeds model maximum {self.max_length}"
            )
        for token_id in ids:
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"token id {token_id} is outside [0, {self.vocab_size})"
                )
        return ids

    def next_token_logits(self, token_ids: Sequence[int]) -> torch.Tensor:
        ids = self._validate_ids(token_ids)

        with torch.no_grad():
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits = self.model(input_ids=input_ids, use_cache=False).logits[0, -1]
        return logits.to(device="cpu", dtype=torch.float32)

    def continuation_log_probs(
        self,
        context: Sequence[int],
        continuations: Sequence[Sequence[int]],
        batch_size: int = 64,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        context_ids = self._validate_ids(context)
        if not continuations:
            raise ValueError("continuations must not be empty")
        continuation_ids = [self._validate_ids(tokens) for tokens in continuations]
        total_length = len(context_ids) + max(len(tokens) for tokens in continuation_ids)
        if total_length > self.max_length:
            raise ValueError(
                f"context plus continuation length {total_length} exceeds model maximum {self.max_length}"
            )

        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            context_tensor = torch.tensor(
                context_ids, dtype=torch.long, device=self.device
            )
            for start in range(0, len(continuation_ids), batch_size):
                batch = continuation_ids[start : start + batch_size]
                lengths = [len(tokens) for tokens in batch]
                width = max(lengths)
                targets = torch.tensor(
                    [tokens + [0] * (width - len(tokens)) for tokens in batch],
                    dtype=torch.long,
                    device=self.device,
                )
                mask = torch.arange(width, device=self.device) < torch.tensor(
                    lengths, device=self.device
                )[:, None]
                input_ids = torch.cat(
                    (context_tensor.expand(len(targets), -1), targets), dim=1
                )
                logits = self.model(
                    input_ids=input_ids,
                    use_cache=False,
                    logits_to_keep=width + 1,
                ).logits[:, :-1]
                token_log_probs = logits.log_softmax(dim=-1).gather(
                    -1, targets.unsqueeze(-1)
                )
                chunks.append(
                    token_log_probs.squeeze(-1).masked_fill(~mask, 0).sum(dim=1).to(
                        device="cpu", dtype=torch.float64
                    )
                )
        return torch.cat(chunks)
