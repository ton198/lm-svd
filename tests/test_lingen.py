import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from utils.inference import CausalLMInference
from utils.lingen import compare_distributions, reconstructed_logits, teacher_logits
from utils.stream_neural_svd import TokenStream, centered_unembedding, column_features


class _TableFeatures(nn.Module):
    def __init__(self, rows: torch.Tensor, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("rows", rows)
        self.register_buffer("weights", weights)

    def row_features(self, ids: torch.Tensor) -> torch.Tensor:
        return self.rows[ids[:, 0]]

    def column_weights(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weights[ids[:, 0]]


class LingenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        temporary_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary_directory.cleanup)
        cls.root = Path(temporary_directory.name)
        cls.model_path = cls.root / "model"
        config = GPT2Config(
            vocab_size=11,
            n_positions=12,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=None,
            eos_token_id=None,
        )
        GPT2LMHeadModel(config).save_pretrained(cls.model_path)
        cls.stream_path = cls.root / "stream.npy"
        stream = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1], dtype=np.uint16)
        np.save(cls.stream_path, stream)
        cls.stream_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "config": "tiny",
                    "split": "train",
                    "tokenizer_path": str(cls.model_path),
                    "vocab_size": 11,
                    "stream_tokens": len(stream),
                    "dtype": str(stream.dtype),
                }
            ),
            encoding="utf-8",
        )
        cls.stream = TokenStream(cls.stream_path)
        cls.inference = CausalLMInference(cls.model_path, "cpu")
        cls.token_matrix = centered_unembedding(cls.inference)

    def test_reconstructed_logits_pair_each_history_with_its_future(self) -> None:
        generator = torch.Generator().manual_seed(42)
        rank = 3
        rows = torch.randn(11, rank, generator=generator)
        weights = torch.randn(11, rank, 8, generator=generator)
        features = _TableFeatures(rows, weights)
        history = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        future = torch.tensor([[4, 5], [7, 8], [1, 2]])

        actual = reconstructed_logits(features, self.token_matrix, history, future)
        for i in range(3):
            expected = column_features(
                features.column_weights(future[i : i + 1]), self.token_matrix
            ) @ features.row_features(history[i : i + 1])[0]
            torch.testing.assert_close(actual[i], expected, atol=1e-6, rtol=1e-5)

    def test_teacher_logits_give_gpt2_next_token_distribution(self) -> None:
        context = torch.tensor([[1, 2, 3, 4, 5]])
        centered = teacher_logits(self.inference, self.token_matrix, context)
        original = self.inference.next_token_logits_batch(context.tolist())
        torch.testing.assert_close(
            torch.log_softmax(centered, dim=-1),
            torch.log_softmax(original, dim=-1),
            atol=1e-5,
            rtol=0,
        )

    def test_compare_distributions_values(self) -> None:
        log3 = math.log(3)
        teacher = torch.tensor([[0.0, log3]], dtype=torch.float64)
        learned = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        result = compare_distributions(teacher, learned)
        p = (0.25, 0.75)
        q = (math.e / (math.e + 1), 1 / (math.e + 1))
        kl_pq = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
        kl_qp = sum(qi * math.log(qi / pi) for pi, qi in zip(p, q))
        self.assertAlmostEqual(result["kl_pq"].item(), kl_pq, places=12)
        self.assertAlmostEqual(result["kl_qp"].item(), kl_qp, places=12)
        torch.testing.assert_close(
            result["topk_overlap"],
            torch.tensor([[0.0, 1.0, 1.0, 1.0]], dtype=torch.float64),
        )
        self.assertAlmostEqual(
            result["logit_sq_error"].item(), 2 * (log3 / 2 + 0.5) ** 2, places=12
        )
        self.assertAlmostEqual(result["logit_sq_norm"].item(), 2 * (log3 / 2) ** 2, places=12)

        shifted = compare_distributions(teacher, teacher + 5)
        self.assertAlmostEqual(shifted["kl_pq"].item(), 0.0, places=12)
        self.assertAlmostEqual(shifted["kl_qp"].item(), 0.0, places=12)
        torch.testing.assert_close(
            shifted["topk_overlap"], torch.ones((1, 4), dtype=torch.float64)
        )
        self.assertAlmostEqual(shifted["logit_sq_error"].item(), 0.0, places=12)
