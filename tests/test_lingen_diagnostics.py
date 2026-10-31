import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.exp15_lingen_diagnostics.diagnostics import (
    fit_diagonal_scale,
    fit_temperature,
    history_gram,
    optimal_left,
    teacher_hidden_grid,
)
from experiments.exp15_lingen_diagnostics.surrogate import anchored_constrained_solve
from utils.inference import CausalLMInference
from utils.stream_neural_svd import centered_logit_block, centered_unembedding


class _TableFeatures(nn.Module):
    def __init__(self, rows: torch.Tensor, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("rows", rows)
        self.register_buffer("weights", weights)

    def row_features(self, ids: torch.Tensor) -> torch.Tensor:
        return self.rows[ids[:, 0]]

    def column_weights(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weights[ids[:, 0]]


class LinGenDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        temporary_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary_directory.cleanup)
        model_path = Path(temporary_directory.name) / "model"
        config = GPT2Config(
            vocab_size=11, n_positions=12, n_embd=8, n_layer=1, n_head=2,
            bos_token_id=None, eos_token_id=None,
        )
        GPT2LMHeadModel(config).save_pretrained(model_path)
        cls.inference = CausalLMInference(model_path, "cpu")
        cls.token_matrix = centered_unembedding(cls.inference)
        cls.token_gram = cls.token_matrix.double().T @ cls.token_matrix.double()
        stream = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1])
        cls.histories = torch.stack([stream[i : i + 4] for i in range(5)])
        cls.futures = torch.stack([stream[i : i + 2] for i in (6, 8, 9)])
        torch.manual_seed(0)
        cls.features = _TableFeatures(torch.randn(11, 2), torch.randn(11, 2, 8))

    def test_teacher_grid_and_oracle_match_materialized_svd(self) -> None:
        states = teacher_hidden_grid(self.inference, self.histories, self.futures, pair_batch=6)
        self.assertEqual(states.shape, (3, 5, 8))
        self.assertEqual(states.dtype, torch.float32)
        self.assertEqual(states.device, self.inference.device)
        logits = (states @ self.token_matrix.T).permute(1, 0, 2).reshape(5, -1)
        block = centered_logit_block(self.inference, self.token_matrix, self.histories, self.futures)
        torch.testing.assert_close(logits, block, atol=1e-5, rtol=0)

        gram = history_gram(states, self.token_gram)
        _, eigenvectors = torch.linalg.eigh(gram)
        basis = eigenvectors[:, -2:]
        projected = torch.einsum("ij,fjd->fid", basis @ basis.T, states.double())
        reconstructed = (projected @ self.token_matrix.double().T).permute(1, 0, 2).reshape(5, -1)
        materialized = (states.double() @ self.token_matrix.double().T).permute(1, 0, 2).reshape(5, -1).numpy()
        u, s, vh = np.linalg.svd(materialized, full_matrices=False)
        expected = (u[:, :2] * s[:2]) @ vh[:2]
        np.testing.assert_allclose(reconstructed.numpy(), expected, atol=1e-4, rtol=0)

        empty_futures = self.futures[:1, :0]
        no_future = teacher_hidden_grid(self.inference, self.histories, empty_futures, pair_batch=2)
        self.assertEqual(no_future.shape, (1, 5, 8))
        expected_hidden = self.inference.final_hidden_states(self.histories)[:, -1]
        torch.testing.assert_close(no_future[0], expected_hidden, atol=1e-6, rtol=0)

    def test_optimal_left_matches_materialized_least_squares(self) -> None:
        states = teacher_hidden_grid(self.inference, self.histories, self.futures, pair_batch=6)
        weights = self.features.column_weights(self.futures)
        actual = optimal_left(weights, states, self.token_gram)
        tokens = self.token_matrix.double()
        design = (weights.double() @ tokens.T).permute(0, 2, 1).reshape(-1, 2).numpy()
        targets = (states.double() @ tokens.T).permute(0, 2, 1).reshape(-1, 5).numpy()
        expected = np.linalg.lstsq(design, targets, rcond=None)[0].T
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)

    def test_diagonal_scale_recovers_known_scale(self) -> None:
        rows = self.features.row_features(self.histories).double()
        weights = self.features.column_weights(self.futures).double()
        scale = torch.tensor([1.7, -0.4], dtype=torch.float64)
        states = torch.einsum("hr,frd->fhd", rows * scale, weights)
        actual = fit_diagonal_scale(rows, weights, states, self.token_gram)
        np.testing.assert_allclose(actual, scale.numpy(), atol=1e-8, rtol=0)

    def test_fit_temperature_recovers_logit_scale(self) -> None:
        generator = torch.Generator().manual_seed(0)
        approx = torch.randn((16, 11), generator=generator, dtype=torch.float64)
        teacher = 2.5 * approx
        actual = fit_temperature(
            lambda: iter((teacher[:7], teacher[7:])),
            lambda: iter((approx[:7], approx[7:])),
        )
        np.testing.assert_allclose(actual, 2.5, rtol=1e-3, atol=0)

    def test_anchor_nsvd_keeps_constraint_and_beats_min_norm(self) -> None:
        rng = np.random.default_rng(0)
        u_h = rng.standard_normal((6, 2))
        factor = rng.standard_normal((2, 2))
        gram = factor @ factor.T + np.eye(2)
        values, vectors = np.linalg.eigh(gram)
        root = (vectors * np.sqrt(values)) @ vectors.T
        constraint = root @ u_h.T
        target = root @ rng.standard_normal((2, 1))
        a = rng.standard_normal((20, 6))
        b = rng.standard_normal((20, 1))
        v0 = np.linalg.lstsq(constraint, target, rcond=None)[0].T
        result = anchored_constrained_solve(a, b, constraint, v0)
        self.assertLess(np.linalg.norm(constraint @ result.T - target), 1e-10)
        self.assertLessEqual(np.linalg.norm(a @ result.T - b), np.linalg.norm(a @ v0.T - b) + 1e-12)
        kkt = np.block([[a.T @ a, constraint.T], [constraint, np.zeros((2, 2))]])
        rhs = np.concatenate((a.T @ b, target), axis=0)
        expected = np.linalg.solve(kkt, rhs)[:6].T
        np.testing.assert_allclose(result, expected, atol=1e-8, rtol=0)


if __name__ == "__main__":
    unittest.main()
