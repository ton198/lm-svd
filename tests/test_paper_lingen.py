import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.exp1_L_M_table.table import select_header_tokens
from experiments.exp14_paper_lingen.lingen import (
    build_sets,
    neural_svd_coefficients,
    regress,
    run_lingen_variant,
    selected_centered_logits,
    weighted_basis_hidden,
)
from utils.common import Sample
from utils.inference import CausalLMInference
from utils.stream_neural_svd import centered_unembedding


class _TableFeatures(nn.Module):
    def __init__(self, rows: torch.Tensor, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("rows", rows)
        self.register_buffer("weights", weights)

    def row_features(self, ids: torch.Tensor) -> torch.Tensor:
        return self.rows[ids[:, 0]]

    def column_weights(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weights[ids[:, 0]]


class PaperLinGenTests(unittest.TestCase):
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
            vocab_size=11,
            n_positions=12,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=None,
            eos_token_id=None,
        )
        GPT2LMHeadModel(config).save_pretrained(model_path)
        cls.inference = CausalLMInference(model_path, "cpu")
        cls.token_matrix = centered_unembedding(cls.inference)
        cls.stream = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1], dtype=torch.int64)

    def test_cached_basis_hidden_matches_full_recompute(self) -> None:
        basis = torch.stack([self.stream[i : i + 4] for i in range(3)])
        continuation = torch.stack([self.stream[i : i + 2] for i in (5, 7)])
        weights = torch.tensor([[0.2, -0.5, 1.1], [-0.7, 0.3, 0.4]], dtype=torch.float64)
        for suffix in (None, continuation):
            with self.subTest(continuation=suffix is not None):
                actual = weighted_basis_hidden(
                    self.inference, basis, suffix, weights, chunk=2
                )
                expected = []
                for g in range(len(weights)):
                    sequences = basis if suffix is None else torch.cat(
                        (basis, suffix[g].expand(len(basis), -1)), dim=1
                    )
                    hidden = self.inference.final_hidden_states(sequences)[:, -1].double()
                    expected.append(torch.einsum("hd,h->d", hidden, weights[g]))
                self.assertEqual(actual.dtype, torch.float64)
                torch.testing.assert_close(actual, torch.stack(expected), atol=1e-5, rtol=0)

    def test_regression_reproduces_target_contained_in_basis(self) -> None:
        basis = torch.stack([self.stream[i : i + 4] for i in range(4)])
        futures = torch.stack([self.stream[i : i + 2] for i in (5, 7, 9)])
        samples = [Sample(i, ids.tolist()) for i, ids in enumerate(futures)]
        headers, _ = select_header_tokens(self.inference, samples, top_k=3)
        rows = selected_centered_logits(
            self.inference, self.token_matrix, basis, futures,
            torch.as_tensor(headers, dtype=torch.int64), pair_batch=4,
        )
        self.assertEqual(rows.shape, (4, 9))
        self.assertEqual(rows.dtype, torch.float32)
        self.assertEqual(rows.device.type, "cpu")
        # Independently compute each future-major, rank-minor column.
        for h, history in enumerate(basis):
            for f, future in enumerate(futures):
                sequence = torch.cat((history, future)).unsqueeze(0)
                hidden = self.inference.final_hidden_states(sequence)[0, -1]
                logits = hidden @ self.token_matrix.T
                torch.testing.assert_close(
                    rows[h, f * 3 : (f + 1) * 3],
                    logits[torch.as_tensor(headers[f], dtype=torch.int64)],
                    atol=1e-6, rtol=1e-5,
                )
        target_rows = selected_centered_logits(
            self.inference, self.token_matrix, basis[:1], futures,
            torch.as_tensor(headers, dtype=torch.int64), pair_batch=4,
        )
        coefficients, residual, singular_values = regress(rows.numpy(), target_rows.numpy())
        self.assertEqual(coefficients.shape, (1, 4))
        self.assertEqual(residual.shape, (1,))
        self.assertEqual(singular_values.shape, (4,))
        self.assertLess(residual[0], 1e-10)

    def test_one_hot_coefficients_give_zero_kl(self) -> None:
        basis = torch.stack([self.stream[i : i + 4] for i in range(3)])
        result = run_lingen_variant(
            self.inference, self.token_matrix,
            basis_ids=basis, target_ids=basis[1:2],
            coefficients=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64),
            samples_per_target=2, steps=3, chunk=2,
            generator=torch.Generator(device="cpu").manual_seed(12),
        )
        for key in ("kl_pq", "kl_qp", "logit_sq_error", "logit_sq_norm", "tokens"):
            self.assertEqual(result[key].shape, (2, 3))
        self.assertEqual(result["topk_overlap"].shape, (2, 3, 4))
        self.assertTrue(np.all(result["kl_pq"] <= 1e-8))
        np.testing.assert_allclose(result["topk_overlap"], 1.0)

    def test_build_sets_permutation_order_and_count_boundaries(self) -> None:
        histories = [Sample(10 + i, [2 * i, 2 * i + 1]) for i in range(4)]
        futures = [Sample(20 + i, [20 + 2 * i, 21 + 2 * i]) for i in range(3)]
        sets = build_sets(histories, futures, basis_histories=2, targets=2,
                          future_count=2, nonsense_seed=9)
        np.testing.assert_array_equal(sets["basis"], [[0, 1], [2, 3]])
        np.testing.assert_array_equal(sets["targets"], [[4, 5], [6, 7]])
        np.testing.assert_array_equal(sets["futures"], [[20, 21], [22, 23]])
        np.testing.assert_array_equal(sets["gram_futures"], [[20, 21], [22, 23], [24, 25]])
        np.testing.assert_array_equal(sets["basis_sample_ids"], [10, 11])
        np.testing.assert_array_equal(sets["target_sample_ids"], [12, 13])
        np.testing.assert_array_equal(sets["future_sample_ids"], [20, 21])
        rng = np.random.default_rng(9)
        for key in ("basis", "futures", "gram_futures"):
            source = sets[key]
            expected = source.ravel()[rng.permutation(source.size)].reshape(source.shape)
            np.testing.assert_array_equal(sets[key + "_nonsense"], expected)
            self.assertEqual(source.dtype, np.int64)
            self.assertEqual(sets[key + "_nonsense"].dtype, np.int64)
        self.assertEqual(sets["targets"].dtype, np.int64)
        with self.assertRaisesRegex(ValueError, "need 5 histories but the dataset has 4"):
            build_sets(histories, futures, basis_histories=3, targets=2,
                       future_count=2, nonsense_seed=9)
        with self.assertRaisesRegex(ValueError, "need 4 futures but the dataset has 3"):
            build_sets(histories, futures, basis_histories=2, targets=2,
                       future_count=4, nonsense_seed=9)

    def test_neural_svd_population_regression(self) -> None:
        rows = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        weights = torch.tensor(
            [[[1.0, 0.5], [0.2, 1.0]], [[0.5, 1.0], [1.5, -0.3]],
             [[0.2, -0.4], [0.8, 0.7]]], dtype=torch.float64,
        )
        token_matrix = torch.tensor([[1.0, -0.5], [-0.4, 1.0], [0.2, 0.3]], dtype=torch.float64)
        features = _TableFeatures(rows, weights)
        basis = torch.tensor([[0]], dtype=torch.int64)
        targets = torch.tensor([[1], [2]], dtype=torch.int64)
        futures = torch.tensor([[0], [1], [2]], dtype=torch.int64)
        actual_v, actual_residual = neural_svd_coefficients(
            features, token_matrix, basis, targets, futures, batch=2
        )
        # Materialize the complete future-by-vocabulary design independently.
        columns = torch.einsum("frd,zd->frz", weights, token_matrix)
        basis_table = torch.einsum("hr,frz->hfz", rows[basis[:, 0]], columns).reshape(1, -1).numpy()
        target_table = torch.einsum("tr,frz->tfz", rows[targets[:, 0]], columns).reshape(2, -1).numpy()
        expected_v, expected_residual, _ = regress(basis_table, target_table)
        np.testing.assert_allclose(actual_v, expected_v, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(actual_residual, expected_residual, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
