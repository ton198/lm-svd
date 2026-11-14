import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from data_pipline.sample_store.store import STORE_FORMAT, SampleSampler, sample_dtype, slot_path
from experiments.exp1_L_M_table.table import build_matrix, select_header_tokens
from experiments.svd_inference.finite_table import (
    build_rows,
    deflated_relative_error,
    fit_coefficients,
    report_finite_table_baseline,
    sample_distinct,
    spectral_fit,
)
from experiments.svd_inference.lingen import summarize_next_token_approximation
from utils.common import Sample
from utils.inference import CausalLMInference

VOCAB = 17
HISTORY_TOKENS = 2
FUTURE_TOKENS = 3
BLOCKS = 4
BLOCK_SIZE = 4
ROWS = BLOCKS * BLOCK_SIZE


def _rows(tensor: torch.Tensor) -> set[tuple[int, ...]]:
    return {tuple(row) for row in tensor.tolist()}


class FiniteTableFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary_directory.name)
        cls.model_path = cls.root / "model"
        GPT2LMHeadModel(GPT2Config(
            vocab_size=VOCAB, n_positions=8, n_embd=8, n_layer=1, n_head=2,
            bos_token_id=None, eos_token_id=None,
        )).save_pretrained(cls.model_path)

        # 16 rows with pairwise distinct histories and pairwise distinct futures.
        cls.store = cls.root / "store"
        (cls.store / "slots").mkdir(parents=True)
        (cls.store / "store.json").write_text(json.dumps({
            "format": STORE_FORMAT, "history_tokens": HISTORY_TOKENS,
            "future_tokens": FUTURE_TOKENS, "vocab_size": VOCAB,
            "samples_per_block": BLOCK_SIZE, "num_slots": BLOCKS,
        }), encoding="utf-8")
        (cls.store / "state.json").write_text(json.dumps({
            "workers": {}, "published": {str(slot): slot for slot in range(BLOCKS)},
            "next_commit": BLOCKS,
        }), encoding="utf-8")
        records = np.zeros(ROWS, dtype=sample_dtype(HISTORY_TOKENS, FUTURE_TOKENS, VOCAB))
        for row in range(ROWS):
            records["h"][row] = [row, (row + 1) % VOCAB]
            records["f"][row] = [(row + 2) % VOCAB, (row + 5) % VOCAB, row]
        for slot in range(BLOCKS):
            np.save(slot_path(cls.store, slot), records[slot * BLOCK_SIZE:(slot + 1) * BLOCK_SIZE])
        cls.records = records

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()


class SampleDistinctTests(FiniteTableFixture):
    def test_returns_requested_number_of_pairwise_distinct_rows(self):
        histories = sample_distinct(SampleSampler(self.store, seed=3), ROWS, "h")
        self.assertEqual(histories.shape, (ROWS, HISTORY_TOKENS))
        self.assertEqual(histories.dtype, torch.int64)
        self.assertEqual(_rows(histories), {tuple(row) for row in self.records["h"].tolist()})

    def test_future_field_returns_futures(self):
        futures = sample_distinct(SampleSampler(self.store, seed=3), 5, "f")
        self.assertEqual(futures.shape, (5, FUTURE_TOKENS))
        self.assertEqual(len(_rows(futures)), 5)
        self.assertLessEqual(_rows(futures), {tuple(row) for row in self.records["f"].tolist()})

    def test_never_returns_an_excluded_row(self):
        sampler = SampleSampler(self.store, seed=5)
        targets = sample_distinct(sampler, 4, "h")
        basis = sample_distinct(sampler, ROWS - 4, "h", exclude=targets)
        self.assertEqual(len(_rows(basis)), ROWS - 4)
        self.assertFalse(_rows(basis) & _rows(targets))

    def test_fails_when_the_store_has_too_few_distinct_rows(self):
        sampler = SampleSampler(self.store, seed=5)
        targets = sample_distinct(sampler, 4, "h")
        with self.assertRaisesRegex(ValueError, "distinct"):
            sample_distinct(sampler, ROWS - 3, "h", exclude=targets)

    def test_rejects_unknown_field(self):
        with self.assertRaises(ValueError):
            sample_distinct(SampleSampler(self.store, seed=0), 1, "logits")


class BuildRowsTests(FiniteTableFixture):
    def test_matches_the_exp1_matrix_entry_by_entry(self):
        inference = CausalLMInference(self.model_path, "cpu")
        histories = torch.from_numpy(self.records["h"][:5].copy())
        futures = torch.from_numpy(self.records["f"][:4].copy())
        future_samples = [Sample(index, row) for index, row in enumerate(futures.tolist())]
        header_token_ids, _ = select_header_tokens(inference, future_samples, 3)
        expected = build_matrix(
            inference,
            [Sample(index, row) for index, row in enumerate(histories.tolist())],
            future_samples,
            header_token_ids,
        )
        # batch_size=3 splits the four futures unevenly across two batches.
        for batch_size in (1, 3, 64):
            with self.subTest(batch_size=batch_size):
                actual = build_rows(inference, histories, futures, header_token_ids, batch_size)
                self.assertEqual(actual.shape, (5, 4 * 3))
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


class FitCoefficientsTests(unittest.TestCase):
    def test_recovers_coefficients_of_an_exact_combination(self):
        generator = torch.Generator().manual_seed(0)
        table = torch.randn(6, 40, generator=generator, dtype=torch.float64)
        true_coefficients = torch.randn(3, 6, generator=generator, dtype=torch.float64)
        coefficients, residual = fit_coefficients(
            table.float().numpy(), (true_coefficients @ table.float().double()).numpy()
        )
        self.assertEqual(coefficients.dtype, torch.float64)
        torch.testing.assert_close(coefficients, true_coefficients, rtol=0, atol=1e-8)
        self.assertEqual(residual.shape, (3,))
        self.assertLess(residual.max().item(), 1e-16)

    def test_reports_relative_squared_residual_of_an_inexact_fit(self):
        table = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        target = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float64)
        coefficients, residual = fit_coefficients(table, target)
        torch.testing.assert_close(coefficients, torch.tensor([[3.0]], dtype=torch.float64))
        # Only the first of two nonzero target entries can be fitted: 16 / 25.
        torch.testing.assert_close(residual, torch.tensor([16 / 25], dtype=torch.float64))

    def test_duplicate_basis_rows_share_the_weight(self):
        row = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float64)
        other = torch.tensor([0.0, 1.0, 1.0, -2.0], dtype=torch.float64)
        table = torch.stack((row, row, other))
        coefficients, residual = fit_coefficients(table, (2 * row + other).unsqueeze(0))
        # The minimum-norm solution splits the weight 2 evenly over the duplicates.
        torch.testing.assert_close(
            coefficients, torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64),
            rtol=0, atol=1e-10,
        )
        self.assertLess(residual.item(), 1e-20)


class SpectralFitTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(1)
        self.table = torch.randn(6, 40, generator=generator, dtype=torch.float64)
        self.targets = torch.randn(3, 40, generator=generator, dtype=torch.float64)
        self.U, self.S, self.Vh = torch.linalg.svd(self.table, full_matrices=False)

    def test_all_directions_reproduce_the_least_squares_fit(self):
        basis_directions, weights = spectral_fit(self.table, self.targets)
        expected, _ = fit_coefficients(self.table, self.targets)
        self.assertEqual(weights.shape, (3, 6))
        torch.testing.assert_close(weights @ basis_directions.T, expected, rtol=0, atol=1e-10)

    def test_first_k_directions_fit_the_rank_k_truncated_table(self):
        basis_directions, weights = spectral_fit(self.table, self.targets)
        for rank in (1, 2, 5):
            with self.subTest(rank=rank):
                truncated = (self.U[:, :rank] * self.S[:rank]) @ self.Vh[:rank]
                expected, _ = fit_coefficients(truncated, self.targets)
                torch.testing.assert_close(
                    weights[:, :rank] @ basis_directions[:, :rank].T, expected,
                    rtol=0, atol=1e-10,
                )

    def test_deflate1_fit_is_the_full_fit_without_its_leading_term(self):
        basis_directions, weights = spectral_fit(self.table, self.targets)
        leading = self.S[0] * torch.outer(self.U[:, 0], self.Vh[0])
        deflated_targets = self.targets - torch.outer(self.targets @ self.Vh[0], self.Vh[0])
        deflated, _ = fit_coefficients(self.table - leading, deflated_targets)
        torch.testing.assert_close(
            weights[:, 1:] @ basis_directions[:, 1:].T, deflated, rtol=0, atol=1e-10
        )
        # Adding the leading direction back restores the ordinary fit exactly.
        full, _ = fit_coefficients(self.table, self.targets)
        torch.testing.assert_close(
            deflated + weights[:, :1] @ basis_directions[:, :1].T, full, rtol=0, atol=1e-10
        )


class DeflatedRelativeErrorTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(2)
        common = torch.randn(30, generator=generator, dtype=torch.float64)
        self.basis = 20 * common + torch.randn(5, 30, generator=generator, dtype=torch.float64)
        self.coefficients = torch.randn(3, 5, generator=generator, dtype=torch.float64)
        centered = self.basis - self.basis.mean(dim=-1, keepdim=True)
        self.direction = torch.linalg.svd(centered, full_matrices=False).Vh[0]

    def test_exact_combination_has_zero_error_and_no_fit_has_error_one(self):
        targets = self.coefficients @ self.basis
        torch.testing.assert_close(
            deflated_relative_error(self.coefficients, self.basis, targets),
            torch.zeros(3, dtype=torch.float64), rtol=0, atol=1e-18,
        )
        torch.testing.assert_close(
            deflated_relative_error(torch.zeros_like(self.coefficients), self.basis, targets),
            torch.ones(3, dtype=torch.float64),
        )

    def test_ignores_the_leading_direction_and_constant_shifts(self):
        targets = self.coefficients @ self.basis
        shifted = targets + 7.0 * self.direction + 3.0
        self.assertLess(
            deflated_relative_error(self.coefficients, self.basis, shifted).max().item(), 1e-18
        )
        # The same shift is a large error without deflation.
        centered = shifted - shifted.mean(dim=-1, keepdim=True)
        fitted = self.coefficients @ (self.basis - self.basis.mean(dim=-1, keepdim=True))
        self.assertGreater((fitted - centered).square().sum(dim=-1).min().item(), 1.0)


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output_path = Path(temporary.name) / "plots" / "approximation.png"

    def test_constant_shift_is_a_perfect_approximation(self):
        true_logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.5, 0.0]])
        metrics, per_history = summarize_next_token_approximation(
            true_logits + 7.0, true_logits, top_k=2, output_path=self.output_path
        )
        for name in ("centered_logit_rmse", "centered_logit_mae", "kl_true_to_approx",
                     "total_variation"):
            self.assertAlmostEqual(metrics[name], 0.0, places=6)
        self.assertAlmostEqual(metrics["mean_logit_bias"], 7.0, places=6)
        self.assertEqual(metrics["top1_agreement"], 1.0)
        self.assertEqual(metrics["top2_overlap"], 1.0)
        self.assertEqual(per_history["centered_logit_rmse"].shape, (2,))
        self.assertTrue(self.output_path.is_file())

    def test_without_output_path_no_figure_is_written(self):
        true_logits = torch.tensor([[2.0, 0.0, -1.0]])
        metrics, _ = summarize_next_token_approximation(
            true_logits, true_logits, top_k=1, output_path=None
        )
        self.assertAlmostEqual(metrics["kl_true_to_approx"], 0.0, places=6)
        self.assertFalse(self.output_path.parent.exists())

    def test_swapped_two_token_distribution(self):
        log3 = math.log(3)
        # p = (3/4, 1/4) against q = (1/4, 3/4).
        metrics, _ = summarize_next_token_approximation(
            torch.tensor([[0.0, log3]]), torch.tensor([[log3, 0.0]]),
            top_k=10, output_path=self.output_path,
        )
        self.assertAlmostEqual(metrics["centered_logit_rmse"], log3, places=6)
        self.assertAlmostEqual(metrics["kl_true_to_approx"], 0.5 * log3, places=6)
        self.assertAlmostEqual(metrics["total_variation"], 0.5, places=6)
        self.assertEqual(metrics["top1_agreement"], 0.0)
        self.assertEqual(metrics["top2_overlap"], 1.0)
        self.assertEqual(metrics["centered_logit_rmse_std"], 0.0)


class ReportTests(FiniteTableFixture):
    def test_small_run_writes_metrics_and_figure(self):
        output_dir = self.root / "report"
        result = report_finite_table_baseline(
            store_dir=self.store, model_path=self.model_path, output_dir=output_dir,
            num_basis=6, num_futures=4, top_k=3, num_targets=2, batch_size=3,
            seed=1, device="cpu",
        )
        self.assertTrue((output_dir / "next_token_approximation.png").is_file())
        saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["summary"], result)
        self.assertEqual(saved["config"]["table_shape"], [6, 12])
        self.assertEqual(saved["config"]["num_targets"], 2)
        for name in ("centered_logit_rmse", "kl_true_to_approx", "total_variation",
                     "top1_agreement", "table_relative_residual", "coefficient_sum",
                     "coefficient_l1"):
            self.assertTrue(math.isfinite(result[name]), name)
            self.assertTrue(math.isfinite(result[f"{name}_std"]), name)
            self.assertEqual(len(saved["per_target"][name]), 2)
        self.assertGreaterEqual(result["table_relative_residual"], 0.0)
        self.assertEqual(len(saved["per_target"]["table_residual_vs_mean_row"]), 2)

    def test_small_run_reports_reference_approximations(self):
        output_dir = self.root / "report-references"
        result = report_finite_table_baseline(
            store_dir=self.store, model_path=self.model_path, output_dir=output_dir,
            num_basis=6, num_futures=4, top_k=3, num_targets=2, batch_size=3,
            seed=1, device="cpu",
        )
        references = json.loads(
            (output_dir / "metrics.json").read_text(encoding="utf-8")
        )["references"]
        self.assertEqual(set(references), {"uniform_mean", "oracle_full_vocab_fit"})
        for metrics in references.values():
            self.assertTrue(math.isfinite(metrics["kl_true_to_approx"]))
            self.assertTrue(math.isfinite(metrics["centered_logit_rmse"]))
        # The oracle minimizes each target's centered-logit squared error over all
        # coefficient vectors, so no other coefficients can beat it on that target.
        oracle_rmse = references["oracle_full_vocab_fit"]["centered_logit_rmse"]
        self.assertLessEqual(oracle_rmse, result["centered_logit_rmse"] + 1e-9)
        self.assertLessEqual(
            oracle_rmse, references["uniform_mean"]["centered_logit_rmse"] + 1e-9
        )
        self.assertEqual(
            sorted(path.name for path in output_dir.iterdir()),
            ["metrics.json", "next_token_approximation.png"],
        )

    def test_small_run_reports_rank_sweep_and_deflate1_identity(self):
        output_dir = self.root / "report-rank-sweep"
        result = report_finite_table_baseline(
            store_dir=self.store, model_path=self.model_path, output_dir=output_dir,
            num_basis=6, num_futures=4, top_k=3, num_targets=2, batch_size=3,
            seed=1, device="cpu",
        )
        saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        sweep = saved["rank_sweep"]
        # Powers of two below the basis size, then the basis size itself.
        self.assertEqual(list(sweep), ["1", "2", "4", "6"])
        for metrics in sweep.values():
            for name in ("kl_true_to_approx", "centered_logit_rmse", "top1_agreement",
                         "table_residual_vs_mean_row", "coefficient_l1"):
                self.assertTrue(math.isfinite(metrics[name]), name)
        # Using every direction is the ordinary least-squares fit.
        for name in ("kl_true_to_approx", "centered_logit_rmse", "coefficient_l1",
                     "table_residual_vs_mean_row"):
            self.assertAlmostEqual(sweep["6"][name], result[name], places=6)
        # More directions can only lower the in-table residual.
        residuals = [metrics["table_residual_vs_mean_row"] for metrics in sweep.values()]
        self.assertEqual(residuals, sorted(residuals, reverse=True))
        self.assertLess(saved["deflate1"]["restore_max_abs_coefficient_difference"], 1e-8)

    def test_small_run_evaluates_the_deflate1_fit_on_its_own(self):
        output_dir = self.root / "report-deflate1"
        report_finite_table_baseline(
            store_dir=self.store, model_path=self.model_path, output_dir=output_dir,
            num_basis=6, num_futures=4, top_k=3, num_targets=2, batch_size=3,
            seed=1, device="cpu",
        )
        saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        deflate1 = saved["deflate1"]
        for name in ("fit_only", "fit_plus_basis_mean"):
            for metric in ("kl_true_to_approx", "centered_logit_rmse", "top1_agreement"):
                self.assertTrue(math.isfinite(deflate1[name][metric]), (name, metric))
        errors = deflate1["deflated_logit_relative_error"]
        self.assertEqual(set(errors), {"deflate1_fit", "ordinary_fit", "uniform_mean"})
        for value in errors.values():
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
        self.assertEqual(len(saved["per_target"]["deflate1_coefficient_sum"]), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_small_run_completes_on_cuda(self):
        output_dir = self.root / "report-cuda"
        result = report_finite_table_baseline(
            store_dir=self.store, model_path=self.model_path, output_dir=output_dir,
            num_basis=6, num_futures=4, top_k=3, num_targets=2, batch_size=3,
            seed=1, device="cuda",
        )
        self.assertTrue(math.isfinite(result["kl_true_to_approx"]))
        self.assertTrue((output_dir / "metrics.json").is_file())

    def test_rejects_non_positive_counts(self):
        for name in ("num_basis", "num_futures", "top_k", "num_targets", "batch_size"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                report_finite_table_baseline(
                    store_dir=self.store, model_path=self.model_path,
                    output_dir=self.root / "unused", device="cpu",
                    **{"num_basis": 6, "num_futures": 4, "top_k": 3, "num_targets": 2,
                       "batch_size": 3, name: 0},
                )


if __name__ == "__main__":
    unittest.main()
