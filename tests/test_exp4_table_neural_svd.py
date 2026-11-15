import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.exp1_L_M_table.table import build_matrix
from experiments.exp4_table_neural_svd.tables import build_exp4_tables
from experiments.exp4_table_neural_svd.train import (
    Table,
    compare_with_exact_svd,
    load_table,
    sample_subtable,
    table_columns,
    table_features,
    train_exp4,
)
from utils.common import Sample
from utils.inference import CausalLMInference
from utils.nestedlora import NestedLoRA

VOCAB = 17
HISTORIES = [[row, (row + 1) % VOCAB] for row in range(6)]
FUTURES = [[(row + 2) % VOCAB, (row + 5) % VOCAB, row + 9] for row in range(5)]
TOP_K = 3


class _LookupSingularNetwork(nn.Module):
    """One free feature vector per row id and per column id."""

    def __init__(self, rows, columns, rank):
        super().__init__()
        self.left = nn.Embedding(rows, rank)
        self.right = nn.Embedding(columns, rank)
        nn.init.normal_(self.left.weight, std=0.1)
        nn.init.normal_(self.right.weight, std=0.1)

    def forward(self, history, column):
        return self.left(history[:, 0]), self.right(column[:, -1])


class _BagSingularNetwork(nn.Module):
    """Mean token embedding per side, so unseen histories and futures get features."""

    def __init__(self, rank):
        super().__init__()
        self.left = nn.Sequential(nn.Embedding(VOCAB, 8), nn.Linear(8, rank))
        self.right = nn.Sequential(nn.Embedding(VOCAB, 8), nn.Linear(8, rank))

    def forward(self, history, column):
        return (
            self.left[1](self.left[0](history).mean(dim=1)),
            self.right[1](self.right[0](column).mean(dim=1)),
        )


def _lookup_table(matrix: torch.Tensor) -> Table:
    """A table whose rows and columns are identified by their index."""
    rows, columns = matrix.shape
    U, S, Vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    return Table(
        histories=torch.arange(rows).unsqueeze(1),
        columns=torch.stack((torch.zeros(columns, dtype=torch.int64), torch.arange(columns)), dim=1),
        matrix=matrix.float(),
        singular_values=S,
        left_vectors=U,
        right_vectors=Vh.T,
    )


def _low_rank(rows, columns, singular_values) -> torch.Tensor:
    rank = len(singular_values)
    left = torch.linalg.qr(torch.randn(rows, rank, dtype=torch.float64)).Q
    right = torch.linalg.qr(torch.randn(columns, rank, dtype=torch.float64)).Q
    return (left * torch.tensor(singular_values, dtype=torch.float64)) @ right.T


class Exp4Fixture(unittest.TestCase):
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
        cls.samples_path = cls.root / "samples.parquet"
        windows = HISTORIES + FUTURES
        pq.write_table(pa.table({
            "sample_id": pa.array(range(len(windows)), type=pa.int32()),
            "group": ["history"] * len(HISTORIES) + ["future"] * len(FUTURES),
            "token_ids": pa.array(windows, type=pa.list_(pa.int32())),
        }), cls.samples_path)
        cls.inference_dir = cls.root / "inference"
        cls.summary = cls.build(cls.inference_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    @classmethod
    def build(cls, output_dir: Path, **changes) -> dict:
        options = dict(
            samples_path=cls.samples_path, model_path=cls.model_path, device="cpu",
            train_histories=4, train_futures=3, val_histories=2, val_futures=2,
            top_k=TOP_K, batch_size=2, output_dir=output_dir, overwrite=False,
        )
        options.update(changes)
        return build_exp4_tables(**options)


class BuildTablesTests(Exp4Fixture):
    def test_tables_take_disjoint_samples_and_match_the_exp1_matrix(self):
        self.assertEqual(self.summary["tables"]["train"]["matrix_shape"], [4, 3 * TOP_K])
        self.assertEqual(self.summary["tables"]["val"]["matrix_shape"], [2, 2 * TOP_K])
        inference = CausalLMInference(self.model_path, "cpu")
        expected_tokens = {
            "train": (HISTORIES[:4], FUTURES[:3]), "val": (HISTORIES[4:], FUTURES[3:]),
        }
        for name, (histories, futures) in expected_tokens.items():
            with self.subTest(table=name):
                tokens = np.load(self.inference_dir / name / "tokens.npz")
                self.assertEqual(tokens["histories"].tolist(), histories)
                self.assertEqual(tokens["futures"].tolist(), futures)
                matrix = np.load(self.inference_dir / name / "matrix.npy")
                self.assertEqual(matrix.dtype, np.float32)
                expected = build_matrix(
                    inference,
                    [Sample(index, row) for index, row in enumerate(histories)],
                    [Sample(index, row) for index, row in enumerate(futures)],
                    tokens["header_token_ids"],
                )
                np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-5)

    def test_saved_svd_reconstructs_each_matrix(self):
        for name in ("train", "val"):
            with self.subTest(table=name):
                matrix = np.load(self.inference_dir / name / "matrix.npy")
                svd = np.load(self.inference_dir / name / "svd.npz")
                self.assertEqual(svd["S"].shape, (min(matrix.shape),))
                self.assertTrue(np.all(np.diff(svd["S"]) <= 0))
                np.testing.assert_allclose(
                    (svd["U"] * svd["S"]) @ svd["Vt"], matrix, rtol=0, atol=1e-5
                )

    def test_existing_outputs_need_overwrite(self):
        with self.assertRaisesRegex(FileExistsError, "--overwrite"):
            self.build(self.inference_dir)
        self.build(self.inference_dir, overwrite=True)

    def test_too_few_samples_fail(self):
        with self.assertRaisesRegex(ValueError, "history"):
            self.build(self.root / "too-many", val_histories=3)

    def test_loaded_table_columns_follow_the_matrix_order(self):
        table = load_table(self.inference_dir / "train", "cpu")
        tokens = np.load(self.inference_dir / "train" / "tokens.npz")
        self.assertEqual(table.columns.shape, (3 * TOP_K, 4))
        self.assertEqual(table.right_vectors.shape[0], 3 * TOP_K)
        for column in range(3 * TOP_K):
            future, rank = divmod(column, TOP_K)
            self.assertEqual(table.columns[column, :-1].tolist(), FUTURES[future])
            self.assertEqual(
                table.columns[column, -1].item(), tokens["header_token_ids"][future, rank]
            )


class SubtableTests(unittest.TestCase):
    def test_columns_are_future_major(self):
        futures = torch.tensor([[1, 2], [3, 4]])
        headers = torch.tensor([[7, 8, 9], [5, 6, 7]])
        self.assertEqual(table_columns(futures, headers).tolist(), [
            [1, 2, 7], [1, 2, 8], [1, 2, 9], [3, 4, 5], [3, 4, 6], [3, 4, 7],
        ])

    def test_kernel_holds_the_matrix_entry_of_every_sampled_row_and_column(self):
        table = _lookup_table(torch.arange(6 * 10, dtype=torch.float64).reshape(6, 10))
        generator = torch.Generator().manual_seed(3)
        seen = set()
        for _ in range(5):
            h, columns, kernel = sample_subtable(table, 4, 7, generator)
            self.assertEqual(kernel.shape, (4, 7))
            rows, column_ids = h[:, 0], columns[:, -1]
            self.assertEqual(len(set(rows.tolist())), 4)
            self.assertEqual(len(set(column_ids.tolist())), 7)
            torch.testing.assert_close(kernel, table.matrix[rows][:, column_ids])
            seen.add(tuple(rows.tolist()))
        self.assertGreater(len(seen), 1)


class CompareWithExactSVDTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(1)
        self.table = _lookup_table(_low_rank(12, 20, [9.0, 5.0, 3.0, 1.0]))

    def test_exact_factors_score_perfectly(self):
        scale = torch.tensor([2.0, 0.5, 4.0])
        left = self.table.left_vectors[:, :3] * self.table.singular_values[:3] / scale
        right = self.table.right_vectors[:, :3] * scale
        result = compare_with_exact_svd(left, right, self.table, ranks=(1, 2, 3, 8))
        self.assertEqual(result["ranks"], [1, 2, 3])
        zeros, ones = torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
        as_tensor = lambda name: torch.tensor(result[name], dtype=torch.float64)
        torch.testing.assert_close(as_tensor("singular_value_relative_error"), zeros, atol=1e-9, rtol=0)
        torch.testing.assert_close(as_tensor("left_subspace_overlap"), ones, atol=1e-9, rtol=0)
        torch.testing.assert_close(as_tensor("right_subspace_overlap"), ones, atol=1e-9, rtol=0)
        torch.testing.assert_close(
            as_tensor("captured_energy"), as_tensor("exact_captured_energy"), atol=1e-6, rtol=0
        )
        energy = self.table.singular_values.square()
        torch.testing.assert_close(
            as_tensor("exact_captured_energy"), energy.cumsum(0)[:3] / energy.sum(),
            atol=1e-6, rtol=0,
        )

    def test_captured_energy_matches_the_direct_reconstruction(self):
        left = torch.randn(12, 3, dtype=torch.float64)
        right = torch.randn(20, 3, dtype=torch.float64)
        result = compare_with_exact_svd(left, right, self.table, ranks=(1, 2, 3), chunk=5)
        matrix = self.table.matrix.double()
        for index, rank in enumerate((1, 2, 3)):
            residual = (matrix - left[:, :rank] @ right[:, :rank].T).square().sum()
            self.assertAlmostEqual(
                result["captured_energy"][index],
                (1 - residual / matrix.square().sum()).item(), places=9,
            )
            self.assertLess(result["left_subspace_overlap"][index], 0.9)
            self.assertLess(result["right_subspace_overlap"][index], 0.9)


class TrainingTests(Exp4Fixture):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def train(self, output_dir: Path, **changes) -> dict:
        options = dict(
            inference_dir=self.inference_dir, output_dir=output_dir, device="cpu", neigs=2,
            history_batch=3, column_batch=5, steps=4, eval_every=2, lr=0.01, seed=5,
            overwrite=False, model=NestedLoRA(_BagSingularNetwork(2), neigs=2),
        )
        options.update(changes)
        return train_exp4(**options)

    def test_subtable_training_recovers_the_leading_exact_modes(self):
        torch.manual_seed(2)
        table = _lookup_table(_low_rank(12, 20, [9.0, 5.0, 0.5]))
        directory = self.root / "synthetic"
        for name in ("train", "val"):
            (directory / name).mkdir(parents=True)
            np.save(directory / name / "matrix.npy", table.matrix.numpy())
            np.savez(
                directory / name / "tokens.npz",
                histories=table.histories.numpy(),
                # One zero future token and one header id per column.
                futures=np.zeros((20, 1), dtype=np.int64),
                header_token_ids=np.arange(20, dtype=np.int32)[:, None],
            )
            np.savez(
                directory / name / "svd.npz", S=table.singular_values.numpy(),
                U=table.left_vectors.numpy(), Vt=table.right_vectors.T.numpy(),
            )
        (directory / "metadata.json").write_text("{}", encoding="utf-8")
        summary = self.train(
            self.root / "synthetic-train", inference_dir=directory, history_batch=8,
            column_batch=12, steps=1500, eval_every=1500, lr=0.02,
            model=NestedLoRA(_LookupSingularNetwork(12, 20, 2), neigs=2),
        )
        self.assertEqual(summary["step"], 1500)
        record = json.loads(
            (self.root / "synthetic-train" / "eval_history.jsonl").read_text().splitlines()[-1]
        )
        for name in ("train", "val"):
            with self.subTest(table=name):
                result = record[name]
                self.assertEqual(result["ranks"], [1, 2])
                for overlap in result["left_subspace_overlap"] + result["right_subspace_overlap"]:
                    self.assertGreater(overlap, 0.98)
                for error in result["singular_value_relative_error"]:
                    self.assertLess(abs(error), 0.05)
                for learned, exact in zip(result["captured_energy"], result["exact_captured_energy"]):
                    self.assertLessEqual(learned, exact + 1e-9)
                    self.assertGreater(learned, exact - 0.01)

    def test_evaluates_both_tables_and_resumes(self):
        output_dir = self.root / "train-run"
        torch.manual_seed(0)
        summary = self.train(output_dir)
        self.assertEqual(summary, {"output_dir": str(output_dir), "step": 4, "evaluations": 2})
        records = [
            json.loads(line) for line in (output_dir / "eval_history.jsonl").read_text().splitlines()
        ]
        self.assertEqual([record["step"] for record in records], [2, 4])
        for record in records:
            self.assertTrue(np.isfinite(record["train_loss"]))
            self.assertGreater(record["kernel_energy"], 0)
            for name in ("train", "val"):
                self.assertEqual(record[name]["ranks"], [1, 2])
                self.assertEqual(len(record[name]["singular_values"]), 2)
                for key in ("left_subspace_overlap", "right_subspace_overlap"):
                    for value in record[name][key]:
                        self.assertTrue(0 <= value <= 1 + 1e-9)

        # A longer budget continues the same run; an uninterrupted run matches it.
        resumed = self.train(output_dir, steps=6)
        self.assertEqual(resumed["step"], 6)
        self.assertEqual(resumed["evaluations"], 3)
        torch.manual_seed(0)
        straight_dir = self.root / "train-straight"
        self.train(straight_dir, steps=6)
        last = lambda path: json.loads((path / "eval_history.jsonl").read_text().splitlines()[-1])
        self.assertEqual(last(output_dir)["step"], 6)
        np.testing.assert_allclose(
            last(output_dir)["train"]["singular_values"],
            last(straight_dir)["train"]["singular_values"], rtol=1e-5,
        )

        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.train(output_dir, lr=0.02)
        restarted = self.train(output_dir, lr=0.02, overwrite=True)
        self.assertEqual(restarted["evaluations"], 2)

    def test_column_mean_centering_removes_each_column_mean_and_refits_the_svd(self):
        raw = load_table(self.inference_dir / "train", "cpu")
        centered = load_table(self.inference_dir / "train", "cpu", "column-mean")
        torch.testing.assert_close(
            centered.matrix, raw.matrix - raw.matrix.mean(dim=0), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            centered.matrix.mean(dim=0), torch.zeros(3 * TOP_K), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(centered.histories, raw.histories)
        torch.testing.assert_close(centered.columns, raw.columns)
        # Centering removes one direction, so the last singular value vanishes.
        self.assertEqual(centered.singular_values.shape, (4,))
        self.assertLess(centered.singular_values[-1].item(), 1e-4)
        self.assertLess(centered.singular_values[0].item(), raw.singular_values[0].item())
        torch.testing.assert_close(
            (centered.left_vectors * centered.singular_values) @ centered.right_vectors.T,
            centered.matrix.double(), rtol=0, atol=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "centering"):
            load_table(self.inference_dir / "train", "cpu", "row")

    def test_centered_training_compares_with_the_centered_svd(self):
        output_dir = self.root / "train-centered"
        torch.manual_seed(0)
        self.train(output_dir, centering="column-mean")
        metadata = json.loads((output_dir / "metadata.json").read_text())
        self.assertEqual(metadata["centering"], "column-mean")
        record = json.loads((output_dir / "eval_history.jsonl").read_text().splitlines()[-1])
        centered = load_table(self.inference_dir / "train", "cpu", "column-mean")
        energy = centered.singular_values.square()
        np.testing.assert_allclose(
            record["train"]["exact_captured_energy"],
            (energy.cumsum(0)[:2] / energy.sum()).tolist(), rtol=1e-5,
        )
        # The centering is part of the run's settings.
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.train(output_dir)

    def test_rejects_missing_tables_and_oversized_batches(self):
        with self.assertRaisesRegex(ValueError, "exp4 inference"):
            self.train(self.root / "nowhere-train", inference_dir=self.root / "nowhere")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.train(self.root / "big-batch", history_batch=5)

    def test_features_cover_every_row_and_column(self):
        table = load_table(self.inference_dir / "val", "cpu")
        model = NestedLoRA(_BagSingularNetwork(2), neigs=2)
        for chunk in (1, 4, 64):
            with self.subTest(chunk=chunk):
                left, right = table_features(model, table, chunk)
                expected_left, expected_right = model(table.histories, table.columns[:2])
                self.assertEqual(left.shape, (2, 2))
                self.assertEqual(right.shape, (2 * TOP_K, 2))
                torch.testing.assert_close(left, expected_left.double())
                torch.testing.assert_close(right[:2], expected_right.double())


if __name__ == "__main__":
    unittest.main()
