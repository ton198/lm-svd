import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_pipline.sample_store.store import STORE_FORMAT, sample_dtype, slot_path
from experiments.svd_inference.train import load_data, train_svd_network
from utils.nestedlora import NestedLoRA


class _LookupSingularNetwork(nn.Module):
    def __init__(self, rows, columns, rank):
        super().__init__()
        self.left = nn.Embedding(rows, rank)
        self.right = nn.Embedding(columns, rank)
        nn.init.normal_(self.left.weight, std=0.1)
        nn.init.normal_(self.right.weight, std=0.1)

    def forward(self, history, future):
        return self.left(history[:, 0]), self.right(future[:, -1])


class _MLPSingularNetwork(nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        self.left = nn.Sequential(
            nn.Linear(size, 64), nn.Tanh(), nn.Linear(64, rank)
        )
        self.right = nn.Sequential(
            nn.Linear(size, 64), nn.Tanh(), nn.Linear(64, rank)
        )

    def forward(self, history, future):
        return self.left(history), self.right(future)


class SVDInferenceTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(7)

    def test_training_recovers_leading_matrix_svd(self):
        left_basis = torch.linalg.qr(torch.randn(4, 3)).Q
        right_basis = torch.linalg.qr(torch.randn(6, 3)).Q
        matrix = (left_basis * torch.tensor([6.0, 3.0, 1.0])) @ right_basis.T
        expected_left, expected_values, expected_right_t = torch.linalg.svd(
            matrix, full_matrices=False
        )
        histories = torch.arange(4).unsqueeze(1)
        columns = torch.stack((torch.zeros(6, dtype=torch.int64), torch.arange(6)), dim=1)
        dataloader = [(histories, columns.unsqueeze(0).expand(4, -1, -1), matrix)]

        for sequential in (False, True):
            with self.subTest(sequential=sequential):
                model = NestedLoRA(
                    _LookupSingularNetwork(rows=4, columns=6, rank=2),
                    neigs=2,
                    sequential=sequential,
                )
                with torch.no_grad():
                    initial_left, initial_right = model(histories, columns)
                    initial_error = torch.linalg.norm(
                        matrix - initial_left @ initial_right.T
                    )
                self.assertGreater(initial_error.item(), 0.5 * matrix.norm().item())
                optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda step: 1.0
                )
                trained = train_svd_network(
                    model, "cpu", dataloader, optimizer, scheduler, epochs=500
                )
                self.assertIs(trained, model)
                self.assertEqual(scheduler.last_epoch, 500)
                print("Training completed successfully.")
                model.eval()
                with torch.no_grad():
                    left, right = model(histories, columns)
                left_norms, right_norms = left.norm(dim=0), right.norm(dim=0)
                torch.testing.assert_close(
                    left_norms * right_norms,
                    expected_values[:2],
                    rtol=1e-3,
                    atol=1e-4,
                )
                for learned, expected in (
                    (left / left_norms, expected_left[:, :2]),
                    (right / right_norms, expected_right_t[:2].T),
                ):
                    alignment = (learned * expected).sum(dim=0).abs()
                    torch.testing.assert_close(
                        alignment, torch.ones(2), rtol=0, atol=1e-3
                    )
                for rank in (1, 2):
                    with self.subTest(rank=rank):
                        expected = (
                            expected_left[:, :rank] * expected_values[:rank]
                        ) @ expected_right_t[:rank]
                        torch.testing.assert_close(
                            left[:, :rank] @ right[:, :rank].T,
                            expected,
                            rtol=1e-3,
                            atol=1e-3,
                        )
                residual = torch.linalg.norm(matrix - left @ right.T)
                self.assertAlmostEqual(residual.item(), expected_values[2].item(), places=3)

    def test_masks_follow_model_device_without_changing_checkpoints(self):
        for sequential in (False, True):
            with self.subTest(sequential=sequential):
                model = NestedLoRA(nn.Linear(2, 3), neigs=3, sequential=sequential)
                checkpoint = {name: value.clone() for name, value in model.state_dict().items()}
                self.assertEqual(set(checkpoint), {'model.weight', 'model.bias'})
                model.to('meta')
                self.assertEqual(model.vector_mask.device.type, 'meta')
                self.assertEqual(model.matrix_mask.device.type, 'meta')
                restored = NestedLoRA(nn.Linear(2, 3), neigs=3, sequential=sequential)
                restored.load_state_dict(checkpoint, strict=True)

    def test_mlp_recovers_svd_of_50_by_50_matrix(self):
        """Learn all five nonzero modes of a dense 50x50 rank-five matrix."""
        size, rank = 50, 5
        left_basis = torch.linalg.qr(torch.randn(size, rank)).Q
        right_basis = torch.linalg.qr(torch.randn(size, rank)).Q
        matrix = (
            left_basis * torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])
        ) @ right_basis.T
        expected_left, expected_values, expected_right_t = torch.linalg.svd(
            matrix, full_matrices=False
        )
        inputs = torch.eye(size)
        model = NestedLoRA(_MLPSingularNetwork(size, rank), neigs=rank)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        with torch.no_grad():
            left, right = model(inputs, inputs)
            initial_relative_error = (left @ right.T - matrix).norm() / matrix.norm()
        self.assertGreater(initial_relative_error.item(), 0.5)

        for _ in range(600):
            optimizer.zero_grad()
            left, right = model(inputs, inputs)
            loss, _ = model.compute_loss_kernel(left, right, matrix)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            left, right = model(inputs, inputs)
        left_norms, right_norms = left.norm(dim=0), right.norm(dim=0)
        learned_values = left_norms * right_norms
        left_alignment = ((left / left_norms) * expected_left[:, :rank]).sum(0).abs()
        right_alignment = ((right / right_norms) * expected_right_t[:rank].T).sum(0).abs()
        relative_error = (left @ right.T - matrix).norm() / matrix.norm()

        torch.testing.assert_close(
            learned_values, expected_values[:rank], rtol=1e-3, atol=1e-4
        )
        for alignment in (left_alignment, right_alignment):
            torch.testing.assert_close(
                alignment, torch.ones(rank), rtol=0, atol=1e-3
            )
        self.assertLess(relative_error.item(), 1e-3)
        self.assertLess(relative_error.item(), initial_relative_error.item() * 1e-3)
        for prefix_rank in range(1, rank + 1):
            with self.subTest(prefix_rank=prefix_rank):
                expected = (
                    expected_left[:, :prefix_rank] * expected_values[:prefix_rank]
                ) @ expected_right_t[:prefix_rank]
                torch.testing.assert_close(
                    left[:, :prefix_rank] @ right[:, :prefix_rank].T,
                    expected,
                    rtol=1e-3,
                    atol=1e-3,
                )

        print("\n50x50 matrix, rank 5; MLP 50 -> 64 -> 5, Adam, 600 steps")
        print("mode  torch.linalg.svd  NestedLoRA  |cos(U)|  |cos(V)|")
        for mode in range(rank):
            print(
                f"{mode + 1:4d}  {expected_values[mode].item():16.6f}"
                f"  {learned_values[mode].item():10.6f}"
                f"  {left_alignment[mode].item():8.6f}"
                f"  {right_alignment[mode].item():8.6f}"
            )
        print(f"Relative reconstruction error: {relative_error.item():.3e}")


class SVDInferenceDataLoaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Path(temporary.name)
        (self.store / "slots").mkdir()
        config = {
            "format": STORE_FORMAT,
            "history_tokens": 2,
            "future_tokens": 3,
            "vocab_size": 17,
            "samples_per_block": 2,
            "num_slots": 2,
        }
        (self.store / "store.json").write_text(json.dumps(config), encoding="utf-8")
        (self.store / "state.json").write_text(json.dumps({
            "workers": {}, "published": {"0": 0, "1": 1}, "next_commit": 2,
        }), encoding="utf-8")
        self.records = np.zeros(4, dtype=sample_dtype(2, 3, 17))
        for row in range(4):
            self.records["h"][row] = [row, row + 4]
            self.records["f"][row] = [row + 5, row + 6, row + 7]
            self.records["logits"][row] = row * 10 - 5 + np.arange(17) * 0.25
        for slot in range(2):
            np.save(slot_path(self.store, slot), self.records[slot * 2:(slot + 1) * 2])

    def test_batches_preserve_shapes_dtypes_and_kernel_column_values(self):
        for batch_size, tokens_per_pair in ((1, 1), (3, 4), (2, 32)):
            with self.subTest(batch_size=batch_size, tokens_per_pair=tokens_per_pair):
                dataloader = load_data(
                    self.store, batch_size, batches_per_epoch=3,
                    seed=7, tokens_per_pair=tokens_per_pair,
                )
                self.assertEqual(len(dataloader), 3)
                batches = list(dataloader)
                self.assertEqual(len(batches), 3)
                for histories, futures, scores in batches:
                    self.assertEqual(histories.shape, (batch_size, 2))
                    self.assertEqual(futures.shape, (batch_size, tokens_per_pair, 4))
                    self.assertEqual(scores.shape, (batch_size, tokens_per_pair))
                    self.assertEqual(histories.dtype, torch.int64)
                    self.assertEqual(futures.dtype, torch.int64)
                    self.assertEqual(scores.dtype, torch.float32)
                    for tensor in (histories, futures, scores):
                        self.assertEqual(tensor.device.type, "cpu")
                        self.assertFalse(tensor.requires_grad)
                    for history, future, kernel_values in zip(histories, futures, scores):
                        record = self.records[int(history[0])]
                        torch.testing.assert_close(
                            history, torch.from_numpy(record["h"].copy())
                        )
                        torch.testing.assert_close(
                            future[:, :-1],
                            torch.from_numpy(record["f"].copy()).expand(tokens_per_pair, -1),
                        )
                        token_ids = future[:, -1]
                        self.assertTrue(((0 <= token_ids) & (token_ids < 17)).all().item())
                        torch.testing.assert_close(
                            kernel_values,
                            torch.from_numpy(record["logits"].copy())[token_ids],
                            rtol=0,
                            atol=0,
                        )

    def test_seed_reproduces_sampling_without_resetting_each_epoch(self):
        first = load_data(self.store, 2, 2, seed=11, tokens_per_pair=5)
        second = load_data(self.store, 2, 2, seed=11, tokens_per_pair=5)
        previous_columns = None
        for _ in range(2):
            first_batches, second_batches = list(first), list(second)
            for first_batch, second_batch in zip(first_batches, second_batches):
                for actual, expected in zip(first_batch, second_batch):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            columns = torch.cat([future[..., -1] for _, future, _ in first_batches])
            if previous_columns is not None:
                self.assertFalse(torch.equal(columns, previous_columns))
            previous_columns = columns

    def test_invalid_sampling_sizes_and_seed_fail(self):
        for changes in (
            {"batch_size": 0}, {"batch_size": -1},
            {"batches_per_epoch": 0}, {"batches_per_epoch": -1},
            {"tokens_per_pair": 0}, {"tokens_per_pair": -1}, {"seed": -1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                options = dict(batch_size=2, batches_per_epoch=2, tokens_per_pair=3)
                options.update(changes)
                load_data(self.store, **options)

if __name__ == "__main__":
    unittest.main()
