import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.exp6_stream_gpt2_mlp.features import build_features as build_mlp_features
from experiments.exp7_stream_transformer.features import (
    build_features as build_transformer_features,
)
from experiments.exp8_stream_token_mlp.features import (
    build_features as build_token_mlp_features,
)
from utils.inference import CausalLMInference
from utils.stream_neural_svd import (
    TokenStream,
    centered_logit_block,
    centered_unembedding,
    column_features,
    learned_probe_spectrum,
    probe_reference,
    train_stream_neural_svd,
)


class _TableFeatures(nn.Module):
    """Deterministic feature functions for checking the probe moment formulas."""

    def __init__(self, rows: torch.Tensor, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("rows", rows)
        self.register_buffer("weights", weights)
        self.config = {}

    def row_features(self, ids: torch.Tensor) -> torch.Tensor:
        return self.rows[ids[:, 0]]

    def column_weights(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weights[ids[:, 0]]


class StreamNeuralSVDTests(unittest.TestCase):
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

    def _train(
        self,
        output_dir: Path,
        *,
        rank: int = 2,
        architecture: str = "exp6",
        mlp_width: int = 16,
        **overrides,
    ) -> dict:
        torch.manual_seed(0)
        if architecture == "exp6":
            features = build_mlp_features(
                {
                    "kind": "frozen_gpt2_mlp",
                    "mlp_width": mlp_width,
                    "hidden_size": 8,
                    "rank": rank,
                },
                self.inference,
            )
        elif architecture == "exp7":
            features = build_transformer_features(
                {
                    "kind": "small_transformer",
                    "model_dim": 16,
                    "layers": 1,
                    "heads": 2,
                    "history_tokens": 3,
                    "future_tokens": 2,
                    "vocab_size": 11,
                    "hidden_size": 8,
                    "rank": rank,
                },
                self.inference,
            )
        elif architecture == "exp8":
            features = build_token_mlp_features(
                {
                    "kind": "token_mlp",
                    "embedding_dim": 8,
                    "mlp_width": mlp_width,
                    "history_tokens": 3,
                    "future_tokens": 2,
                    "vocab_size": 11,
                    "hidden_size": 8,
                    "rank": rank,
                },
                self.inference,
            )
        else:
            raise ValueError(f"unknown test architecture: {architecture}")
        kwargs = dict(
            features=features,
            inference=self.inference,
            stream=self.stream,
            stream_path=self.stream_path,
            model_path=self.model_path,
            device="cpu",
            rank=rank,
            nesting="sequential",
            history_tokens=3,
            future_tokens=2,
            history_batch=4,
            future_batch=3,
            probe_histories=8,
            probe_futures=8,
            probe_seed=1,
            max_steps=10,
            eval_every=5,
            tol=1e-3,
            patience=3,
            checkpoint_every=3,
            lr=1e-3,
            lr_schedule="constant",
            min_lr=1e-5,
            lr_decay_steps=20,
            seed=0,
            log_every=100000,
            output_dir=output_dir,
            overwrite=False,
        )
        kwargs.update(overrides)
        return train_stream_neural_svd(**kwargs)

    def _dense_probe(
        self,
        histories: torch.Tensor,
        futures: torch.Tensor,
        *,
        future_batch: int = 1,
    ) -> np.ndarray:
        # Use float64 products, not an already-rounded float32 logit block.
        # Match forward batching so float32 GEMM roundoff cannot swamp the
        # strict reference-SVD tolerance for the smallest singular values.
        token_matrix = self.token_matrix.double()
        columns = []
        for start in range(0, len(futures), future_batch):
            batch = futures[start : start + future_batch]
            pairs = [
                torch.cat((history, future))
                for future in batch
                for history in histories
            ]
            hidden = self.inference.final_hidden_states(torch.stack(pairs))[:, -1].double()
            logits = hidden @ token_matrix.T
            columns.append(
                logits.reshape(len(batch), len(histories), 11)
                .permute(1, 0, 2)
                .reshape(len(histories), -1)
            )
        return torch.cat(columns, dim=1).numpy()

    def _assert_state_equal(self, actual, expected) -> None:
        if isinstance(actual, torch.Tensor):
            self.assertIsInstance(expected, torch.Tensor)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                with self.subTest(state=key):
                    self._assert_state_equal(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for item, other in zip(actual, expected):
                self._assert_state_equal(item, other)
        else:
            self.assertEqual(actual, expected)

    def test_centered_logit_block_matches_every_full_vocabulary_entry(self) -> None:
        histories = self.stream.windows(torch.tensor([0, 3, 6, 9]), 3)
        futures = self.stream.windows(torch.tensor([0, 4, 10]), 2)
        block = centered_logit_block(
            self.inference, self.token_matrix, histories, futures
        )
        self.assertEqual(block.shape, (4, 3 * 11))
        for h, history in enumerate(histories):
            for f, future in enumerate(futures):
                with self.subTest(history=h, future=f):
                    logits = self.inference.next_token_logits(
                        torch.cat((history, future)).tolist()
                    )
                    torch.testing.assert_close(
                        block[h, f * 11 : (f + 1) * 11],
                        logits - logits.mean(),
                        rtol=0,
                        atol=1e-5,
                    )

    def test_final_hidden_states_match_logits_without_gradients(self) -> None:
        ids = self.stream.windows(torch.tensor([0, 4]), 5)
        hidden = self.inference.final_hidden_states(ids)
        self.assertEqual(hidden.shape, (2, 5, 8))
        self.assertEqual(hidden.dtype, torch.float32)
        self.assertFalse(hidden.requires_grad)
        with torch.no_grad():
            expected = self.inference.model(input_ids=ids, use_cache=False).logits
            actual = hidden @ self.inference.model.get_output_embeddings().weight.T
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                torch.testing.assert_close(
                    self.inference.final_hidden_states(ids.to(dtype)), hidden,
                    rtol=0, atol=0,
                )

    def test_final_hidden_states_reject_invalid_inputs(self) -> None:
        invalid = (
            [[1, 2]],
            torch.tensor(1),
            torch.tensor([1, 2]),
            torch.ones((1, 1, 1), dtype=torch.long),
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((2, 0), dtype=torch.long),
            torch.ones((1, 2), dtype=torch.float32),
            torch.ones((1, 2), dtype=torch.bool),
        )
        for ids in invalid:
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(
                    ValueError, "^input_ids must be a non-empty 2-D integer tensor$"
                ):
                    self.inference.final_hidden_states(ids)
        with self.assertRaisesRegex(
            ValueError, "^token_ids length 13 exceeds model maximum 12$"
        ):
            self.inference.final_hidden_states(torch.ones((1, 13), dtype=torch.long))
        for token_id in (-1, 11):
            with self.subTest(token_id=token_id):
                with self.assertRaisesRegex(
                    ValueError, r"^token ids must be in \[0, 11\)$"
                ):
                    self.inference.final_hidden_states(torch.tensor([[1, token_id]]))

    def test_probe_reference_matches_exact_dense_svd(self) -> None:
        histories = self.stream.windows(torch.arange(6), 3)
        futures = self.stream.windows(torch.arange(5), 2)
        for future_batch in (1, 2, 5):
            with self.subTest(future_batch=future_batch):
                dense = self._dense_probe(
                    histories, futures, future_batch=future_batch
                )
                expected_u, expected_s, _ = np.linalg.svd(
                    dense / np.sqrt(6 * 5 * 11), full_matrices=False
                )
                reference = probe_reference(
                    self.inference, self.token_matrix, histories, futures,
                    future_batch=future_batch,
                )
                self.assertEqual(reference.singular_values.shape, (6,))
                self.assertEqual(reference.left_singular_vectors.shape, (6, 6))
                self.assertEqual(reference.singular_values.dtype, np.float64)
                self.assertEqual(reference.left_singular_vectors.dtype, np.float64)
                np.testing.assert_allclose(
                    reference.singular_values, expected_s, rtol=1e-8, atol=0
                )
                alignment = np.abs(
                    np.sum(reference.left_singular_vectors[:, :2] * expected_u[:, :2], axis=0)
                )
                self.assertTrue(np.all(alignment > 1 - 1e-8), alignment)
                np.testing.assert_allclose(reference.energy, np.mean(dense**2), rtol=1e-12)

    def test_learned_probe_spectrum_matches_dense_feature_moments(self) -> None:
        generator = torch.Generator().manual_seed(7)
        rows = torch.randn(7, 3, generator=generator, dtype=torch.float64)
        weights = torch.randn(5, 3, 8, generator=generator, dtype=torch.float64)
        token_matrix = self.token_matrix.double()
        features = _TableFeatures(rows, weights)
        histories, futures = torch.arange(7)[:, None], torch.arange(5)[:, None]
        dense_columns = column_features(weights, token_matrix)
        expected_columns = torch.stack(
            [token_matrix @ future_weights.T for future_weights in weights]
        ).reshape(5 * 11, 3)
        torch.testing.assert_close(dense_columns, expected_columns, rtol=1e-14, atol=1e-14)
        f_rms = rows.square().mean(dim=0).sqrt()
        g_rms = dense_columns.square().mean(dim=0).sqrt()
        normalized_f = rows / (f_rms * np.sqrt(len(rows)))
        normalized_g = dense_columns / (g_rms * np.sqrt(len(dense_columns)))
        off_diagonal = ~torch.eye(3, dtype=torch.bool)
        expected_left_error = (normalized_f.T @ normalized_f)[off_diagonal].abs().max().item()
        expected_right_error = (normalized_g.T @ normalized_g)[off_diagonal].abs().max().item()
        for chunk in (1, 2, 256):
            with self.subTest(chunk=chunk):
                spectrum = learned_probe_spectrum(
                    features, histories, futures, token_matrix.T @ token_matrix,
                    vocab_size=11, chunk=chunk,
                )
                self.assertEqual(spectrum.singular_values.dtype, np.float64)
                self.assertEqual(spectrum.left_singular_vectors.dtype, np.float64)
                np.testing.assert_allclose(spectrum.singular_values, (f_rms * g_rms).numpy(), rtol=1e-12)
                np.testing.assert_allclose(spectrum.left_singular_vectors, normalized_f.numpy(), rtol=1e-12)
                self.assertAlmostEqual(spectrum.left_orthogonality_error, expected_left_error, places=12)
                self.assertAlmostEqual(spectrum.right_orthogonality_error, expected_right_error, places=12)

    def test_learned_probe_spectrum_rejects_zero_feature_columns(self) -> None:
        ids = torch.arange(3)[:, None]
        gram = self.token_matrix.double().T @ self.token_matrix.double()
        for zero_side in ("row", "column"):
            with self.subTest(zero_side=zero_side):
                rows = torch.ones(3, 2, dtype=torch.float64)
                weights = torch.ones(3, 2, 8, dtype=torch.float64)
                if zero_side == "row":
                    rows[:, 0] = 0
                else:
                    weights[:, 0] = 0
                with self.assertRaisesRegex(
                    ValueError, "^every feature column must have nonzero norm$"
                ):
                    learned_probe_spectrum(
                        _TableFeatures(rows, weights), ids, ids, gram,
                        vocab_size=11, chunk=2,
                    )

    def test_interrupted_training_resumes_identically(self) -> None:
        for schedule in ("constant", "cosine"):
            with self.subTest(lr_schedule=schedule):
                uninterrupted_dir = self.root / "resume" / schedule / "uninterrupted"
                resumed_dir = self.root / "resume" / schedule / "resumed"
                uninterrupted_result = self._train(
                    uninterrupted_dir, max_steps=10, lr_schedule=schedule
                )
                interrupted_result = self._train(
                    resumed_dir, max_steps=4, lr_schedule=schedule
                )
                self.assertEqual(interrupted_result["steps"], 4)
                resumed_result = self._train(
                    resumed_dir, max_steps=10, lr_schedule=schedule
                )
                for output_dir, result in (
                    (uninterrupted_dir, uninterrupted_result),
                    (resumed_dir, resumed_result),
                ):
                    self.assertEqual(result["stop_reason"], "max_steps")
                    self.assertEqual(result["steps"], 10)
                    self.assertEqual(result["pairs_evaluated"], 10 * 4 * 3)
                    metadata = json.loads((output_dir / "metadata.json").read_text())
                    self.assertEqual(metadata["steps"], 10)
                    self.assertIs(metadata["converged"], False)
                    self.assertIsNotNone(metadata["probe_singular_values"])
                uninterrupted = torch.load(
                    uninterrupted_dir / "checkpoint.pt", map_location="cpu", weights_only=True
                )
                resumed = torch.load(
                    resumed_dir / "checkpoint.pt", map_location="cpu", weights_only=True
                )
                # Covers feature tensors, sampling generator, loss history,
                # convergence state and every Adam moment/step/parameter group.
                self._assert_state_equal(uninterrupted, resumed)
                self.assertGreater(len(uninterrupted["optimizer"]["state"]), 0)
                for state in uninterrupted["optimizer"]["state"].values():
                    self.assertIn("exp_avg", state)
                    self.assertIn("exp_avg_sq", state)
                    self.assertEqual(state["step"].item(), 10)
                snapshots = []
                for output_dir, checkpoint in (
                    (uninterrupted_dir, uninterrupted), (resumed_dir, resumed)
                ):
                    snapshot = torch.load(
                        output_dir / "features.pt", map_location="cpu", weights_only=True
                    )
                    self._assert_state_equal(snapshot, checkpoint["features"])
                    snapshots.append(snapshot)
                self._assert_state_equal(*snapshots)
                np.testing.assert_array_equal(
                    np.load(uninterrupted_dir / "loss_history.npy"),
                    np.load(resumed_dir / "loss_history.npy"),
                )
                with np.load(uninterrupted_dir / "probe.npz") as left, np.load(
                    resumed_dir / "probe.npz"
                ) as right:
                    self.assertEqual(set(left.files), {"history_starts", "future_starts"})
                    for name in left.files:
                        np.testing.assert_array_equal(left[name], right[name])
                        self.assertEqual(left[name].dtype, np.int64)

    def test_resume_rejects_changed_rank(self) -> None:
        output_dir = self.root / "rank_mismatch"
        self._train(output_dir, max_steps=4)
        with self.assertRaisesRegex(ValueError, "checkpoint config mismatch.*rank"):
            self._train(output_dir, rank=3, max_steps=10)

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW") == "1",
        "convergence training is slow; set RUN_SLOW=1 to run it",
    )
    def test_both_architectures_recover_all_starts_operator_top_two(self) -> None:
        histories = self.stream.windows(torch.arange(10), 3)
        futures = self.stream.windows(torch.arange(11), 2)
        dense = self._dense_probe(histories, futures)
        expected = np.linalg.svd(
            dense / np.sqrt(10 * 11 * 11), compute_uv=False
        )[:2]
        gram = self.token_matrix.double().T @ self.token_matrix.double()
        for architecture in ("exp6", "exp7", "exp8"):
            with self.subTest(architecture=architecture):
                output_dir = self.root / "convergence" / architecture
                result = self._train(
                    output_dir,
                    architecture=architecture,
                    mlp_width=64,
                    history_batch=8,
                    future_batch=8,
                    lr=3e-3,
                    lr_schedule="cosine",
                    lr_decay_steps=30000,
                    min_lr=1e-5,
                    max_steps=None,
                    eval_every=1000,
                    checkpoint_every=1000,
                )
                self.assertEqual(result["stop_reason"], "lr_schedule_end")
                self.assertEqual(result["steps"], 30000)
                self.assertEqual(result["pairs_evaluated"], 30000 * 8 * 8)
                metadata = json.loads((output_dir / "metadata.json").read_text())
                builder = {
                    "exp6": build_mlp_features,
                    "exp7": build_transformer_features,
                    "exp8": build_token_mlp_features,
                }[architecture]
                features = builder(metadata["features"], self.inference)
                features.load_state_dict(
                    torch.load(output_dir / "features.pt", map_location="cpu", weights_only=True)
                )
                spectrum = learned_probe_spectrum(
                    features, histories, futures, gram, vocab_size=11,
                )
                np.testing.assert_allclose(
                    spectrum.singular_values, expected, rtol=2e-2, atol=0
                )


if __name__ == "__main__":
    unittest.main()
