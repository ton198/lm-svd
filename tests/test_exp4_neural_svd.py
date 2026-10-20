import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.exp4_neural_svd.spectrum import analyze_exp4
from experiments.exp4_neural_svd.train import centered_logit_block, train_exp4
from utils.common import load_samples
from utils.inference import CausalLMInference


class Exp4NeuralSVDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        previous_rng_state = torch.get_rng_state()
        cls.addClassCleanup(torch.set_rng_state, previous_rng_state)
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
        cls.samples_path = cls.root / "samples.parquet"
        schema = pa.schema(
            [
                ("sample_id", pa.int32()),
                ("group", pa.string()),
                ("token_count", pa.int32()),
                ("start_token", pa.int64()),
                ("token_ids", pa.list_(pa.int32())),
                ("text", pa.string()),
            ],
            metadata={b"lm_svd": json.dumps({"fixture": True}).encode("utf-8")},
        )
        token_ids = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 0, 1],
            [7, 8],
            [9, 10],
            [2, 4],
        ]
        table = pa.Table.from_arrays(
            [
                pa.array(range(7), type=pa.int32()),
                pa.array(["history"] * 4 + ["future"] * 3, type=pa.string()),
                pa.array([3] * 4 + [2] * 3, type=pa.int32()),
                pa.array([0, 3, 6, 9, 12, 14, 16], type=pa.int64()),
                pa.array(token_ids, type=pa.list_(pa.int32())),
                pa.array(["h0", "h1", "h2", "h3", "f0", "f1", "f2"], type=pa.string()),
            ],
            schema=schema,
        )
        pq.write_table(table, cls.samples_path)
        cls.histories, cls.futures = load_samples(cls.samples_path, None, None)
        cls.inference = CausalLMInference(cls.model_path, "cpu")

    def _train(
        self,
        output_dir: Path,
        *,
        history_batch: int = 4,
        max_steps: int | None = 100000,
        eval_every: int = 1000,
        tol: float = 1e-4,
        patience: int = 3,
        checkpoint_every: int = 1000,
        overwrite: bool = False,
    ) -> dict:
        return train_exp4(
            samples_path=self.samples_path,
            model_path=self.model_path,
            device="cpu",
            history_count=None,
            future_count=None,
            rank=2,
            nesting="sequential",
            history_batch=history_batch,
            future_batch=3,
            max_steps=max_steps,
            eval_every=eval_every,
            tol=tol,
            patience=patience,
            checkpoint_every=checkpoint_every,
            lr=1e-2,
            seed=0,
            log_every=100000,
            output_dir=output_dir,
            overwrite=overwrite,
        )

    def test_centered_logit_block_matches_every_full_vocabulary_entry(self) -> None:
        block = centered_logit_block(self.inference, self.histories, self.futures)
        self.assertEqual(block.shape, (4, 33))
        for history_index, history in enumerate(self.histories):
            for future_index, future in enumerate(self.futures):
                logits = self.inference.next_token_logits(
                    history.token_ids + future.token_ids
                )
                expected = logits - logits.mean()
                with self.subTest(history=history_index, future=future_index):
                    torch.testing.assert_close(
                        block[history_index, future_index * 11 : (future_index + 1) * 11],
                        expected,
                        rtol=0,
                        atol=1e-6,
                    )

    def test_training_and_analysis_recover_top_two_singular_values(self) -> None:
        result_dir = self.root / "convergence"
        result = self._train(result_dir / "train")
        self.assertEqual((result["rows"], result["columns"]), (4, 33))
        self.assertEqual(result["stop_reason"], "converged")
        self.assertEqual(result["pairs_evaluated"], result["steps"] * 4 * 3)
        self.assertFalse((result_dir / "train" / "checkpoint.pt").exists())
        metadata = json.loads((result_dir / "train" / "metadata.json").read_text())
        self.assertIs(metadata["converged"], True)
        self.assertEqual(metadata["steps"], result["steps"])
        analyze_exp4(result_dir=result_dir, device="cpu", overwrite=False)
        matrix = centered_logit_block(
            self.inference, self.histories, self.futures
        ).double().numpy()
        expected = np.linalg.svd(matrix, compute_uv=False)[:2]
        learned = np.load(result_dir / "analysis" / "singular_values.npy")
        self.assertEqual(learned.shape, (2,))
        np.testing.assert_allclose(learned, expected, rtol=1e-2, atol=0)
        self.assertTrue((result_dir / "analysis" / "neural_svd_spectrum.png").is_file())

    def test_interrupted_training_resumes_identically(self) -> None:
        uninterrupted_dir = self.root / "uninterrupted" / "train"
        resumed_dir = self.root / "resumed" / "train"
        uninterrupted_result = self._train(
            uninterrupted_dir, max_steps=10, eval_every=5, checkpoint_every=3
        )
        self._train(resumed_dir, max_steps=4, eval_every=5, checkpoint_every=3)
        resumed_result = self._train(
            resumed_dir, max_steps=10, eval_every=5, checkpoint_every=3
        )
        for output_dir, result in (
            (uninterrupted_dir, uninterrupted_result),
            (resumed_dir, resumed_result),
        ):
            with self.subTest(output_dir=output_dir):
                self.assertEqual(result["stop_reason"], "max_steps")
                self.assertEqual(result["steps"], 10)
                metadata = json.loads((output_dir / "metadata.json").read_text())
                self.assertEqual(metadata["steps"], 10)
                self.assertIs(metadata["converged"], False)
        uninterrupted = torch.load(
            uninterrupted_dir / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        resumed = torch.load(
            resumed_dir / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        self.assertEqual(uninterrupted["features"].keys(), resumed["features"].keys())
        for name, tensor in uninterrupted["features"].items():
            with self.subTest(feature=name):
                self.assertTrue(torch.equal(tensor, resumed["features"][name]))
        for name in ("loss_history", "generator"):
            with self.subTest(state=name):
                self.assertTrue(torch.equal(uninterrupted[name], resumed[name]))
        for name in ("previous_singular_values", "stable_evaluations"):
            with self.subTest(state=name):
                self.assertEqual(uninterrupted[name], resumed[name])
        uninterrupted_optimizer = uninterrupted["optimizer"]["state"]
        resumed_optimizer = resumed["optimizer"]["state"]
        self.assertEqual(len(uninterrupted_optimizer), 2)
        self.assertEqual(uninterrupted_optimizer.keys(), resumed_optimizer.keys())
        for parameter, state in uninterrupted_optimizer.items():
            for name in ("exp_avg", "exp_avg_sq"):
                with self.subTest(parameter=parameter, optimizer_state=name):
                    self.assertTrue(
                        torch.equal(state[name], resumed_optimizer[parameter][name])
                    )
        uninterrupted_features = torch.load(
            uninterrupted_dir / "features.pt", map_location="cpu", weights_only=True
        )
        resumed_features = torch.load(
            resumed_dir / "features.pt", map_location="cpu", weights_only=True
        )
        for name, tensor in uninterrupted_features.items():
            with self.subTest(snapshot_feature=name):
                self.assertTrue(torch.equal(tensor, resumed_features[name]))
                self.assertTrue(torch.equal(tensor, uninterrupted["features"][name]))
                self.assertTrue(
                    torch.equal(resumed_features[name], resumed["features"][name])
                )

    def test_analysis_reads_unconverged_snapshot(self) -> None:
        result_dir = self.root / "snapshot"
        train_dir = result_dir / "train"
        result = self._train(
            train_dir, max_steps=3, checkpoint_every=2, eval_every=1000
        )
        self.assertEqual(result["stop_reason"], "max_steps")
        self.assertTrue((train_dir / "checkpoint.pt").exists())
        summary = analyze_exp4(result_dir=result_dir, device="cpu", overwrite=False)
        self.assertIn("training: step 3, converged False", summary)
        spectrum = json.loads((result_dir / "analysis" / "spectrum.json").read_text())
        metadata = spectrum["train_metadata"]
        self.assertEqual(metadata["steps"], 3)
        self.assertIs(metadata["converged"], False)
        self.assertIsNone(metadata["final_relative_change"])
        self.assertEqual(np.load(train_dir / "loss_history.npy").shape, (3,))

    def test_resume_rejects_changed_config(self) -> None:
        output_dir = self.root / "changed_config" / "train"
        self._train(output_dir, max_steps=2)
        with self.assertRaisesRegex(
            ValueError, "checkpoint config mismatch.*history_batch"
        ):
            self._train(output_dir, max_steps=4, history_batch=3)

    def test_overwrite_restarts_from_scratch(self) -> None:
        output_dir = self.root / "overwrite" / "train"
        self._train(output_dir, max_steps=3, checkpoint_every=1)
        result = self._train(
            output_dir, max_steps=2, checkpoint_every=1, overwrite=True
        )
        self.assertEqual(result["steps"], 2)
        checkpoint = torch.load(
            output_dir / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        self.assertEqual(checkpoint["step"], 2)

    def test_existing_training_metadata_requires_overwrite(self) -> None:
        output_dir = self.root / "existing" / "train"
        output_dir.mkdir(parents=True)
        (output_dir / "metadata.json").touch()
        with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
            self._train(output_dir)

    def test_oversize_history_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds history count 4"):
            self._train(self.root / "oversize" / "train", history_batch=5)


if __name__ == "__main__":
    unittest.main()
