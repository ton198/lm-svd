import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.exp4_neural_svd.analysis import run_analysis
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
        lr_schedule: str = "constant",
        min_lr: float = 1e-4,
        lr_decay_steps: int = 20,
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
            lr_schedule=lr_schedule,
            min_lr=min_lr,
            lr_decay_steps=lr_decay_steps,
            seed=0,
            log_every=100000,
            output_dir=output_dir,
            overwrite=overwrite,
        )

    def test_legacy_constant_checkpoint_switches_to_cosine(self) -> None:
        output_dir = self.root / "legacy_cosine" / "train"
        self._train(output_dir, max_steps=3, checkpoint_every=1)
        checkpoint_path = output_dir / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        del checkpoint["lr_schedule"]
        torch.save(checkpoint, checkpoint_path)
        kwargs = dict(
            lr_schedule="cosine",
            lr_decay_steps=4,
            max_steps=None,
            checkpoint_every=1,
        )
        result = self._train(output_dir, **kwargs)
        self.assertEqual(result["stop_reason"], "lr_schedule_end")
        self.assertEqual(result["steps"], 7)
        metadata = json.loads((output_dir / "metadata.json").read_text())
        self.assertEqual(metadata["lr_schedule"]["decay_start_step"], 3)
        self.assertEqual(metadata["lr_schedule"]["name"], "cosine")
        self.assertEqual(metadata["final_lr"], 1e-4)
        self.assertTrue(checkpoint_path.exists())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for group in checkpoint["optimizer"]["param_groups"]:
            self.assertEqual(group["lr"], 1e-4)
        result = self._train(output_dir, **kwargs)
        self.assertEqual(result["stop_reason"], "lr_schedule_end")
        self.assertEqual(result["steps"], 7)

    def test_resume_rejects_changed_lr_schedule(self) -> None:
        output_dir = self.root / "changed_schedule" / "train"
        self._train(output_dir, lr_schedule="cosine", max_steps=2, lr_decay_steps=20)
        with self.assertRaisesRegex(
            ValueError, "checkpoint lr schedule mismatch.*decay_steps"
        ):
            self._train(output_dir, lr_schedule="cosine", lr_decay_steps=10)
        with self.assertRaisesRegex(
            ValueError, "checkpoint lr schedule mismatch.*name"
        ):
            self._train(output_dir, lr_schedule="constant")

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

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW") == "1",
        "convergence training is slow; set RUN_SLOW=1 to run it",
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
        run_analysis(result_dir=result_dir, device="cpu", overwrite=False)
        matrix = centered_logit_block(
            self.inference, self.histories, self.futures
        ).double().numpy()
        expected = np.linalg.svd(matrix, compute_uv=False)[:2]
        learned = np.load(result_dir / "analysis" / "singular_values.npy")
        self.assertEqual(learned.shape, (2,))
        np.testing.assert_allclose(learned, expected, rtol=1e-2, atol=0)
        self.assertTrue((result_dir / "analysis" / "neural_svd_spectrum.png").is_file())

    def test_interrupted_training_resumes_identically(self) -> None:
        for lr_schedule in ("constant", "cosine"):
            with self.subTest(lr_schedule=lr_schedule):
                uninterrupted_dir = self.root / lr_schedule / "uninterrupted" / "train"
                resumed_dir = self.root / lr_schedule / "resumed" / "train"
                kwargs = dict(
                    eval_every=5, checkpoint_every=3,
                    lr_schedule=lr_schedule, lr_decay_steps=20,
                )
                uninterrupted_result = self._train(
                    uninterrupted_dir, max_steps=10, **kwargs
                )
                self._train(resumed_dir, max_steps=4, **kwargs)
                resumed_result = self._train(resumed_dir, max_steps=10, **kwargs)
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
                self.assertEqual(
                    uninterrupted["optimizer"]["param_groups"][0]["lr"],
                    resumed["optimizer"]["param_groups"][0]["lr"],
                )
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
        summary = run_analysis(result_dir=result_dir, device="cpu", overwrite=False)
        self.assertIn("training: step 3, converged False", summary)
        metadata = json.loads((result_dir / "analysis" / "metadata.json").read_text())
        train_metadata = metadata["input"]["train_metadata"]
        self.assertEqual(train_metadata["steps"], 3)
        self.assertIs(train_metadata["converged"], False)
        self.assertIsNone(train_metadata["final_relative_change"])
        self.assertEqual(np.load(train_dir / "loss_history.npy").shape, (3,))

    def test_resume_rejects_changed_config(self) -> None:
        output_dir = self.root / "changed_config" / "train"
        self._train(output_dir, max_steps=2)
        with self.assertRaisesRegex(
            ValueError, "checkpoint config mismatch.*history_batch"
        ):
            self._train(output_dir, max_steps=4, history_batch=3)

    def test_resume_rejects_sparse_adam_checkpoint(self) -> None:
        output_dir = self.root / "legacy_optimizer" / "train"
        self._train(output_dir, max_steps=2, checkpoint_every=1)
        checkpoint_path = output_dir / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        del checkpoint["config"]["optimizer"]
        torch.save(checkpoint, checkpoint_path)
        with self.assertRaisesRegex(ValueError, "checkpoint config mismatch.*optimizer"):
            self._train(output_dir, max_steps=4)

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


class Exp4AnalysisTests(unittest.TestCase):
    def _write_snapshot(self, result_dir: Path, values: np.ndarray) -> None:
        train_dir = result_dir / "train"
        train_dir.mkdir(parents=True)
        r = values.size
        torch.save(
            {
                "history.weight": torch.diag(
                    torch.tensor(values, dtype=torch.float32)
                ),
                "future.weight": torch.eye(r, dtype=torch.float32).reshape(1, r * r),
            },
            train_dir / "features.pt",
        )
        np.save(train_dir / "token_gram.npy", np.eye(r, dtype=np.float64))
        np.save(train_dir / "loss_history.npy", np.array([3.0, 2.0, 1.0]))
        metadata = {
            "history_count": r,
            "future_count": 1,
            "vocab_size": 4,
            "hidden_size": r,
            "rank": r,
            "entry_mean_square": float((values**2).sum() / (r * 4)),
            "steps": 3,
            "converged": False,
        }
        (train_dir / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def test_log_log_fit_recovers_power_law_from_snapshot(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        result_dir = Path(temporary_directory.name)
        values = 10.0 * np.arange(1, 4) ** -1.5
        self._write_snapshot(result_dir, values)

        summary = run_analysis(result_dir=result_dir, device="cpu", overwrite=False)

        analysis_dir = result_dir / "analysis"
        np.testing.assert_allclose(
            np.load(analysis_dir / "singular_values.npy"), values, rtol=1e-6
        )
        metadata = json.loads((analysis_dir / "metadata.json").read_text())
        self.assertEqual(
            list(metadata), ["input", "spectrum", "orthogonality", "loglog_fit"]
        )
        self.assertEqual(
            {path.name for path in analysis_dir.iterdir()},
            {
                "singular_values.npy",
                "left_singular_vectors.npy",
                "metadata.json",
                "neural_svd_spectrum.png",
            },
        )
        np.testing.assert_allclose(
            metadata["spectrum"]["singular_values"], values, rtol=1e-6
        )
        self.assertEqual(metadata["spectrum"]["order"], "learned")
        self.assertEqual(metadata["orthogonality"]["left_error"], 0.0)
        fit = metadata["loglog_fit"]
        self.assertAlmostEqual(fit["slope"], -1.5, places=6)
        self.assertAlmostEqual(fit["intercept"], 1.0, places=6)
        self.assertEqual(fit["point_count"], 3)
        self.assertEqual(fit["log_base"], 10)
        self.assertIn(
            "loglog fit: k = -1.5, b = 1 (3 points, log10)",
            summary.splitlines(),
        )
        self.assertEqual(
            (analysis_dir / "neural_svd_spectrum.png").read_bytes()[:8],
            b"\x89PNG\r\n\x1a\n",
        )

    def test_log_log_fit_uses_learned_order(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        result_dir = Path(temporary_directory.name)
        values = (10.0 * np.arange(1, 4) ** -1.5)[::-1].copy()
        self._write_snapshot(result_dir, values)

        run_analysis(result_dir=result_dir, device="cpu", overwrite=False)

        analysis_dir = result_dir / "analysis"
        np.testing.assert_allclose(
            np.load(analysis_dir / "singular_values.npy"), values, rtol=1e-6
        )
        metadata = json.loads((analysis_dir / "metadata.json").read_text())
        fit = metadata["loglog_fit"]
        expected_slope = np.polyfit(
            np.log10([1, 2, 3]), np.log10(values), 1
        )[0]
        self.assertAlmostEqual(fit["slope"], expected_slope, places=6)
        self.assertGreater(fit["slope"], 0)

    def test_rank_one_snapshot_skips_fit(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        result_dir = Path(temporary_directory.name)
        values = np.array([2.0])
        self._write_snapshot(result_dir, values)

        summary = run_analysis(result_dir=result_dir, device="cpu", overwrite=False)

        analysis_dir = result_dir / "analysis"
        metadata = json.loads((analysis_dir / "metadata.json").read_text())
        fit = metadata["loglog_fit"]
        self.assertIsNone(fit["slope"])
        self.assertIsNone(fit["intercept"])
        self.assertEqual(fit["point_count"], 1)
        self.assertIn(
            "loglog fit: skipped (fewer than 2 points: 1)", summary.splitlines()
        )
        self.assertTrue((analysis_dir / "neural_svd_spectrum.png").is_file())

    def test_existing_outputs_require_overwrite(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        result_dir = Path(temporary_directory.name)
        self._write_snapshot(result_dir, np.array([3.0, 2.0, 1.0]))

        run_analysis(result_dir=result_dir, device="cpu", overwrite=False)
        with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
            run_analysis(result_dir=result_dir, device="cpu", overwrite=False)
        run_analysis(result_dir=result_dir, device="cpu", overwrite=True)



if __name__ == "__main__":
    unittest.main()
