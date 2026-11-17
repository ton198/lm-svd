import contextlib
import io
import json
import shutil
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.exp5_gpt2_mlp_table.tables import load_eval_table
from experiments.exp7_independent_subnets import center
from experiments.exp7_independent_subnets.center import (
    CenteredPairs,
    CenterPool,
    build_center_pool,
    center_states,
    centered_rows,
    table_baseline,
)
from experiments.exp7_independent_subnets.cli import main
from test_exp5_gpt2_mlp_table import (
    FUTURE_TOKENS,
    FUTURES,
    HISTORIES,
    HISTORY_TOKENS,
    TOP_K,
)
# setUpModule silences the progress bars of the models the fixtures load.
from test_exp7_independent_subnets import Exp7Fixture, TrainingHelpers, setUpModule  # noqa: F401

POOL_HISTORIES, POOL_FUTURES = 3, 10


class CenterFixture(TrainingHelpers, Exp7Fixture):
    """Exp7Fixture with a finished center pool of the tiny model."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.stream_path = cls.root / "stream.npy"
        cls.pool_dir = cls.root / "pool"
        cls.build_pool(cls.pool_dir)
        cls.pool = CenterPool(cls.pool_dir)

    @classmethod
    def build_pool(cls, pool_dir, **changes):
        options = dict(
            pool_dir=pool_dir, model_path=cls.model_path, stream_path=cls.stream_path,
            device="cpu", histories=POOL_HISTORIES, futures=POOL_FUTURES,
            history_tokens=HISTORY_TOKENS, future_tokens=FUTURE_TOKENS, seed=0,
            overwrite=False, chunk=4, save_every=4,
        )
        options.update(changes)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            summary = build_center_pool(**options)
        return summary, output.getvalue()

    def entries(self, histories, futures):
        """(N, F, V): centered logits after every history + future, one pass per pair."""
        ids = [h + f for h in histories.tolist() for f in futures.tolist()]
        logits = self.inference.next_token_logits_batch(ids)
        centered = logits - logits.mean(dim=-1, keepdim=True)
        return centered.unflatten(0, (len(histories), len(futures)))


class CenterStateTests(CenterFixture):
    def test_center_states_are_the_mean_final_state_in_any_chunking(self):
        histories, futures = torch.tensor(HISTORIES[:4]), torch.tensor(FUTURES)
        expected = torch.stack([
            self.inference.final_hidden_states(
                torch.cat([histories, future.expand(len(histories), -1)], dim=1))[:, -1].mean(0)
            for future in futures
        ])
        for chunk in (1, 3, 16, 1024):
            states = center_states(self.inference, histories, futures, chunk)
            self.assertEqual(states.dtype, torch.float32)
            torch.testing.assert_close(states, expected, rtol=0, atol=1e-5)

    def test_the_center_of_a_token_is_the_mean_entry_over_the_histories(self):
        histories, futures = torch.tensor(HISTORIES[:4]), torch.tensor(FUTURES)
        states = center_states(self.inference, histories, futures)
        center = states @ centered_rows(self.inference).T  # (F, V)
        torch.testing.assert_close(
            center, self.entries(histories, futures).mean(dim=0), rtol=0, atol=1e-4)


class PoolTests(CenterFixture):
    def test_pool_files_hold_the_windows_and_their_center_states(self):
        pool = self.pool
        self.assertEqual(tuple(pool.histories.shape), (POOL_HISTORIES, HISTORY_TOKENS))
        self.assertEqual(tuple(pool.futures.shape), (POOL_FUTURES, FUTURE_TOKENS))
        torch.testing.assert_close(
            pool.states, center_states(self.inference, pool.histories, pool.futures),
            rtol=0, atol=1e-5)
        self.assertFalse((self.pool_dir / "progress.json").exists())
        self.assertEqual(pool.config["histories"], POOL_HISTORIES)
        # Windows of the stream, which is an arithmetic run.
        stream = np.load(self.stream_path).astype(np.int64)
        runs = {tuple(stream[i:i + FUTURE_TOKENS]) for i in range(len(stream) - 1)}
        self.assertTrue(all(tuple(row) in runs for row in pool.futures.tolist()))

    def test_a_finished_pool_is_kept_and_other_settings_are_rejected(self):
        before = (self.pool_dir / "states.npy").stat().st_mtime_ns
        _, log = self.build_pool(self.pool_dir)
        self.assertIn("already holds a finished center pool", log)
        self.assertEqual((self.pool_dir / "states.npy").stat().st_mtime_ns, before)
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.build_pool(self.pool_dir, histories=POOL_HISTORIES + 1)

    def test_an_interrupted_build_continues_where_it_stopped(self):
        pool_dir = self.root / "pool-interrupted"
        calls = []
        real = center.center_states

        def failing(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return real(*args, **kwargs)

        with patch.object(center, "center_states", failing), \
                self.assertRaises(KeyboardInterrupt):
            self.build_pool(pool_dir)
        progress = json.loads((pool_dir / "progress.json").read_text())
        self.assertEqual(progress["done"], 4)
        self.assertFalse((pool_dir / "metadata.json").exists())
        with self.assertRaisesRegex(ValueError, "no finished center pool"):
            CenterPool(pool_dir)
        _, log = self.build_pool(pool_dir)
        self.assertIn("Resuming the center pool at future 4", log)
        resumed = CenterPool(pool_dir)
        torch.testing.assert_close(resumed.states, self.pool.states, rtol=0, atol=1e-6)
        self.assertEqual(resumed.futures.tolist(), self.pool.futures.tolist())

    def test_overwrite_starts_over_with_other_settings(self):
        pool_dir = self.root / "pool-overwritten"
        self.build_pool(pool_dir)
        self.build_pool(pool_dir, futures=6, overwrite=True)
        self.assertEqual(len(CenterPool(pool_dir).futures), 6)

    def test_rejects_windows_longer_than_the_model(self):
        with self.assertRaisesRegex(ValueError, "positions"):
            self.build_pool(self.root / "pool-long", history_tokens=7)
        self.assertFalse((self.root / "pool-long").exists())


class CenteredTargetTests(CenterFixture):
    def test_a_table_centered_on_its_own_histories_has_columns_of_mean_zero(self):
        table_dir = self.inference_dir / "train"
        tokens = np.load(table_dir / "tokens.npz")
        own = types.SimpleNamespace(histories=torch.from_numpy(tokens["histories"]).long())
        table = load_eval_table(
            table_dir, self.inference, "cpu", baseline=table_baseline(self.inference, own))
        stored = np.load(table_dir / "matrix.npy")
        expected = stored - stored.mean(axis=0)
        np.testing.assert_allclose(table.table.matrix.numpy(), expected, rtol=0, atol=1e-4)
        np.testing.assert_allclose(
            table.table.singular_values.numpy()[:3],
            np.linalg.svd(expected, compute_uv=False)[:3], rtol=1e-4)

    def test_pairs_are_pool_futures_and_stream_histories_with_the_centered_entries(self):
        pairs = CenteredPairs(self.inference, self.pool, TOP_K, seed=0, encode_batch=4)
        histories, futures, header_ids, target = pairs.draw(12)
        self.assertEqual(tuple(target.shape), (12, TOP_K))
        pool_futures = {tuple(row) for row in self.pool.futures.tolist()}
        self.assertTrue(all(tuple(row) in pool_futures for row in futures.tolist()))
        for b in range(12):
            entry = self.entries(histories[b:b + 1], futures[b:b + 1])[0, 0]
            center = self.entries(self.pool.histories, futures[b:b + 1])[:, 0].mean(dim=0)
            torch.testing.assert_close(
                target[b], (entry - center)[header_ids[b]], rtol=0, atol=1e-4)
            # The header tokens are the top-k tokens after the future alone.
            alone = self.inference.next_token_logits_batch([futures[b].tolist()])[0]
            self.assertEqual(
                sorted(header_ids[b].tolist()), sorted(alone.topk(TOP_K).indices.tolist()))

    def test_two_draws_differ_and_a_seed_repeats_them(self):
        first = CenteredPairs(self.inference, self.pool, TOP_K, seed=3, encode_batch=4)
        again = CenteredPairs(self.inference, self.pool, TOP_K, seed=3, encode_batch=4)
        a, b = first.draw(8), first.draw(8)
        self.assertNotEqual(a[0].tolist(), b[0].tolist())
        self.assertEqual(a[0].tolist(), again.draw(8)[0].tolist())


class CenteredTrainingTests(CenterFixture):
    def test_a_centered_run_trains_two_stages_without_a_store(self):
        output_dir = self.root / "centered-run"
        summary = self.train(
            output_dir, center_pool=self.pool_dir, store_dir=self.root / "no-such-store")
        self.assertEqual(summary["modes"], 2)
        self.assertIn("fixed histories", self.log)
        self.assertNotIn("Waiting for the first sample block", self.log)
        metadata = json.loads((output_dir / "metadata.json").read_text())
        self.assertEqual(metadata["center_pool"], self.pool.config)
        self.assertNotIn("store", metadata)
        self.assertIsNone(metadata["store_dir"])
        self.assertTrue(metadata["first_mode"].startswith("none"))
        checkpoint = torch.load(output_dir / "checkpoint.pt", weights_only=True)
        self.assertEqual(checkpoint["config"]["center_pool"], str(self.pool_dir))
        records = self.records(output_dir)
        self.assertEqual([record["stage"] for record in records], [1, 1, 2, 2])
        # Stage 1 fits the target itself, so its residual is the target.
        self.assertAlmostEqual(records[0]["residual_energy"], records[0]["kernel_energy"])
        self.assertEqual(len(self.stages(output_dir)), 2)
        # Repeating the command finds both modes finished.
        again = self.train(
            output_dir, center_pool=self.pool_dir, store_dir=self.root / "no-such-store")
        self.assertEqual(again["evaluations"], 4)

    def test_the_tables_of_a_centered_run_lose_the_pool_center(self):
        output_dir = self.root / "centered-tables"
        seen = {}
        real = center.prepare_centered_run

        def recording(**kwargs):
            result = real(**kwargs)
            seen["tables"] = result[1]
            return result

        with patch("experiments.exp7_independent_subnets.train.prepare_centered_run", recording):
            self.train(output_dir, center_pool=self.pool_dir, modes=1)
        for name in ("train", "val"):
            tokens = np.load(self.inference_dir / name / "tokens.npz")
            futures = torch.from_numpy(tokens["futures"]).long()
            header = torch.from_numpy(tokens["header_token_ids"]).long()
            center_entries = self.entries(self.pool.histories, futures).mean(dim=0)  # (F, V)
            expected = (np.load(self.inference_dir / name / "matrix.npy")
                        - center_entries.gather(1, header).reshape(-1).numpy())
            np.testing.assert_allclose(
                seen["tables"][name].table.matrix.numpy(), expected, rtol=0, atol=1e-4)

    def test_a_run_on_the_target_minus_g_keeps_its_old_settings(self):
        output_dir = self.root / "store-run"
        self.train(output_dir, modes=1)
        config = torch.load(output_dir / "checkpoint.pt", weights_only=True)["config"]
        self.assertNotIn("center_pool", config)
        self.assertEqual(config["store_dir"], str(self.store_dir))
        # The same directory cannot be continued on the other target.
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.train(output_dir, modes=2, center_pool=self.pool_dir)

    def test_a_pool_of_another_window_length_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "pool"
            self.build_pool(other, history_tokens=HISTORY_TOKENS - 1)
            with self.assertRaisesRegex(ValueError, "history_tokens"):
                self.train(self.root / "wrong-pool", center_pool=other)
        self.assertFalse((self.root / "wrong-pool").exists())

    def test_an_unfinished_pool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no finished center pool"):
            self.train(self.root / "no-pool-run", center_pool=self.root / "no-pool")


class CenterCLITests(CenterFixture):
    def test_center_pool_then_train_writes_a_run_named_after_the_histories(self):
        results_root = self.root / "center-results"
        shutil.copytree(
            self.inference_dir,
            results_root / "exp4_table_neural_svd" / "tiny__model" / "inference",
        )
        pool_dir = self.root / "cli-pool"
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main([
                "center-pool", "--pool-dir", str(pool_dir), "--model-path", str(self.model_path),
                "--stream-path", str(self.stream_path), "--histories", "2", "--futures", "6",
                "--history-tokens", str(HISTORY_TOKENS), "--future-tokens", str(FUTURE_TOKENS),
                "--device", "cpu",
            ])
        self.assertIn("futures: 6", output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main([
                "train", "--results-root", str(results_root), "--dataset", "tiny",
                "--model-path", str(self.model_path), "--center-pool", str(pool_dir),
                "--subnet", "scratch", "--modes", "1", "--stage-steps", "2",
                "--min-stage-steps", "0", "--patience", "0", "--eval-every", "2",
                "--align-pairs", "16", "--scratch-layers", "1", "--scratch-width", "4",
                "--scratch-heads", "2", "--batch-size", "8", "--device", "cpu",
            ])
        run = results_root / "exp7_independent_subnets" / "tiny__model" / "train-scratch-center2"
        self.assertTrue((run / "modes" / "mode-01.pt").is_file())
        self.assertIn(f"output_dir: {run}", output.getvalue())

    def test_train_with_a_missing_pool_is_a_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()) as error, \
                self.assertRaises(SystemExit) as raised:
            main(["train", "--center-pool", str(self.root / "missing-pool"), "--device", "cpu"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("exp7 center-pool", error.getvalue())
