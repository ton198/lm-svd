import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from utils.inference import CausalLMInference
from utils.sample_store import (
    SampleStoreReader,
    block_seed,
    produce_blocks,
    read_shard_states,
    read_store_config,
    slot_bytes,
    slot_path,
)
from utils.stream_neural_svd import TokenStream, teacher_hidden_grid


class SampleStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        root = Path(directory.name)
        cls.model_path = root / "model"
        config = GPT2Config(vocab_size=11, n_positions=12, n_embd=8, n_layer=1, n_head=2,
                            bos_token_id=None, eos_token_id=None)
        GPT2LMHeadModel(config).save_pretrained(cls.model_path)
        cls.stream_path = root / "stream.npy"
        tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1], dtype=np.uint16)
        np.save(cls.stream_path, tokens)
        cls.stream_path.with_suffix(".json").write_text(json.dumps({
            "config": "tiny", "split": "train", "tokenizer_path": str(cls.model_path),
            "vocab_size": 11, "stream_tokens": len(tokens), "dtype": str(tokens.dtype),
        }), encoding="utf-8")
        cls.stream = TokenStream(cls.stream_path)
        cls.inference = CausalLMInference(cls.model_path, "cpu")

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store_dir = Path(directory.name) / "store"

    def produce(self, **overrides) -> dict:
        kwargs = dict(store_dir=self.store_dir, model_path=self.model_path,
                      stream_path=self.stream_path, device="cpu", history_tokens=3,
                      future_tokens=2, block_histories=4, block_futures=3,
                      max_bytes=3 * slot_bytes(4, 3, 8), shards=1, shard=0, seed=0,
                      pair_batch=12, max_blocks=1, log_every=100, overwrite=False)
        kwargs.update(overrides)
        return produce_blocks(**kwargs)

    def reader(self, **overrides) -> SampleStoreReader:
        kwargs = dict(store_dir=self.store_dir, model_path=self.model_path,
                      stream_path=self.stream_path, stream=self.stream, hidden_size=8,
                      vocab_size=11, history_tokens=3, future_tokens=2,
                      history_batch=4, future_batch=3)
        kwargs.update(overrides)
        return SampleStoreReader(**kwargs)

    def test_blocks_match_teacher_and_ring_wraps(self) -> None:
        self.produce(max_blocks=5)
        self.assertEqual(read_store_config(self.store_dir)["num_slots"], 3)
        self.assertEqual(read_shard_states(self.store_dir, 1)[0], {"next_sequence": 5, "filled": 3})
        for slot, sequence in enumerate((3, 4, 2)):
            with np.load(slot_path(self.store_dir, slot)) as data:
                self.assertEqual(int(data["sequence"]), sequence)
                self.assertEqual(int(data["shard"]), 0)
        gen = torch.Generator().manual_seed(block_seed(0, 0, 2))
        hs = self.stream.sample_starts(4, 3, gen)
        fs = self.stream.sample_starts(3, 2, gen)
        expected = teacher_hidden_grid(self.inference, self.stream.windows(hs, 3),
                                       self.stream.windows(fs, 2), pair_batch=12)
        with np.load(slot_path(self.store_dir, 2)) as data:
            np.testing.assert_array_equal(data["history_starts"], hs.numpy())
            np.testing.assert_array_equal(data["future_starts"], fs.numpy())
            np.testing.assert_array_equal(data["states"], expected.permute(1, 0, 2).half().numpy())

    def test_restart_continues_and_config_change_requires_overwrite(self) -> None:
        self.produce(max_blocks=2)
        self.produce(max_blocks=2)
        self.assertEqual(read_shard_states(self.store_dir, 1)[0]["next_sequence"], 4)
        with self.assertRaisesRegex(ValueError, "sample store config mismatch.*block_futures"):
            self.produce(block_futures=2)
        self.produce(block_futures=2, overwrite=True)
        self.assertEqual(read_shard_states(self.store_dir, 1)[0]["next_sequence"], 1)

    def test_second_producer_on_same_shard_is_rejected(self) -> None:
        self.produce(max_blocks=0)
        process = subprocess.Popen(
            [sys.executable, "-c", "import fcntl, sys, time; f = open(sys.argv[1], 'w'); "
             "fcntl.lockf(f, fcntl.LOCK_EX); print('locked', flush=True); time.sleep(60)",
             str(self.store_dir / "shard-000.lock")],
            stdout=subprocess.PIPE, text=True,
        )
        try:
            self.assertEqual(process.stdout.readline().strip(), "locked")
            with self.assertRaisesRegex(ValueError, "another producer holds shard 0"):
                self.produce()
        finally:
            process.terminate()
            process.wait()
            process.stdout.close()

    def test_shards_fill_disjoint_slots_and_reader_samples_only_filled(self) -> None:
        params = dict(shards=2, max_bytes=4 * slot_bytes(4, 3, 8))
        self.produce(**params, shard=0, max_blocks=3)
        self.produce(**params, shard=1, max_blocks=1)
        for slot in range(3):
            self.assertTrue(slot_path(self.store_dir, slot).is_file())
        self.assertFalse(slot_path(self.store_dir, 3).exists())
        all_histories = []
        for slot in range(3):
            with np.load(slot_path(self.store_dir, slot)) as data:
                all_histories.append(sorted(data["history_starts"].tolist()))
        gen = torch.Generator().manual_seed(7)
        reader = self.reader()
        for _ in range(200):
            hs, _, _ = reader.sample(gen)
            self.assertIn(sorted(hs.tolist()), all_histories)

    def test_reader_subgrid_and_validation(self) -> None:
        self.produce()
        reader = self.reader(history_batch=2, future_batch=2)
        hs, fs, states = reader.sample(torch.Generator().manual_seed(17))
        with np.load(slot_path(self.store_dir, 0)) as data:
            all_hs = data["history_starts"]
            all_fs = data["future_starts"]
            all_states = data["states"]
            # Starts may be duplicated; any matching row and column combination is valid.
            for i, h in enumerate(hs.tolist()):
                for j, f in enumerate(fs.tolist()):
                    candidates = [all_states[r, c] for r in np.flatnonzero(all_hs == h)
                                  for c in np.flatnonzero(all_fs == f)]
                    self.assertTrue(any(np.array_equal(states[i, j].numpy(), value)
                                        for value in candidates))
        with self.assertRaisesRegex(ValueError, "history_batch 5 exceeds sample store block_histories 4"):
            self.reader(history_batch=5)
        with self.assertRaisesRegex(ValueError, r"does not match training \(future_tokens"):
            self.reader(future_tokens=3)
        with self.assertRaisesRegex(ValueError, "sample store does not exist"):
            self.reader(store_dir=self.store_dir / "missing")
