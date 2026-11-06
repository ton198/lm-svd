import fcntl
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from data_pipline import SampleSampler
from data_pipline.sample_store import read_shard_states, slot_bytes, slot_path
from data_pipline.sample_store_producer import produce_blocks
from utils.common import write_atomic
from utils.inference import CausalLMInference


class SampleStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        root = Path(tmp.name)
        cls.model_path = root / 'model'
        GPT2LMHeadModel(GPT2Config(
            vocab_size=11, n_positions=12, n_embd=8, n_layer=1, n_head=2,
            bos_token_id=None, eos_token_id=None,
        )).save_pretrained(cls.model_path)
        tokenizer = Tokenizer(WordLevel({'[UNK]': 0, **{str(i): i for i in range(1, 11)}}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', model_max_length=12).save_pretrained(cls.model_path)
        cls.stream_path = root / 'stream.npy'
        np.save(cls.stream_path, np.arange(80, dtype=np.uint16) % 11)
        cls.stream_path.with_suffix('.json').write_text(json.dumps({
            'vocab_size': 11, 'config': 'tiny', 'split': 'train',
            'tokenizer_path': str(cls.model_path),
        }))
        cls.inference = CausalLMInference(cls.model_path, 'cpu')

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = Path(tmp.name) / 'store'

    def produce(self, **changes):
        kwargs = dict(
            store_dir=self.store, model_path=self.model_path, stream_path=self.stream_path,
            device='cpu', history_tokens=3, future_tokens=2, samples_per_block=4,
            max_bytes=3 * slot_bytes(4, 3, 2, 11), shards=1, shard=0, seed=0,
            batch_size=2, max_blocks=1, log_every=100, overwrite=False,
        )
        kwargs.update(changes)
        return produce_blocks(**kwargs)

    def assert_sample_matches_teacher(self, sample):
        h, f, logits = sample
        self.assertEqual((h.shape, f.shape, logits.shape), ((3,), (2,), (11,)))
        self.assertEqual((h.dtype, f.dtype, logits.dtype), (torch.int64, torch.int64, torch.float32))
        self.assertTrue(all(t.device.type == 'cpu' and not t.requires_grad for t in sample))
        expected = self.inference.next_token_logits(torch.cat((h, f)).tolist())
        torch.testing.assert_close(logits, expected, rtol=0, atol=1e-6)

    def test_saved_samples_contain_raw_logits_and_ring_wraps(self):
        self.produce(max_blocks=5)
        self.assertEqual(read_shard_states(self.store, 1)[0], {'next_sequence': 5, 'filled': 3})
        files = list((self.store / 'slots').glob('*.npy'))
        self.assertEqual(len(files), 3)
        self.assertLessEqual(sum(p.stat().st_size for p in files), 3 * slot_bytes(4, 3, 2, 11))
        for path in files:
            block = np.load(path, mmap_mode='r', allow_pickle=False)
            self.assertIsInstance(block, np.memmap)
            self.assertEqual(block.shape, (4,))
            for row in block:
                self.assert_sample_matches_teacher(tuple(torch.from_numpy(row[k].copy()) for k in ('h', 'f', 'logits')))
            block._mmap.close()
        sampler = SampleSampler(self.store, seed=7)
        for _ in range(12):
            self.assert_sample_matches_teacher(sampler.sample())

    def test_reproducible_uniform_sampling_returns_copies(self):
        self.produce(max_blocks=2)
        for slot in range(2):
            path = slot_path(self.store, slot)
            data = np.load(path).copy()
            for i in range(4):
                data['h'][i, 0] = slot * 4 + i
            np.save(path, data)
        a, b = SampleSampler(self.store, seed=19), SampleSampler(self.store, seed=19)
        counts = Counter()
        for _ in range(1600):
            x, y = a.sample(), b.sample()
            for left, right in zip(x, y):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            counts[int(x[0][0])] += 1
        self.assertEqual(set(counts), set(range(8)))
        self.assertTrue(all(140 < n < 260 for n in counts.values()), counts)
        held = a.sample()
        copies = tuple(t.clone() for t in held)
        self.produce(max_blocks=4)
        for left, right in zip(held, copies):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        held[0].fill_(99)
        self.assertTrue(all(int(a.sample()[0].max()) < 11 for _ in range(10)))

    def test_sampling_needs_no_source_or_model(self):
        stream = self.store.parent / 'local.npy'
        np.save(stream, np.arange(30, dtype=np.uint16) % 11)
        stream.with_suffix('.json').write_text(json.dumps({'vocab_size': 11, 'tokenizer_path': str(self.model_path)}))
        self.produce(stream_path=stream)
        stream.unlink()
        stream.with_suffix('.json').unlink()
        script = """
import sys
from data_pipline import SampleSampler
assert 'transformers' not in sys.modules
h, f, logits = SampleSampler(sys.argv[1], seed=3).sample()
assert h.shape == (3,) and f.shape == (2,) and logits.shape == (11,)
assert 'transformers' not in sys.modules
"""
        subprocess.run([sys.executable, '-c', script, str(self.store)], check=True)

    def test_restart_is_identical_to_uninterrupted_generation(self):
        self.produce(max_blocks=2)
        self.produce(max_blocks=3)
        other = self.store.parent / 'continuous'
        self.produce(store_dir=other, max_blocks=5)
        for slot in range(3):
            np.testing.assert_array_equal(np.load(slot_path(self.store, slot)), np.load(slot_path(other, slot)))
        with self.assertRaisesRegex(ValueError, 'config mismatch'):
            self.produce(samples_per_block=2)
        self.produce(samples_per_block=2, overwrite=True)
        self.assertEqual(read_shard_states(self.store, 1)[0]['next_sequence'], 1)

    def test_multishard_sampler_only_reads_published_slots(self):
        kwargs = dict(shards=2, max_bytes=4 * slot_bytes(4, 3, 2, 11))
        self.produce(**kwargs, shard=0, max_blocks=3)
        self.produce(**kwargs, shard=1)
        self.assertFalse(slot_path(self.store, 3).exists())
        sampler = SampleSampler(self.store, seed=2)
        for _ in range(30):
            self.assert_sample_matches_teacher(sampler.sample())
        self.produce(**kwargs, shard=1, max_blocks=2)
        for _ in range(10):
            self.assert_sample_matches_teacher(sampler.sample())

    def test_atomic_overwrite_keeps_reader_snapshot_consistent(self):
        self.produce(max_bytes=slot_bytes(4, 3, 2, 11))
        sampler = SampleSampler(self.store)
        observed = []
        def pause_before_replace(path, writer):
            def wrapped(tmp):
                writer(tmp)
                if path.suffix == '.npy':
                    observed.append(sampler.sample())
            write_atomic(path, wrapped)
        before = np.load(slot_path(self.store, 0)).copy()
        with patch('data_pipline.sample_store_producer.write_atomic', side_effect=pause_before_replace):
            self.produce(max_bytes=slot_bytes(4, 3, 2, 11), max_blocks=3)
        self.assertEqual(len(observed), 3)
        for sample in observed:
            self.assert_sample_matches_teacher(sample)
        self.assertTrue(any(np.array_equal(observed[0][0].numpy(), row['h']) and
                            np.array_equal(observed[0][1].numpy(), row['f']) for row in before))

    def test_empty_store_errors_and_waits_for_first_published_block(self):
        self.produce(max_blocks=0)
        sampler = SampleSampler(self.store)
        with self.assertRaisesRegex(ValueError, 'no published samples'):
            sampler.sample()
        finished = threading.Event()
        thread = threading.Thread(target=lambda: (sampler.wait_until_ready(), finished.set()), daemon=True)
        thread.start()
        self.assertFalse(finished.wait(0.05))
        self.produce()
        self.assertTrue(finished.wait(3))
        thread.join()
        self.assert_sample_matches_teacher(sampler.sample())

    def test_legacy_store_requires_explicit_rebuild(self):
        self.store.mkdir()
        (self.store / 'store.json').write_text(json.dumps({'format': 'lm-svd-sample-store-v1'}))
        with self.assertRaisesRegex(ValueError, 'format.*rebuild'):
            self.produce()
        with self.assertRaisesRegex(ValueError, 'format.*rebuild'):
            SampleSampler(self.store)
        self.produce(overwrite=True)
        self.assert_sample_matches_teacher(SampleSampler(self.store).sample())

    def test_duplicate_producer_and_overwrite_are_rejected_while_active(self):
        self.produce(max_blocks=0)
        script = """
import fcntl, os, sys, time
d = os.open(sys.argv[1], os.O_RDONLY)
fcntl.flock(d, fcntl.LOCK_SH)
f = open(sys.argv[1] + '/shard-000.lock', 'a')
fcntl.flock(f, fcntl.LOCK_EX)
print('locked', flush=True)
time.sleep(60)
"""
        process = subprocess.Popen([sys.executable, '-c', script, str(self.store)], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), 'locked')
            with self.assertRaisesRegex(ValueError, 'another producer'):
                self.produce()
            with self.assertRaisesRegex(ValueError, 'active'):
                self.produce(overwrite=True)
        finally:
            process.terminate()
            process.wait()
            process.stdout.close()
        self.assertEqual(read_shard_states(self.store, 1)[0]['next_sequence'], 0)

    def test_invalid_dimensions_capacity_and_seed_fail(self):
        for changes in ({'history_tokens': 0}, {'future_tokens': 0}, {'history_tokens': 11},
                        {'samples_per_block': 0}, {'batch_size': 0}, {'max_bytes': 1},
                        {'seed': -1}, {'shards': 0}, {'shard': 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.produce(**changes)
        self.produce()
        with self.assertRaises(ValueError):
            SampleSampler(self.store, seed=-1)

    def test_nested_cli_generates_and_reports_published_samples(self):
        root = self.store.parent
        streams = root / 'data' / 'token_streams'
        streams.mkdir(parents=True)
        source = root / 'wiki' / 'tiny'
        source.mkdir(parents=True)
        pq.write_table(pa.table({'text': ['1 2 3 4 5 6 7 8 9 10'] * 3}), source / 'train-00000.parquet')
        build = subprocess.run(
            [sys.executable, '-m', 'data_pipline', 'build-token-stream', '--dataset-dir', str(source.parent),
             '--config', 'tiny', '--tokenizer-path', str(self.model_path), '--output', str(streams / 'tiny.npy')],
            cwd=root, text=True, capture_output=True,
        )
        self.assertEqual(build.returncode, 0, build.stderr)
        self.assertIn('tokenizer_fingerprint', json.loads((streams / 'tiny.json').read_text()))
        command = [sys.executable, '-m', 'data_pipline', 'sample-store']
        common = ['--store-dir', str(self.store), '--stream', 'tiny', '--model-path', str(self.model_path),
                  '--history-tokens', '3', '--future-tokens', '2']
        result = subprocess.run(command + ['produce', *common, '--device', 'cpu', '--max-blocks', '1',
                                          '--samples-per-block', '4', '--max-gib', '0.000001'],
                                cwd=root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_sample_matches_teacher(SampleSampler(self.store).sample())
        status = subprocess.run(command + ['status', *common], cwd=root, text=True, capture_output=True)
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertIn('published_samples: 4', status.stdout)

    def test_stream_with_different_token_mapping_is_rejected(self):
        other = self.store.parent / 'other-tokenizer'
        # Same vocabulary size, but 1 and 2 have swapped IDs.
        tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '1': 2, '2': 1, **{str(i): i for i in range(3, 11)}}, unk_token='[UNK]'))
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]').save_pretrained(other)
        stream = self.store.parent / 'mismatched.npy'
        shutil.copy2(self.stream_path, stream)
        stream.with_suffix('.json').write_text(json.dumps({'vocab_size': 11, 'tokenizer_path': str(other)}))
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            self.produce(stream_path=stream)
        self.assertFalse((self.store / 'store.json').exists())

    def test_replacing_model_weights_requires_rebuild(self):
        model = self.store.parent / 'local-model'
        shutil.copytree(self.model_path, model)
        self.produce(model_path=model)
        replacement = GPT2LMHeadModel(GPT2Config(
            vocab_size=11, n_positions=12, n_embd=8, n_layer=1, n_head=2,
            bos_token_id=None, eos_token_id=None,
        ))
        replacement.save_pretrained(model)
        with self.assertRaisesRegex(ValueError, 'config mismatch'):
            self.produce(model_path=model)
        self.assertEqual(read_shard_states(self.store, 1)[0]['next_sequence'], 1)

    def test_shard_ownership_also_prevents_same_process_writers(self):
        self.produce(max_blocks=0)
        with (self.store / 'shard-000.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.assertRaisesRegex(ValueError, 'another producer'):
                self.produce()


if __name__ == '__main__':
    unittest.main()
