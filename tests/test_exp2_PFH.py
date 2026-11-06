import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from experiments.exp2_PFH.cli import main
from experiments.exp2_PFH.feature_logprob import Exp2Inputs, build_exp2_pfh, wikitext_inputs
from utils.inference import CausalLMInference


class Exp2PFHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        cls.model_path = cls.root / 'model'
        GPT2LMHeadModel(GPT2Config(
            vocab_size=11, n_positions=12, n_embd=8, n_layer=1, n_head=2,
            bos_token_id=None, eos_token_id=None,
        )).save_pretrained(cls.model_path)
        tokenizer = Tokenizer(WordLevel({'[UNK]': 0, **{str(i): i for i in range(1, 11)}}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', model_max_length=12).save_pretrained(cls.model_path)
        cls.dataset = cls.root / 'wiki'
        folder = cls.dataset / 'tiny'
        folder.mkdir(parents=True)
        pq.write_table(pa.table({'text': ['1 2 3 4 5', '6 7 8 9 10', '1 2 3 4 5 6 7 8 9 10']}), folder / 'train-00000.parquet')

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)

    def inputs(self, **changes):
        args = dict(dataset_dir=self.dataset, config='tiny', split='train', model_path=self.model_path,
                    history_count=3, future_count=4, history_tokens=3, future_tokens=2, seed=3, pairing='independent')
        args.update(changes)
        return wikitext_inputs(**args)

    def test_direct_wikitext_windows_are_reproducible_and_preserve_source(self):
        inputs = self.inputs()
        self.assertEqual(inputs, self.inputs())
        self.assertEqual(inputs.source['kind'], 'wikitext')
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2
        self.assertEqual(inputs.histories, [tokens[s:s + 3] for s in inputs.source['history_starts']])
        self.assertEqual(inputs.futures, [tokens[s:s + 2] for s in inputs.source['future_starts']])
        self.assertNotEqual(inputs, self.inputs(seed=4))

    def test_adjacent_pairing_uses_continuations_and_valid_start_range(self):
        inputs = self.inputs(pairing='adjacent', history_count=30, future_count=30)
        self.assertEqual(inputs.source['future_starts'], [s + 3 for s in inputs.source['history_starts']])
        self.assertTrue(all(len(h) == 3 for h in inputs.histories))
        self.assertTrue(all(len(f) == 2 for f in inputs.futures))
        with self.assertRaises(ValueError):
            self.inputs(pairing='adjacent')

    def test_matrix_matches_brute_force_pmi(self):
        inputs = self.inputs(history_count=2, future_count=3)
        build_exp2_pfh(inputs=inputs, model_path=self.model_path, device='cpu', batch_size=2,
                       output_dir=self.output, overwrite=False)
        matrix = np.load(self.output / 'pmi.npy')
        inference = CausalLMInference(self.model_path, 'cpu')
        def score(ids):
            return sum(float(torch.log_softmax(inference.next_token_logits(ids[:i]), -1)[ids[i]]) for i in range(1, len(ids)))
        expected = [[score(h + f) - score(h) - score(f) for f in inputs.futures] for h in inputs.histories]
        np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-6)
        metadata = json.loads((self.output / 'metadata.json').read_text())
        self.assertEqual(metadata['source'], inputs.source)

    def test_inference_cli_reads_raw_wikitext_and_selects_run_name(self):
        with redirect_stdout(StringIO()):
            main(['inference', '--dataset-dir', str(self.dataset), '--config', 'tiny',
                  '--model-path', str(self.model_path), '--device', 'cpu',
                  '--history-count', '2', '--future-count', '3', '--history-tokens', '3',
                  '--future-tokens', '2', '--results-root', str(self.output), '--run-name', 'direct-test'])
        path = self.output / 'exp2_PFH' / 'direct-test__model' / 'inference' / 'pmi.npy'
        self.assertEqual(np.load(path).shape, (2, 3))
        for flag in ('--stream', '--sample-store', '--dataset'):
            with self.subTest(flag=flag), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(['inference', flag, 'old-input'])

    def test_invalid_sampling_options_fail(self):
        for change in ({'seed': -1}, {'history_tokens': 0}, {'future_count': 0},
                       {'pairing': 'bad'}, {'history_tokens': 11}, {'history_tokens': 40}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.inputs(**change)

    def test_existing_output_fails_before_model_load(self):
        (self.output / 'metadata.json').touch()
        with self.assertRaisesRegex(FileExistsError, 'overwrite'):
            build_exp2_pfh(inputs=Exp2Inputs([[1, 2]], [[3]], {}), model_path=self.root / 'missing',
                           device='cpu', batch_size=2, output_dir=self.output, overwrite=False)


if __name__ == '__main__':
    unittest.main()
