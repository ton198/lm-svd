import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from data_pipline.sample_store.producer import produce_blocks
from data_pipline.sample_store.store import slot_bytes
from experiments.exp1_L_M_table.table import centered_logits, select_header_tokens
from experiments.exp4_table_neural_svd.tables import build_exp4_tables
from experiments.exp5_gpt2_mlp_table.features import (
    future_only_logits,
    header_tokens,
    last_hidden,
    pair_targets,
)
from experiments.exp5_gpt2_mlp_table.loss import paired_loss
from experiments.exp5_gpt2_mlp_table.model import Exp5Network, build_model
from utils.common import Sample
from utils.inference import CausalLMInference
from utils.nestedlora import NestedLoRA

VOCAB, HISTORY_TOKENS, FUTURE_TOKENS, TOP_K = 17, 3, 2, 3
HISTORIES = [[row, (row + 1) % VOCAB, (row + 4) % VOCAB] for row in range(6)]
FUTURES = [[(row + 2) % VOCAB, (row + 5) % VOCAB] for row in range(5)]


class Exp5Fixture(unittest.TestCase):
    """A tiny GPT-2, the exp4 tables built from it and a sample store of its pairs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        cls.model_path = cls.root / "model"
        GPT2LMHeadModel(GPT2Config(
            vocab_size=VOCAB, n_positions=8, n_embd=8, n_layer=1, n_head=2,
            bos_token_id=None, eos_token_id=None,
        )).save_pretrained(cls.model_path)
        tokenizer = Tokenizer(WordLevel(
            {"[UNK]": 0, **{str(i): i for i in range(1, VOCAB)}}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="[UNK]", model_max_length=8,
        ).save_pretrained(cls.model_path)
        cls.inference = CausalLMInference(cls.model_path, "cpu")
        # exp4 tables: train 4 histories x 3 futures, val 2 histories x 2 futures.
        samples_path = cls.root / "samples.parquet"
        windows = HISTORIES + FUTURES
        pq.write_table(pa.table({
            "sample_id": pa.array(range(len(windows)), type=pa.int32()),
            "group": ["history"] * len(HISTORIES) + ["future"] * len(FUTURES),
            "token_ids": pa.array(windows, type=pa.list_(pa.int32())),
        }), samples_path)
        cls.inference_dir = cls.root / "inference"
        build_exp4_tables(
            samples_path=samples_path, model_path=cls.model_path, device="cpu",
            train_histories=4, train_futures=3, val_histories=2, val_futures=2,
            top_k=TOP_K, batch_size=2, output_dir=cls.inference_dir, overwrite=False,
        )
        # Sample store: 6 blocks of 4 pairs.
        stream_path = cls.root / "stream.npy"
        np.save(stream_path, (np.arange(200, dtype=np.uint16) * 7) % VOCAB)
        stream_path.with_suffix(".json").write_text(json.dumps({
            "vocab_size": VOCAB, "config": "tiny", "split": "train",
            "tokenizer_path": str(cls.model_path),
        }))
        cls.store_dir = cls.root / "store"
        produce_blocks(
            store_dir=cls.store_dir, model_path=cls.model_path, stream_path=stream_path,
            device="cpu", history_tokens=HISTORY_TOKENS, future_tokens=FUTURE_TOKENS,
            samples_per_block=4,
            max_bytes=8 * slot_bytes(4, HISTORY_TOKENS, FUTURE_TOKENS, VOCAB),
            worker_id=0, seed=0, batch_size=2, max_blocks=6, log_every=100, overwrite=False,
        )


class FeatureTests(Exp5Fixture):
    def test_last_hidden_matches_the_model_and_ignores_chunking(self):
        ids = torch.tensor(HISTORIES)
        expected = self.inference.final_hidden_states(ids)[:, -1]
        for batch_size in (1, 4, 100):
            hidden = last_hidden(self.inference, ids, batch_size)
            self.assertFalse(hidden.requires_grad)
            torch.testing.assert_close(hidden, expected, rtol=0, atol=1e-6)

    def test_headers_match_exp1_selection(self):
        futures = torch.tensor(FUTURES)
        centered = future_only_logits(self.inference, last_hidden(self.inference, futures))
        ids, g = header_tokens(centered, TOP_K)
        expected_ids, expected_g = select_header_tokens(
            self.inference, [Sample(i, row) for i, row in enumerate(FUTURES)], TOP_K)
        self.assertEqual(ids.dtype, torch.int64)
        self.assertEqual(ids.tolist(), expected_ids.tolist())
        np.testing.assert_allclose(g.numpy(), expected_g, rtol=0, atol=1e-5)

    def test_ties_go_to_the_lower_token_id(self):
        ids, values = header_tokens(torch.tensor([[1.0, 3.0, 3.0, 0.0, 3.0]]), 2)
        self.assertEqual(ids.tolist(), [[1, 2]])
        self.assertEqual(values.tolist(), [[3.0, 3.0]])

    def test_top_k_must_fit_the_vocabulary(self):
        for top_k in (0, 6):
            with self.assertRaises(ValueError):
                header_tokens(torch.zeros(1, 5), top_k)

    def test_pair_targets_are_the_table_entries_minus_g(self):
        h, f = torch.tensor(HISTORIES[:2]), torch.tensor(FUTURES[:2])
        ids, g = header_tokens(
            future_only_logits(self.inference, last_hidden(self.inference, f)), TOP_K)
        raw = torch.stack([
            self.inference.next_token_logits(h[i].tolist() + f[i].tolist()) for i in range(2)])
        target = pair_targets(raw + 5.0, ids, g)  # a per-pair shift must not matter
        for i in range(2):
            entries = centered_logits(raw[i])[ids[i]]
            torch.testing.assert_close(target[i], entries - g[i], rtol=0, atol=1e-5)


class ModelTests(unittest.TestCase):
    def test_shapes_and_column_order(self):
        torch.manual_seed(0)
        network = Exp5Network(hidden_size=8, rank=4, top_k=3, mlp_width=16)
        left, right = network(torch.randn(5, 8), torch.randn(2, 8))
        self.assertEqual(left.shape, (5, 4))
        self.assertEqual(right.shape, (2 * 3, 4))
        # Row f * top_k + j is slot j of future f: a future's block depends only on it.
        hidden = torch.randn(2, 8)
        both = network.right_features(hidden)
        torch.testing.assert_close(both[3:], network.right_features(hidden[1:]))
        # Slot j of one future is the j-th row of the final layer's (top_k, rank) output.
        raw = network.right[-1](network.right[:-1](hidden[:1])).reshape(3, 4)
        torch.testing.assert_close(both[:3], raw)

    def test_layers_and_parameter_count(self):
        network = Exp5Network(hidden_size=8, rank=4, top_k=3, mlp_width=16)
        for side, out in ((network.left, 4), (network.right, 12)):
            self.assertEqual(
                [type(layer).__name__ for layer in side],
                ["LayerNorm", "Linear", "GELU", "Linear"],
            )
            self.assertEqual((side[1].in_features, side[1].out_features), (8, 16))
            self.assertEqual((side[3].in_features, side[3].out_features), (16, out))
            self.assertFalse(side[0].elementwise_affine)

    def test_build_model_wraps_the_network_with_joint_nesting(self):
        model = build_model(hidden_size=8, rank=4, top_k=3, mlp_width=16)
        self.assertIsInstance(model, NestedLoRA)
        self.assertFalse(model.sequential)
        self.assertEqual(model.neigs, 4)


class PairedLossTests(unittest.TestCase):
    def reference(self, model, left, right, target):
        batch, top_k = target.shape
        slots = right.reshape(batch, top_k, -1)
        operator = -2 * (
            target[:, :, None] * left[:, None, :] * slots * model.vector_mask
        ).sum(-1).mean()
        metric = (
            model.matrix_mask * (left.T @ left / batch) * (right.T @ right / (batch * top_k))
        ).sum()
        return operator + metric

    def check(self, batch, top_k, rank):
        torch.manual_seed(0)
        model = build_model(hidden_size=8, rank=rank, top_k=top_k, mlp_width=16).double()
        left = torch.randn(batch, rank, dtype=torch.float64, requires_grad=True)
        right = torch.randn(batch * top_k, rank, dtype=torch.float64, requires_grad=True)
        target = torch.randn(batch, top_k, dtype=torch.float64)
        loss = paired_loss(model, left, right, target)
        expected = self.reference(model, left, right, target)
        torch.testing.assert_close(loss, expected)
        for got, want in zip(
            torch.autograd.grad(loss, (left, right)),
            torch.autograd.grad(expected, (left, right)),
        ):
            torch.testing.assert_close(got, want)
            self.assertTrue(torch.isfinite(got).all())

    def test_value_and_gradients_match_the_formula(self):
        self.check(batch=5, top_k=3, rank=4)

    def test_single_pair_batch_is_finite(self):
        self.check(batch=1, top_k=3, rank=4)

    def test_mismatched_shapes_are_rejected(self):
        model = build_model(hidden_size=8, rank=4, top_k=3, mlp_width=16)
        with self.assertRaises(ValueError):
            paired_loss(model, torch.zeros(2, 4), torch.zeros(5, 4), torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            paired_loss(model, torch.zeros(3, 4), torch.zeros(6, 4), torch.zeros(2, 3))


if __name__ == "__main__":
    unittest.main()
