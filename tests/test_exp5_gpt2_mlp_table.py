import contextlib
import io
import json
import shutil
import tempfile
import unittest
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

from data_pipline.sample_store import SampleSampler
from data_pipline.sample_store.producer import produce_blocks
from data_pipline.sample_store.store import slot_bytes
from experiments.exp1_L_M_table.table import centered_logits, select_header_tokens
from experiments.exp4_table_neural_svd.tables import build_exp4_tables
from experiments.exp4_table_neural_svd.train import compare_with_exact_svd
from experiments.exp5_gpt2_mlp_table.cli import main
from experiments.exp5_gpt2_mlp_table.features import (
    future_only_logits,
    header_tokens,
    last_hidden,
    pair_targets,
)
from experiments.exp5_gpt2_mlp_table.loss import paired_loss
from experiments.exp5_gpt2_mlp_table.model import (
    Exp5FinetuneNetwork,
    Exp5Network,
    build_finetune_model,
    build_model,
    header_attention,
    trainable_encoder,
)
from experiments.exp5_gpt2_mlp_table.pairs import StreamPairs
from experiments.exp5_gpt2_mlp_table.tables import eval_features, load_eval_table
from experiments.exp5_gpt2_mlp_table.train import (
    check_store,
    draw_pairs,
    lr_factor,
    sampler_seed,
    train_exp5,
)
from experiments.exp7_independent_subnets.center import (
    CenterPool,
    build_center_pool,
    table_baseline,
)
from utils.common import Sample
from utils.inference import CausalLMInference
from utils.nestedlora import NestedLoRA

VOCAB, HISTORY_TOKENS, FUTURE_TOKENS, TOP_K = 17, 3, 2, 3
# The token stream is 7 * i mod VOCAB, so its windows are arithmetic runs with step 7.
# The table rows and columns are such windows too: the tables are sub-grids of the
# 17 x 17 grid of windows that the training pairs and the store's pairs come from.
STREAM_STEP = 7
HISTORIES = [[(row + STREAM_STEP * i) % VOCAB for i in range(HISTORY_TOKENS)] for row in range(6)]
FUTURES = [[(row + 2 + STREAM_STEP * i) % VOCAB for i in range(FUTURE_TOKENS)] for row in range(5)]


class Exp5Fixture(unittest.TestCase):
    """A tiny GPT-2, the exp4 tables built from it, a token stream and a sample store.

    exp5 and exp6 train on the stream; the store serves exp7's tests.
    """

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
        with contextlib.redirect_stdout(io.StringIO()):  # the builder's progress lines
            build_exp4_tables(
                samples_path=samples_path, model_path=cls.model_path, device="cpu",
                train_histories=4, train_futures=3, val_histories=2, val_futures=2,
                top_k=TOP_K, batch_size=2, output_dir=cls.inference_dir, overwrite=False,
            )
        # Sample store: 100 blocks of 16 pairs, several pairs per cell of the window grid.
        # With few pairs a network can fit which history came with which future instead.
        stream_path = cls.stream_path = cls.root / "stream.npy"
        np.save(stream_path, (np.arange(200, dtype=np.uint16) * STREAM_STEP) % VOCAB)
        stream_path.with_suffix(".json").write_text(json.dumps({
            "vocab_size": VOCAB, "config": "tiny", "split": "train",
            "tokenizer_path": str(cls.model_path),
        }))
        cls.store_dir = cls.root / "store"
        with contextlib.redirect_stdout(io.StringIO()):
            produce_blocks(
                store_dir=cls.store_dir, model_path=cls.model_path, stream_path=stream_path,
                device="cpu", history_tokens=HISTORY_TOKENS, future_tokens=FUTURE_TOKENS,
                samples_per_block=16,
                max_bytes=100 * slot_bytes(16, HISTORY_TOKENS, FUTURE_TOKENS, VOCAB),
                worker_id=0, seed=0, batch_size=16, max_blocks=100, log_every=1000,
                overwrite=False,
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


class TrainableEncoderTests(Exp5Fixture):
    def test_copy_is_trainable_and_leaves_the_model_alone(self):
        encoder = trainable_encoder(self.inference)
        original = self.inference.model.base_model
        self.assertIsNot(encoder, original)
        for (name, copied), (_, kept) in zip(
            encoder.named_parameters(), original.named_parameters()
        ):
            self.assertTrue(copied.requires_grad, name)
            self.assertIsNot(copied, kept)
            torch.testing.assert_close(copied, kept)
        before = [parameter.clone() for parameter in original.parameters()]
        with torch.no_grad():
            for parameter in encoder.parameters():
                parameter.add_(1.0)
        for parameter, kept in zip(original.parameters(), before):
            torch.testing.assert_close(parameter, kept)

    def test_dropout_is_off_in_training_mode(self):
        encoder = trainable_encoder(self.inference).train()
        ids = torch.tensor(FUTURES)
        first = encoder(input_ids=ids).last_hidden_state
        torch.testing.assert_close(encoder(input_ids=ids).last_hidden_state, first)

    def test_right_side_reads_token_ids_through_the_encoder(self):
        torch.manual_seed(0)
        encoder = trainable_encoder(self.inference)
        network = Exp5Network(
            hidden_size=8, rank=4, top_k=TOP_K, mlp_width=16, right_encoder=encoder)
        ids = torch.tensor(FUTURES)
        right = network.right_features(ids)
        self.assertEqual(right.shape, (len(FUTURES) * TOP_K, 4))
        # Before any training the encoder is the model, so this is the frozen path.
        hidden = last_hidden(self.inference, ids)
        torch.testing.assert_close(right, network.right(hidden).reshape(-1, 4), atol=1e-6, rtol=0)
        right.sum().backward()
        self.assertTrue(all(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in encoder.wte.parameters()))
        left, both = network(torch.randn(2, 8), ids)
        self.assertEqual((left.shape, both.shape), ((2, 4), right.shape))

    def test_a_network_without_an_encoder_has_no_encoder_weights(self):
        network = Exp5Network(hidden_size=8, rank=4, top_k=TOP_K, mlp_width=16)
        self.assertIsNone(network.right_encoder)
        self.assertTrue(all(key.startswith(("left.", "right.")) for key in network.state_dict()))


class FinetuneNetworkTests(Exp5Fixture):
    HEADER = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 16]])

    def network(self):
        torch.manual_seed(0)
        return Exp5FinetuneNetwork(
            trainable_encoder(self.inference), trainable_encoder(self.inference),
            hidden_size=8, rank=4)

    def test_attention_lets_each_header_token_see_the_future_and_itself(self):
        mask, position = header_attention(future_tokens=2, top_k=3, batch=2)
        self.assertEqual(mask.shape, (2, 1, 5, 5))
        self.assertEqual(position.tolist(), [[0, 1, 2, 2, 2]] * 2)
        allowed = (mask[0, 0] == 0).int().tolist()
        self.assertEqual(allowed, [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
        ])

    def test_batched_header_tokens_match_one_forward_per_column(self):
        network = self.network()
        futures = torch.tensor(FUTURES)
        with torch.no_grad():
            right = network.right_features(futures, self.HEADER)
        self.assertEqual(right.shape, (len(FUTURES) * TOP_K, 4))
        # Row f * top_k + j is f followed by its j-th header token.
        for f in range(len(FUTURES)):
            for j in range(TOP_K):
                ids = torch.cat([futures[f], self.HEADER[f, j:j + 1]])[None]
                with torch.no_grad():
                    hidden = network.right_encoder(input_ids=ids).last_hidden_state[:, -1]
                    expected = network.right_head(hidden)[0]
                torch.testing.assert_close(right[f * TOP_K + j], expected, atol=1e-5, rtol=0)

    def test_left_side_reads_history_ids_through_its_own_encoder(self):
        network = self.network()
        self.assertIsNot(network.left_encoder, network.right_encoder)
        ids = torch.tensor(HISTORIES)
        left = network.left_features(ids)
        self.assertEqual(left.shape, (len(HISTORIES), 4))
        # Before any training the encoder is the model.
        hidden = last_hidden(self.inference, ids)
        torch.testing.assert_close(left, network.left_head(hidden), atol=1e-6, rtol=0)
        left_out, right_out = network(ids, torch.tensor(FUTURES), self.HEADER)
        (left_out.sum() + right_out.sum()).backward()
        for encoder in (network.left_encoder, network.right_encoder):
            self.assertTrue(all(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in encoder.wte.parameters()))

    def test_heads_are_linear_on_normalised_hidden_states(self):
        network = self.network()
        for head in (network.left_head, network.right_head):
            self.assertEqual([type(layer).__name__ for layer in head], ["LayerNorm", "Linear"])
            self.assertFalse(head[0].elementwise_affine)
            self.assertEqual((head[1].in_features, head[1].out_features), (8, 4))

    def test_build_finetune_model_copies_the_transformer_twice(self):
        model = build_finetune_model(self.inference, rank=4)
        self.assertIsInstance(model, NestedLoRA)
        self.assertFalse(model.sequential)
        self.assertEqual(model.neigs, 4)
        network = model.model
        original = self.inference.model.base_model
        for encoder in (network.left_encoder, network.right_encoder):
            self.assertIsNot(encoder, original)
            self.assertTrue(all(parameter.requires_grad for parameter in encoder.parameters()))


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


class EvalTableTests(Exp5Fixture):
    def stored(self, name):
        directory = self.inference_dir / name
        return np.load(directory / "matrix.npy"), dict(np.load(directory / "tokens.npz"))

    def expected_g(self, tokens):
        centered = future_only_logits(
            self.inference, last_hidden(self.inference, torch.from_numpy(tokens["futures"])))
        return centered.gather(1, torch.from_numpy(tokens["header_token_ids"]).long())

    def test_matrix_loses_g_and_the_svd_belongs_to_the_result(self):
        for name in ("train", "val"):
            with self.subTest(table=name):
                matrix, tokens = self.stored(name)
                loaded = load_eval_table(self.inference_dir / name, self.inference, "cpu")
                g = self.expected_g(tokens).reshape(-1).numpy()
                np.testing.assert_allclose(loaded.table.matrix.numpy(), matrix - g, atol=1e-5)
                table = loaded.table
                rebuilt = (
                    table.left_vectors * table.singular_values[: table.left_vectors.shape[1]]
                ) @ table.right_vectors.T
                np.testing.assert_allclose(rebuilt.numpy(), matrix - g, atol=1e-4)
                residual = np.square(matrix - g).sum()
                self.assertAlmostEqual(
                    loaded.residual_energy, residual / np.square(matrix).sum(), places=5)
                self.assertAlmostEqual(
                    loaded.residual_top_mode,
                    table.singular_values[0].item() ** 2 / residual, places=5)
                self.assertEqual(table.histories.tolist(), tokens["histories"].tolist())
                self.assertEqual(table.columns.shape, (matrix.shape[1], FUTURE_TOKENS + 1))

    def test_hidden_states_are_cached_for_every_history_and_future(self):
        _, tokens = self.stored("train")
        loaded = load_eval_table(
            self.inference_dir / "train", self.inference, "cpu", batch_size=2)
        for cached, ids in ((loaded.history_hidden, tokens["histories"]),
                            (loaded.future_hidden, tokens["futures"])):
            expected = self.inference.final_hidden_states(torch.from_numpy(ids))[:, -1]
            torch.testing.assert_close(cached, expected, rtol=0, atol=1e-6)

    def copy_table(self, name):
        target = self.root / f"copy-{self.id().rsplit('.', 1)[-1]}"
        shutil.copytree(self.inference_dir / name, target)
        return target

    def test_reordered_headers_load_and_g_follows_the_stored_order(self):
        directory = self.copy_table("train")
        matrix = np.load(directory / "matrix.npy")
        tokens = dict(np.load(directory / "tokens.npz"))
        # Swap slots 0 and 1 of the first future, in the header ids and the matrix columns.
        tokens["header_token_ids"][0, [0, 1]] = tokens["header_token_ids"][0, [1, 0]]
        matrix[:, [0, 1]] = matrix[:, [1, 0]]
        np.save(directory / "matrix.npy", matrix)
        np.savez(directory / "tokens.npz", **tokens)
        loaded = load_eval_table(directory, self.inference, "cpu")
        g = self.expected_g(tokens).reshape(-1).numpy()
        np.testing.assert_allclose(loaded.table.matrix.numpy(), matrix - g, atol=1e-5)

    def test_a_header_outside_the_top_k_is_rejected(self):
        directory = self.copy_table("train")
        tokens = dict(np.load(directory / "tokens.npz"))
        centered = future_only_logits(
            self.inference, last_hidden(self.inference, torch.from_numpy(tokens["futures"])))
        tokens["header_token_ids"][0, -1] = centered[0].argmin().item()
        np.savez(directory / "tokens.npz", **tokens)
        with self.assertRaisesRegex(ValueError, "1 future"):
            load_eval_table(directory, self.inference, "cpu")

    def test_features_cover_every_row_and_column_in_table_order(self):
        loaded = load_eval_table(self.inference_dir / "train", self.inference, "cpu")
        torch.manual_seed(0)
        model = build_model(hidden_size=8, rank=4, top_k=TOP_K, mlp_width=16)
        left, right = eval_features(model, loaded, chunk=2)
        self.assertEqual((left.dtype, right.dtype), (torch.float64, torch.float64))
        self.assertEqual(left.shape, (4, 4))
        self.assertEqual(right.shape, (3 * TOP_K, 4))
        with torch.no_grad():
            expected = model.model.right_features(loaded.future_hidden)
        torch.testing.assert_close(right.float(), expected, atol=1e-6, rtol=0)
        self.assertFalse(model.training)
        self.assertEqual(loaded.future_ids.tolist(), self.stored("train")[1]["futures"].tolist())
        # exp4's comparison accepts the result unchanged.
        comparison = compare_with_exact_svd(left, right, loaded.table)
        self.assertEqual(comparison["ranks"], [1, 2, 4])

    def test_features_of_an_unfrozen_right_side_come_from_its_encoder(self):
        loaded = load_eval_table(self.inference_dir / "train", self.inference, "cpu")
        torch.manual_seed(0)
        encoder = trainable_encoder(self.inference)
        with torch.no_grad():  # move the encoder away from the frozen model
            encoder.wte.weight.add_(0.1 * torch.randn_like(encoder.wte.weight))
        model = build_model(
            hidden_size=8, rank=4, top_k=TOP_K, mlp_width=16, right_encoder=encoder)
        left, right = eval_features(model, loaded, chunk=2)
        with torch.no_grad():
            expected = model.model.right_features(loaded.future_ids)
            frozen = model.model.right(loaded.future_hidden).reshape(-1, 4)
        torch.testing.assert_close(right.float(), expected, atol=1e-6, rtol=0)
        self.assertGreater((expected - frozen).abs().max().item(), 1e-4)
        self.assertEqual(left.shape, (4, 4))


    def test_features_of_a_finetuned_network_read_histories_and_columns(self):
        loaded = load_eval_table(self.inference_dir / "train", self.inference, "cpu")
        torch.manual_seed(0)
        model = build_finetune_model(self.inference, rank=4)
        with torch.no_grad():  # move both encoders away from the frozen model
            for encoder in (model.model.left_encoder, model.model.right_encoder):
                encoder.wte.weight.add_(0.1 * torch.randn_like(encoder.wte.weight))
        left, right = eval_features(model, loaded, chunk=2)
        self.assertEqual((left.dtype, right.dtype), (torch.float64, torch.float64))
        futures = len(loaded.future_ids)
        columns = loaded.table.columns.reshape(futures, TOP_K, -1)
        self.assertTrue(torch.equal(columns[:, 0, :-1], loaded.future_ids))
        with torch.no_grad():
            expected_left = model.model.left_features(loaded.table.histories)
            expected_right = model.model.right_features(loaded.future_ids, columns[..., -1])
        torch.testing.assert_close(left.float(), expected_left, atol=1e-6, rtol=0)
        torch.testing.assert_close(right.float(), expected_right, atol=1e-6, rtol=0)
        self.assertFalse(model.training)
        self.assertEqual(compare_with_exact_svd(left, right, loaded.table)["ranks"], [1, 2, 4])


class TrainingTests(Exp5Fixture):
    def train(self, output_dir, **changes):
        options = dict(
            inference_dir=self.inference_dir, stream_path=self.stream_path,
            model_path=self.model_path, output_dir=output_dir, device="cpu", neigs=2,
            mlp_width=16, batch_size=8, steps=4, eval_every=2, lr=1e-2, seed=0,
            overwrite=False,
        )
        options.update(changes)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            summary = train_exp5(**options)
        self.log = output.getvalue()
        return summary

    def records(self, output_dir):
        lines = (output_dir / "eval_history.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]

    def test_writes_outputs_and_compares_both_tables(self):
        output = self.root / "run-outputs"
        summary = self.train(output)
        self.assertEqual(summary, {"output_dir": str(output), "step": 4, "evaluations": 2})
        records = self.records(output)
        self.assertEqual([record["step"] for record in records], [2, 4])
        for record in records:
            self.assertEqual(set(record), {
                "step", "train_loss", "kernel_energy", "seconds_per_step", "train", "val"})
            self.assertEqual(record["train"]["ranks"], [1, 2])
            self.assertTrue(np.isfinite(record["train_loss"]))
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["top_k"], TOP_K)
        self.assertEqual(metadata["mlp_width"], 16)
        self.assertEqual(metadata["table_shapes"], {"train": [4, 9], "val": [2, 6]})
        self.assertEqual(
            set(metadata["tables"]["val"]), {"residual_energy", "residual_top_mode"})
        self.assertIn("0.1%", metadata["isolation"])
        self.assertIn("in the step", metadata["sampling"])
        self.assertNotIn("store", metadata)
        self.assertNotIn("store_dir", metadata)
        self.assertEqual(metadata["stream_path"], str(self.stream_path))
        self.assertEqual(
            (metadata["stream"]["vocab_size"], metadata["stream"]["tokens"]), (VOCAB, 200))
        self.assertNotIn("lr_schedule", metadata)

    def test_only_the_mlps_are_trained_and_saved(self):
        output = self.root / "run-frozen"
        self.train(output)
        checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
        self.assertTrue(all(
            key.startswith(("model.left.", "model.right."))
            for key in checkpoint["model_state_dict"]))
        metadata = json.loads((output / "metadata.json").read_text())
        network = Exp5Network(hidden_size=8, rank=2, top_k=TOP_K, mlp_width=16)
        self.assertEqual(
            metadata["trainable_parameters"], sum(p.numel() for p in network.parameters()))

    def test_training_recovers_the_leading_mode_of_both_tables(self):
        output = self.root / "run-learns"
        self.train(output, steps=300, eval_every=100, batch_size=16)
        records = self.records(output)
        # An untrained network scores at or below zero; the exact value is 1.
        for name in ("train", "val"):
            self.assertGreater(records[-1][name]["captured_energy"][0], 0.5)
        self.assertLess(records[-1]["train_loss"], records[0]["train_loss"])
        # A fit of the table itself ends near -E[K^2]; far below it means the
        # network fits which history came with which future.
        self.assertGreater(records[-1]["train_loss"], -1.5 * records[-1]["kernel_energy"])

    def test_best_checkpoint_follows_the_validation_energy(self):
        output = self.root / "run-best"
        self.train(output, steps=300, eval_every=100, batch_size=16)
        records = self.records(output)
        best_record = max(records, key=lambda record: record["val"]["captured_energy"][-1])
        best = torch.load(output / "checkpoint-best.pt", weights_only=True)
        self.assertEqual(best["step"], best_record["step"])
        self.assertAlmostEqual(best["val_energy"], best_record["val"]["captured_energy"][-1])
        self.assertEqual(set(best), {"model_state_dict", "step", "val_energy", "config"})
        # A continued run keeps the earlier best unless it does better.
        self.train(output, steps=400, eval_every=100, batch_size=16)
        records = self.records(output)
        best_record = max(records, key=lambda record: record["val"]["captured_energy"][-1])
        best = torch.load(output / "checkpoint-best.pt", weights_only=True)
        self.assertEqual(best["step"], best_record["step"])

    def test_training_reads_no_sample_store(self):
        with patch("experiments.exp5_gpt2_mlp_table.train.SampleSampler",
                   side_effect=AssertionError("sample store")), \
                patch("experiments.exp5_gpt2_mlp_table.train.read_store_config",
                      side_effect=AssertionError("sample store")):
            self.assertEqual(self.train(self.root / "run-no-store")["step"], 4)

    def test_cosine_schedule_lowers_the_rate_and_belongs_to_the_settings(self):
        output = self.root / "run-cosine"
        self.train(output, steps=8, eval_every=2, lr_schedule="cosine")
        self.assertIn("cosine learning rate", self.log)
        records = self.records(output)
        # An evaluation records the rate of the step before it.
        self.assertEqual(
            [record["lr"] for record in records],
            [1e-2 * lr_factor("cosine", 8)(step - 1) for step in (2, 4, 6, 8)])
        checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
        self.assertEqual(
            (checkpoint["config"]["lr_schedule"], checkpoint["config"]["steps"]), ("cosine", 8))
        # Another length gives other rates: it is another run.
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.train(output, steps=12, eval_every=2, lr_schedule="cosine")
        with self.assertRaisesRegex(ValueError, "lr_schedule"):
            self.train(self.root / "run-linear", lr_schedule="linear")

    def test_a_continued_cosine_run_follows_the_uninterrupted_rates(self):
        whole, parts = self.root / "run-cosine-whole", self.root / "run-cosine-parts"
        self.train(whole, steps=8, eval_every=4, lr_schedule="cosine")
        original = torch.optim.Adam.step
        calls = {"count": 0}

        def step(optimizer, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 6:
                raise RuntimeError("interrupted")
            return original(optimizer, *args, **kwargs)

        # The settings hold the length of the run, so it stops by an error, not by --steps.
        with patch.object(torch.optim.Adam, "step", step), \
                self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.train(parts, steps=8, eval_every=4, lr_schedule="cosine")
        self.train(parts, steps=8, eval_every=4, lr_schedule="cosine")
        self.assertIn("Resuming from step 4", self.log)
        self.assertEqual(
            [record["lr"] for record in self.records(parts)],
            [record["lr"] for record in self.records(whole)])

    def test_unfrozen_right_trains_the_encoder_copy(self):
        output = self.root / "run-unfrozen"
        summary = self.train(output, steps=300, eval_every=100, batch_size=16,
                             unfreeze_right=True, encoder_lr=1e-3)
        self.assertEqual(summary["evaluations"], 3)
        checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
        state = checkpoint["model_state_dict"]
        self.assertTrue(any(key.startswith("model.right_encoder.") for key in state))
        original = self.inference.model.base_model.state_dict()
        moved = max(
            (state[f"model.right_encoder.{key}"] - value).abs().max().item()
            for key, value in original.items() if value.is_floating_point())
        self.assertGreater(moved, 1e-4)
        self.assertEqual(checkpoint["config"]["unfreeze_right"], True)
        self.assertEqual(checkpoint["config"]["encoder_lr"], 1e-3)
        metadata = json.loads((output / "metadata.json").read_text())
        mlps = Exp5Network(hidden_size=8, rank=2, top_k=TOP_K, mlp_width=16)
        self.assertEqual(
            metadata["trainable_parameters"],
            sum(p.numel() for p in mlps.parameters())
            + sum(p.numel() for p in self.inference.model.base_model.parameters()))
        records = self.records(output)
        for name in ("train", "val"):
            self.assertGreater(records[-1][name]["captured_energy"][0], 0.5)
        # The settings are part of the run: a frozen run cannot continue it.
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.train(output, steps=400, eval_every=100, batch_size=16)
        self.assertEqual(
            self.train(output, steps=400, eval_every=100, batch_size=16,
                       unfreeze_right=True, encoder_lr=1e-3)["step"], 400)

    def test_finetune_trains_both_encoder_copies(self):
        output = self.root / "run-finetune"
        summary = self.train(output, steps=300, eval_every=100, batch_size=16,
                             finetune=True, encoder_lr=1e-3)
        self.assertEqual(summary["evaluations"], 3)
        checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
        state = checkpoint["model_state_dict"]
        original = self.inference.model.base_model.state_dict()
        for side in ("left_encoder", "right_encoder"):
            moved = max(
                (state[f"model.{side}.{key}"] - value).abs().max().item()
                for key, value in original.items() if value.is_floating_point())
            self.assertGreater(moved, 1e-4, side)
        self.assertEqual(checkpoint["config"]["finetune"], True)
        self.assertEqual(checkpoint["config"]["encoder_lr"], 1e-3)
        self.assertNotIn("mlp_width", checkpoint["config"])
        metadata = json.loads((output / "metadata.json").read_text())
        heads = 2 * (8 * 2 + 2)
        self.assertEqual(
            metadata["trainable_parameters"],
            heads + 2 * sum(p.numel() for p in self.inference.model.base_model.parameters()))
        self.assertIn("each header token", metadata["right_encoder"])
        records = self.records(output)
        for name in ("train", "val"):
            self.assertGreater(records[-1][name]["captured_energy"][0], 0.5)
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.train(output, steps=400, eval_every=100, batch_size=16,
                       unfreeze_right=True, encoder_lr=1e-3)
        self.assertEqual(
            self.train(output, steps=400, eval_every=100, batch_size=16,
                       finetune=True, encoder_lr=1e-3)["step"], 400)

    def test_finetune_and_unfreeze_right_are_exclusive(self):
        with self.assertRaisesRegex(ValueError, "finetune"):
            self.train(self.root / "run-both", finetune=True, unfreeze_right=True)

    def test_a_frozen_run_keeps_its_earlier_settings(self):
        output = self.root / "run-frozen-config"
        self.train(output)
        config = torch.load(output / "checkpoint.pt", weights_only=True)["config"]
        self.assertNotIn("unfreeze_right", config)
        self.assertNotIn("encoder_lr", config)

    def test_resume_and_step_edge_cases(self):
        output = self.root / "run-resume"
        self.train(output, steps=2)
        self.assertEqual(self.train(output, steps=4)["evaluations"], 2)  # continues from 2
        self.assertEqual(
            self.train(output, steps=4), {"output_dir": str(output), "step": 4, "evaluations": 2})
        self.assertEqual(self.train(output, steps=3)["step"], 4)  # not above the checkpoint
        self.assertEqual([record["step"] for record in self.records(output)], [2, 4])
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.train(output, steps=6, lr=1e-3)
        self.assertEqual(self.train(output, steps=2, lr=1e-3, overwrite=True)["step"], 2)
        late = self.root / "run-late-eval"
        self.assertEqual(self.train(late, steps=3, eval_every=10)["evaluations"], 1)

    def test_rejects_missing_tables_and_a_negative_seed(self):
        with self.assertRaisesRegex(ValueError, "exp4 inference"):
            self.train(self.root / "run-none", inference_dir=self.root / "absent")
        with self.assertRaises(ValueError):
            self.train(self.root / "run-seed", seed=-1)

    def store_config(self):
        return json.loads((self.store_dir / "store.json").read_text())

    def check(self, config):
        check_store(
            config, self.model_path, history_tokens=HISTORY_TOKENS,
            future_tokens=FUTURE_TOKENS, vocab_size=VOCAB)

    def test_rejects_a_store_with_other_window_lengths_or_vocabulary(self):
        self.check(self.store_config())
        for key in ("history_tokens", "future_tokens", "vocab_size"):
            with self.subTest(key=key):
                config = self.store_config()
                config[key] += 1
                with self.assertRaisesRegex(ValueError, key):
                    self.check(config)

    def test_rejects_a_store_from_other_model_files(self):
        config = self.store_config()
        name = next(iter(config["model_files"]))
        config["model_files"][name]["size"] += 1
        with self.assertRaisesRegex(ValueError, "model"):
            self.check(config)
        # A different modification time alone is accepted.
        config = self.store_config()
        config["model_files"][name]["mtime_ns"] += 1
        self.check(config)

    def test_a_missing_stream_or_another_vocabulary_stops_training(self):
        with self.assertRaisesRegex(ValueError, "no token stream"):
            self.train(self.root / "run-no-stream", stream_path=self.root / "absent.npy")
        other = self.root / "stream-other-vocabulary.npy"
        shutil.copy(self.stream_path, other)
        metadata = json.loads(self.stream_path.with_suffix(".json").read_text())
        other.with_suffix(".json").write_text(json.dumps({**metadata, "vocab_size": VOCAB + 1}))
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.train(self.root / "run-other-vocabulary", stream_path=other)

    def test_draw_pairs_stacks_store_samples(self):
        h, f, logits = draw_pairs(SampleSampler(self.store_dir, seed=3), 5)
        self.assertEqual(
            (h.shape, f.shape, logits.shape),
            ((5, HISTORY_TOKENS), (5, FUTURE_TOKENS), (5, VOCAB)))
        expected = self.inference.next_token_logits(h[0].tolist() + f[0].tolist())
        torch.testing.assert_close(logits[0], expected, rtol=0, atol=1e-6)
        self.assertNotEqual(sampler_seed(0, 0), sampler_seed(0, 2))

    def stream_pairs(self, seed):
        return StreamPairs(
            self.inference, self.stream_path, history_tokens=HISTORY_TOKENS,
            future_tokens=FUTURE_TOKENS, top_k=TOP_K, seed=seed, encode_batch=4)

    def test_stream_pairs_are_stream_windows_with_the_table_entries_minus_g(self):
        histories, futures, header_ids, target = self.stream_pairs(3).draw(6)
        self.assertEqual(
            (histories.shape, futures.shape, header_ids.shape, target.shape),
            ((6, HISTORY_TOKENS), (6, FUTURE_TOKENS), (6, TOP_K), (6, TOP_K)))
        for windows in (histories, futures):  # arithmetic runs of the stream
            self.assertTrue(((windows[:, 1:] - windows[:, :-1]) % VOCAB == STREAM_STEP).all())
        expected_ids, expected_g = select_header_tokens(
            self.inference, [Sample(i, row) for i, row in enumerate(futures.tolist())], TOP_K)
        self.assertEqual(header_ids.tolist(), expected_ids.tolist())
        for row in range(6):
            logits = self.inference.next_token_logits(
                histories[row].tolist() + futures[row].tolist())
            entries = (logits - logits.mean())[header_ids[row]]
            torch.testing.assert_close(
                target[row], entries - torch.from_numpy(expected_g[row]).float(),
                rtol=0, atol=1e-5)

    def test_stream_pairs_draw_histories_and_futures_independently(self):
        histories, futures, _, _ = self.stream_pairs(0).draw(400)
        cells = {(h[0].item(), f[0].item()) for h, f in zip(histories, futures)}
        self.assertGreater(len(cells), 200)  # of the 17 x 17 grid, not one future per history
        again = self.stream_pairs(0).draw(400)
        self.assertEqual(again[0].tolist(), histories.tolist())
        self.assertNotEqual(self.stream_pairs(1).draw(400)[0].tolist(), histories.tolist())

    def test_lr_factor(self):
        self.assertIsNone(lr_factor("constant", 10))
        cosine = lr_factor("cosine", 10)
        self.assertEqual(cosine(0), 1.0)
        self.assertAlmostEqual(cosine(5), 0.5)
        self.assertLess(cosine(9), 0.03)
        self.assertGreater(cosine(9), 0.0)
        self.assertTrue(all(cosine(step + 1) < cosine(step) for step in range(9)))


class CenteredTrainingTests(Exp5Fixture):
    """exp5 on the target of a center pool."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.pool_dir = cls.root / "pool"
        with contextlib.redirect_stdout(io.StringIO()):
            build_center_pool(
                pool_dir=cls.pool_dir, model_path=cls.model_path, stream_path=cls.stream_path,
                device="cpu", histories=3, futures=10, history_tokens=HISTORY_TOKENS,
                future_tokens=FUTURE_TOKENS, seed=0, overwrite=False, chunk=4, save_every=4,
            )

    def train(self, output_dir, **changes):
        options = dict(
            inference_dir=self.inference_dir, stream_path=self.root / "unread.npy",
            model_path=self.model_path, output_dir=output_dir, device="cpu", neigs=2,
            mlp_width=16, batch_size=8, steps=4, eval_every=2, lr=1e-2, seed=0,
            overwrite=False, center_pool=self.pool_dir,
        )
        options.update(changes)
        with contextlib.redirect_stdout(io.StringIO()):
            return train_exp5(**options)

    def test_a_centered_run_trains_on_the_pool_target(self):
        output = self.root / "run-centered"
        self.assertEqual(self.train(output)["step"], 4)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["center_pool"]["histories"], 3)
        self.assertIn("none", metadata["first_mode"])
        self.assertIn("center pool", metadata["value"])
        self.assertNotIn("stream_path", metadata)
        self.assertNotIn("store", metadata)
        # The tables lose the pool's center, not g.
        pool = CenterPool(self.pool_dir)
        centered = load_eval_table(
            self.inference_dir / "val", self.inference, "cpu",
            baseline=table_baseline(self.inference, pool))
        self.assertAlmostEqual(
            metadata["tables"]["val"]["residual_energy"], centered.residual_energy)
        self.assertNotAlmostEqual(
            metadata["tables"]["val"]["residual_energy"],
            load_eval_table(self.inference_dir / "val", self.inference, "cpu").residual_energy)
        # A run on the target minus g cannot continue it.
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.train(output, steps=6, center_pool=None, stream_path=self.stream_path)

    def test_every_kind_of_network_takes_the_centered_pairs(self):
        for name, changes in (
            ("unfreeze", {"unfreeze_right": True}), ("finetune", {"finetune": True}),
        ):
            with self.subTest(name=name):
                output = self.root / f"run-centered-{name}"
                self.assertEqual(self.train(output, **changes)["step"], 4)

    def test_a_missing_pool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "center pool"):
            self.train(self.root / "run-no-pool", center_pool=self.root / "absent-pool")


class CLITests(Exp5Fixture):
    def setUp(self):
        # Place the fixture's tables where the CLI looks for exp4's output.
        self.results_root = self.root / f"results-{self.id().rsplit('.', 1)[-1]}"
        shutil.copytree(
            self.inference_dir,
            self.results_root / "exp4_table_neural_svd" / "tiny__model" / "inference",
        )

    def run_cli(self, *extra):
        main([
            "train", "--results-root", str(self.results_root), "--dataset", "tiny",
            "--model-path", str(self.model_path), "--stream-path", str(self.stream_path),
            "--device", "cpu", "--neigs", "2", "--mlp-width", "16", "--batch-size", "8",
            "--steps", "2", "--eval-every", "2", *extra,
        ])

    def test_train_writes_under_the_exp5_run_directory(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.run_cli()
        run = self.results_root / "exp5_gpt2_mlp_table" / "tiny__model" / "train"
        self.assertTrue((run / "eval_history.jsonl").is_file())
        self.assertIn(f"output_dir: {run}", output.getvalue())
        self.assertIn("step: 2", output.getvalue())

    def test_errors_are_reported_as_usage_errors(self):
        for extra in (["--stream-path", str(self.root / "no-stream.npy")],
                      ["--dataset", "missing"], ["--lr-schedule", "linear"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.run_cli(*extra)
                self.assertEqual(raised.exception.code, 2)

    def test_defaults_match_the_design(self):
        with patch("experiments.exp5_gpt2_mlp_table.train.train_exp5") as train:
            train.return_value = {"output_dir": "x", "step": 0, "evaluations": 0}
            with contextlib.redirect_stdout(io.StringIO()):
                main(["train"])
        options = train.call_args.kwargs
        self.assertEqual(
            {key: options[key] for key in (
                "neigs", "mlp_width", "batch_size", "steps", "eval_every", "lr", "seed",
                "device", "overwrite", "unfreeze_right", "finetune", "encoder_lr",
                "center_pool", "lr_schedule")},
            {"neigs": 128, "mlp_width": 4096, "batch_size": 256, "steps": 100000,
             "eval_every": 1000, "lr": 3e-4, "seed": 42, "device": "cuda",
             "overwrite": False, "unfreeze_right": False, "finetune": False,
             "encoder_lr": 2e-5, "center_pool": None, "lr_schedule": "constant"})
        self.assertEqual(options["model_path"], Path("models/gpt2-small"))
        self.assertEqual(
            options["stream_path"], Path("data/token_streams/wikitext-103-train.npy"))
        self.assertNotIn("store_dir", options)
        self.assertEqual(options["inference_dir"], Path(
            "data/experiment_results/exp4_table_neural_svd/wikitext-large__gpt2-small/inference"))
        self.assertEqual(options["output_dir"], Path(
            "data/experiment_results/exp5_gpt2_mlp_table/wikitext-large__gpt2-small/train"))

    def test_unfreezing_the_right_side_gets_its_own_run_directory(self):
        with patch("experiments.exp5_gpt2_mlp_table.train.train_exp5") as train:
            train.return_value = {"output_dir": "x", "step": 0, "evaluations": 0}
            with contextlib.redirect_stdout(io.StringIO()):
                main(["train", "--unfreeze-right", "--encoder-lr", "1e-5"])
        options = train.call_args.kwargs
        self.assertIs(options["unfreeze_right"], True)
        self.assertEqual(options["encoder_lr"], 1e-5)
        self.assertEqual(options["output_dir"], Path(
            "data/experiment_results/exp5_gpt2_mlp_table/wikitext-large__gpt2-small/"
            "train-unfreeze-right"))


    def test_finetuning_gets_its_own_run_directory(self):
        with patch("experiments.exp5_gpt2_mlp_table.train.train_exp5") as train:
            train.return_value = {"output_dir": "x", "step": 0, "evaluations": 0}
            with contextlib.redirect_stdout(io.StringIO()):
                main(["train", "--finetune"])
        options = train.call_args.kwargs
        self.assertIs(options["finetune"], True)
        self.assertIs(options["unfreeze_right"], False)
        self.assertEqual(options["output_dir"], Path(
            "data/experiment_results/exp5_gpt2_mlp_table/wikitext-large__gpt2-small/"
            "train-finetune"))

    def test_a_center_pool_gets_its_own_run_directory(self):
        pool = self.root / "cli-pool"
        with contextlib.redirect_stdout(io.StringIO()):
            build_center_pool(
                pool_dir=pool, model_path=self.model_path, stream_path=self.stream_path,
                device="cpu", histories=3, futures=10, history_tokens=HISTORY_TOKENS,
                future_tokens=FUTURE_TOKENS, seed=0, overwrite=False, chunk=4, save_every=4,
            )
            self.run_cli("--center-pool", str(pool), "--lr-schedule", "cosine")
        run = self.results_root / "exp5_gpt2_mlp_table" / "tiny__model" / "train-center3"
        metadata = json.loads((run / "metadata.json").read_text())
        self.assertEqual((metadata["lr_schedule"], metadata["steps"]), ("cosine", 2))
        self.assertEqual(metadata["center_pool"]["histories"], 3)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            self.run_cli("--center-pool", str(self.root / "absent-pool"))
        self.assertEqual(raised.exception.code, 2)

    def test_finetune_and_unfreeze_right_cannot_be_combined(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(["train", "--finetune", "--unfreeze-right"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
