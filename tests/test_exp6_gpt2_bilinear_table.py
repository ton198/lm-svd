import contextlib
import io
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.exp4_table_neural_svd.train import compare_with_exact_svd
from experiments.exp5_gpt2_mlp_table.features import (
    future_only_logits,
    header_tokens,
    last_hidden,
    pair_targets,
)
from experiments.exp5_gpt2_mlp_table.model import trainable_encoder
from experiments.exp5_gpt2_mlp_table.tables import load_eval_table, token_features
from experiments.exp6_gpt2_bilinear_table.cli import main
from experiments.exp6_gpt2_bilinear_table.model import Exp6Network, build_model
from experiments.exp6_gpt2_bilinear_table.train import train_exp6
from test_exp5_gpt2_mlp_table import FUTURES, HISTORIES, TOP_K, VOCAB, Exp5Fixture
from utils.nestedlora import NestedLoRA

HIDDEN, RANK, HEAD_RANK = 8, 4, 5


class NetworkTests(Exp5Fixture):
    HEADER = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 16]])

    def unembedding(self):
        return self.inference.model.get_output_embeddings().weight

    def network(self):
        torch.manual_seed(0)
        return Exp6Network(
            trainable_encoder(self.inference), trainable_encoder(self.inference),
            self.unembedding(), hidden_size=HIDDEN, rank=RANK, head_rank=HEAD_RANK)

    def test_token_vectors_are_the_centered_unembedding_at_unit_scale(self):
        network = self.network()
        centered = self.unembedding().detach() - self.unembedding().detach().mean(dim=0)
        vectors = network.token_vectors
        self.assertEqual(vectors.shape, (VOCAB, HIDDEN))
        self.assertFalse(vectors.requires_grad)
        torch.testing.assert_close(vectors.mean(dim=0), torch.zeros(HIDDEN), atol=1e-6, rtol=0)
        self.assertAlmostEqual(vectors.square().mean().item(), 1.0, places=5)
        # One scale for the whole matrix.
        torch.testing.assert_close(
            vectors * centered.square().mean().sqrt(), centered, atol=1e-6, rtol=0)
        # The frozen model's, so neither trained nor saved.
        self.assertNotIn("token_vectors", network.state_dict())
        self.assertNotIn("token_vectors", dict(network.named_parameters()))

    def test_right_features_are_the_factorised_bilinear_form(self):
        network = self.network()
        futures = torch.tensor(FUTURES)
        with torch.no_grad():
            right = network.right_features(futures, self.HEADER)
            coefficients = network.coefficients(futures)
            hidden = network.right_encoder(input_ids=futures).last_hidden_state[:, -1]
            torch.testing.assert_close(
                coefficients, network.right_head(hidden).reshape(len(FUTURES), RANK, HEAD_RANK))
            # w_l(future) = B_l(future) P, a rank-d factorisation of rank x hidden numbers.
            vectors = coefficients @ network.token_projection.weight
        self.assertEqual(coefficients.shape, (len(FUTURES), RANK, HEAD_RANK))
        self.assertEqual(vectors.shape, (len(FUTURES), RANK, HIDDEN))
        self.assertEqual(right.shape, (len(FUTURES) * TOP_K, RANK))
        # Row f * top_k + j is <w(future f), e_z> for z = HEADER[f, j].
        for f in range(len(FUTURES)):
            for j in range(TOP_K):
                expected = vectors[f] @ network.token_vectors[self.HEADER[f, j]] / HEAD_RANK ** 0.5
                torch.testing.assert_close(right[f * TOP_K + j], expected, atol=1e-5, rtol=0)

    def test_the_header_token_enters_only_through_its_unembedding_row(self):
        network = self.network()
        futures = torch.tensor(FUTURES[:1])
        with torch.no_grad():
            # Token 2's row becomes 2 x row 0 - 3 x row 1; its features must follow.
            network.token_vectors[2] = 2 * network.token_vectors[0] - 3 * network.token_vectors[1]
            right = network.right_features(futures, torch.tensor([[0, 1, 2]]))
        torch.testing.assert_close(right[2], 2 * right[0] - 3 * right[1], atol=1e-5, rtol=0)

    def test_a_head_rank_of_top_k_reaches_any_values_on_a_future(self):
        # With d >= top_k the projected rows of a future's header tokens are
        # independent, so some B_l(future) gives any values on them.
        network = self.network()
        self.assertGreaterEqual(HEAD_RANK, TOP_K)
        with torch.no_grad():
            projected = network.token_projection(network.token_vectors[self.HEADER])
        self.assertEqual(projected.shape, (len(FUTURES), TOP_K, HEAD_RANK))
        self.assertTrue((torch.linalg.matrix_rank(projected) == TOP_K).all())

    def test_left_side_is_a_linear_head_on_its_own_encoder(self):
        network = self.network()
        self.assertIsNot(network.left_encoder, network.right_encoder)
        ids = torch.tensor(HISTORIES)
        left = network.left_features(ids)
        self.assertEqual(left.shape, (len(HISTORIES), RANK))
        # Before any training the encoder is the model.
        torch.testing.assert_close(
            left, network.left_head(last_hidden(self.inference, ids)), atol=1e-6, rtol=0)
        for head, outputs in ((network.left_head, RANK), (network.right_head, RANK * HEAD_RANK)):
            self.assertEqual([type(layer).__name__ for layer in head], ["LayerNorm", "Linear"])
            self.assertFalse(head[0].elementwise_affine)
            self.assertEqual((head[1].in_features, head[1].out_features), (HIDDEN, outputs))
        self.assertEqual(network.token_projection.weight.shape, (HEAD_RANK, HIDDEN))
        self.assertIsNone(network.token_projection.bias)

    def test_gradients_reach_both_encoders_the_heads_and_the_projection(self):
        network = self.network()
        left, right = network(torch.tensor(HISTORIES[:5]), torch.tensor(FUTURES), self.HEADER)
        self.assertEqual((left.shape, right.shape), ((5, RANK), (5 * TOP_K, RANK)))
        (left.sum() + right.square().sum()).backward()
        for name, parameter in network.named_parameters():
            if ".wpe." in name or ".ln_f." in name or "left_encoder" in name:
                continue  # only what the right side's last token passes through is checked
            self.assertIsNotNone(parameter.grad, name)
        for module in (network.left_encoder.wte, network.right_encoder.wte, network.left_head,
                       network.right_head, network.token_projection):
            self.assertTrue(all(p.grad.abs().sum() > 0 for p in module.parameters()))
        # The right encoder reads the future only: the header tokens' input
        # embeddings get no gradient.
        used = set(torch.tensor(FUTURES).flatten().tolist())
        unused = [token for token in range(VOCAB) if token not in used]
        self.assertTrue(unused)
        self.assertEqual(network.right_encoder.wte.weight.grad[unused].abs().sum().item(), 0.0)

    def test_build_model_copies_the_transformer_twice(self):
        model = build_model(self.inference, rank=RANK, head_rank=HEAD_RANK)
        self.assertIsInstance(model, NestedLoRA)
        self.assertFalse(model.sequential)
        self.assertEqual(model.neigs, RANK)
        network = model.model
        self.assertEqual((network.rank, network.head_rank), (RANK, HEAD_RANK))
        original = self.inference.model.base_model
        for encoder in (network.left_encoder, network.right_encoder):
            self.assertIsNot(encoder, original)
            self.assertTrue(all(parameter.requires_grad for parameter in encoder.parameters()))
        # Nothing of the frozen model is trained, its unembedding included.
        frozen = {id(parameter) for parameter in self.inference.model.parameters()}
        self.assertFalse(any(id(parameter) in frozen for parameter in model.parameters()))

    def test_the_exact_target_is_bilinear_in_the_header_token(self):
        # The identity the right side's form rests on: the table entry minus g is
        # <s(history + future) - s(future), e_z>, e_z the centered unembedding row.
        histories, futures = torch.tensor(HISTORIES[:5]), torch.tensor(FUTURES)
        future_hidden = last_hidden(self.inference, futures)
        header_ids, g = header_tokens(future_only_logits(self.inference, future_hidden), TOP_K)
        pair_logits = self.inference.next_token_logits_batch(
            torch.cat([histories, futures], dim=1).tolist())
        target = pair_targets(pair_logits, header_ids, g)
        pair_hidden = last_hidden(self.inference, torch.cat([histories, futures], dim=1))
        unembedding = self.unembedding().detach()
        rows = (unembedding - unembedding.mean(dim=0))[header_ids]
        expected = torch.einsum("be,bke->bk", pair_hidden - future_hidden, rows)
        torch.testing.assert_close(target, expected, atol=1e-5, rtol=0)


class TrainingTests(Exp5Fixture):
    def train(self, output_dir, **changes):
        options = dict(
            inference_dir=self.inference_dir, store_dir=self.store_dir,
            model_path=self.model_path, output_dir=output_dir, device="cpu", neigs=2,
            head_rank=4, batch_size=8, steps=4, eval_every=2, lr=1e-2, encoder_lr=1e-3,
            seed=0, overwrite=False,
        )
        options.update(changes)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            summary = train_exp6(**options)
        self.log = output.getvalue()
        return summary

    def records(self, output_dir):
        lines = (output_dir / "eval_history.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]

    def test_writes_outputs_and_compares_both_tables(self):
        output = self.root / "exp6-outputs"
        summary = self.train(output)
        self.assertEqual(summary, {"output_dir": str(output), "step": 4, "evaluations": 2})
        self.assertIn("bilinear right side with head rank 4", self.log)
        records = self.records(output)
        self.assertEqual([record["step"] for record in records], [2, 4])
        for record in records:
            self.assertEqual(set(record), {
                "step", "train_loss", "kernel_energy", "seconds_per_step", "train", "val"})
            self.assertEqual(record["train"]["ranks"], [1, 2])
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual((metadata["neigs"], metadata["head_rank"]), (2, 4))
        self.assertEqual((metadata["lr"], metadata["encoder_lr"]), (1e-2, 1e-3))
        self.assertEqual(metadata["top_k"], TOP_K)
        self.assertEqual(metadata["table_shapes"], {"train": [4, 9], "val": [2, 6]})
        self.assertIn("head_rank", metadata["right_features"])
        heads = (HIDDEN * 2 + 2) + (HIDDEN * 2 * 4 + 2 * 4) + 4 * HIDDEN
        self.assertEqual(
            metadata["trainable_parameters"],
            heads + 2 * sum(p.numel() for p in self.inference.model.base_model.parameters()))
        for name in ("checkpoint.pt", "checkpoint-best.pt"):
            state = torch.load(output / name, weights_only=True)["model_state_dict"]
            self.assertFalse(any("token_vectors" in key for key in state))
            self.assertIn("model.token_projection.weight", state)

    def test_training_moves_everything_and_recovers_the_leading_mode(self):
        output = self.root / "exp6-learns"
        self.train(output, steps=300, eval_every=100, batch_size=16)
        state = torch.load(output / "checkpoint.pt", weights_only=True)["model_state_dict"]
        original = self.inference.model.base_model.state_dict()
        for side in ("left_encoder", "right_encoder"):
            moved = max(
                (state[f"model.{side}.{key}"] - value).abs().max().item()
                for key, value in original.items() if value.is_floating_point())
            self.assertGreater(moved, 1e-4, side)
        torch.manual_seed(0)
        initial = build_model(self.inference, 2, 4).state_dict()
        for key in ("left_head.1.weight", "right_head.1.weight", "token_projection.weight"):
            self.assertGreater((state[f"model.{key}"] - initial[f"model.{key}"]).abs().max(), 1e-4)
        records = self.records(output)
        # An untrained network scores at or below zero; the exact value is 1.
        for name in ("train", "val"):
            self.assertGreater(records[-1][name]["captured_energy"][0], 0.5)
        self.assertLess(records[-1]["train_loss"], records[0]["train_loss"])
        self.assertGreater(records[-1]["train_loss"], -1.5 * records[-1]["kernel_energy"])

    def test_resume_needs_the_same_settings(self):
        output = self.root / "exp6-resume"
        self.train(output, steps=2)
        self.assertEqual(self.train(output, steps=4)["evaluations"], 2)
        self.assertIn("Resuming from step 2", self.log)
        self.assertEqual([record["step"] for record in self.records(output)], [2, 4])
        for change in ({"head_rank": 3}, {"encoder_lr": 1e-4}, {"neigs": 3}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "--overwrite"):
                self.train(output, steps=6, **change)
        self.assertEqual(self.train(output, steps=2, head_rank=3, overwrite=True)["step"], 2)

    def test_rejects_missing_tables_a_negative_seed_and_another_store(self):
        with self.assertRaisesRegex(ValueError, "exp4 inference"):
            self.train(self.root / "exp6-none", inference_dir=self.root / "absent")
        with self.assertRaises(ValueError):
            self.train(self.root / "exp6-seed", seed=-1)
        store = self.root / "exp6-store-other-model"
        shutil.copytree(self.store_dir, store)
        config = json.loads((store / "store.json").read_text())
        config["model_files"][next(iter(config["model_files"]))]["size"] += 1
        (store / "store.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "model"):
            self.train(self.root / "exp6-other-model", store_dir=store)

    def test_table_features_read_histories_and_columns(self):
        loaded = load_eval_table(self.inference_dir / "train", self.inference, "cpu")
        torch.manual_seed(0)
        model = build_model(self.inference, rank=4, head_rank=4)
        left, right = token_features(model, loaded, chunk=2)
        self.assertEqual((left.dtype, right.dtype), (torch.float64, torch.float64))
        futures = len(loaded.future_ids)
        header_ids = loaded.table.columns[:, -1].reshape(futures, TOP_K)
        with torch.no_grad():
            expected_left = model.model.left_features(loaded.table.histories)
            expected_right = model.model.right_features(loaded.future_ids, header_ids)
        torch.testing.assert_close(left.float(), expected_left, atol=1e-6, rtol=0)
        torch.testing.assert_close(right.float(), expected_right, atol=1e-6, rtol=0)
        self.assertFalse(model.training)
        self.assertEqual(compare_with_exact_svd(left, right, loaded.table)["ranks"], [1, 2, 4])


class CLITests(Exp5Fixture):
    def test_train_writes_under_the_exp6_run_directory(self):
        results_root = self.root / "exp6-results"
        shutil.copytree(
            self.inference_dir,
            results_root / "exp4_table_neural_svd" / "tiny__model" / "inference",
        )
        arguments = [
            "train", "--results-root", str(results_root), "--dataset", "tiny",
            "--model-path", str(self.model_path), "--store-dir", str(self.store_dir),
            "--device", "cpu", "--neigs", "2", "--head-rank", "4", "--batch-size", "8",
            "--steps", "2", "--eval-every", "2",
        ]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(arguments)
        run = results_root / "exp6_gpt2_bilinear_table" / "tiny__model" / "train"
        self.assertTrue((run / "eval_history.jsonl").is_file())
        self.assertIn(f"output_dir: {run}", output.getvalue())
        self.assertIn("step: 2", output.getvalue())
        # Other settings on the same run directory are a usage error.
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main([*arguments, "--head-rank", "3"])
        self.assertEqual(raised.exception.code, 2)

    def test_defaults_match_the_design(self):
        with patch("experiments.exp6_gpt2_bilinear_table.train.train_exp6") as train:
            train.return_value = {"output_dir": "x", "step": 0, "evaluations": 0}
            with contextlib.redirect_stdout(io.StringIO()):
                main(["train"])
        options = train.call_args.kwargs
        self.assertEqual(
            {key: options[key] for key in (
                "neigs", "head_rank", "batch_size", "steps", "eval_every", "lr", "encoder_lr",
                "seed", "device", "overwrite")},
            {"neigs": 128, "head_rank": 128, "batch_size": 128, "steps": 100000,
             "eval_every": 1000, "lr": 3e-4, "encoder_lr": 2e-5, "seed": 42, "device": "cuda",
             "overwrite": False})
        self.assertEqual(options["model_path"], Path("models/gpt2-small"))
        self.assertEqual(
            options["store_dir"],
            Path("data/sample_stores/wikitext-103-train__gpt2-small__h64-f32"))
        self.assertEqual(options["inference_dir"], Path(
            "data/experiment_results/exp4_table_neural_svd/wikitext-large__gpt2-small/inference"))
        self.assertEqual(options["output_dir"], Path(
            "data/experiment_results/exp6_gpt2_bilinear_table/wikitext-large__gpt2-small/train"))


if __name__ == "__main__":
    unittest.main()
