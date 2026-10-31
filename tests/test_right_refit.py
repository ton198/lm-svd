import tempfile
import unittest
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.exp6_stream_gpt2_mlp.features import build_features as build_gpt_features
from experiments.exp11_stream_student.features import build_features as build_student_features
from experiments.exp16_right_refit.features import build_features
from utils.inference import CausalLMInference
from utils.neural_svd import NestedLoRASVDLoss
from utils.stream_neural_svd import centered_unembedding, column_features


class RightRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        model_path = Path(directory.name) / "model"
        GPT2LMHeadModel(GPT2Config(
            vocab_size=11, n_positions=12, n_embd=8, n_layer=1, n_head=2,
            bos_token_id=None, eos_token_id=None,
        )).save_pretrained(model_path)
        cls.inference = CausalLMInference(model_path, "cpu")
        cls.tokens = centered_unembedding(cls.inference)
        cls.histories = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [4, 5, 6, 7]])
        cls.futures = torch.tensor([[6, 7], [8, 9], [9, 10], [0, 1]])

    def make_source(self, name: str):
        if name == "exp6":
            config = {"kind": "frozen_gpt2_mlp", "mlp_width": 16, "hidden_size": 8, "rank": 3}
            return build_gpt_features(config, self.inference).eval()
        config = {
            "kind": "distilled_student_svd",
            "student": {"model_dim": 8, "layers": 1, "heads": 2, "max_positions": 12},
            "rank": 3, "column_dim": 4, "skip": True, "init": "random",
            "student_checkpoint": None, "hidden_size": 8,
        }
        return build_student_features(config, self.inference).eval()

    def make_refit(self, name: str, variant: str, source):
        right = {} if variant == "continue" else (
            {"width": 16, "blocks": 1, "block_hidden": 32, "bottleneck": 8}
            if name == "exp6" else {"head_width": 16}
        )
        config = {
            "kind": "right_refit", "source": name, "variant": variant,
            "source_features": source.config, "source_train_dir": "pinned/train",
            "source_steps": 10, "right": right,
        }
        model = build_features(config, self.inference)
        model.load_source_state(source.state_dict())
        return model.eval()

    def test_continue_starts_at_source_and_only_right_trains(self) -> None:
        for name in ("exp6", "exp11"):
            with self.subTest(source=name):
                source = self.make_source(name)
                model = self.make_refit(name, "continue", source)
                prefixes = ("right.",) if name == "exp6" else (
                    "column_head.", "column_basis.", "trunk.out_proj.",
                )
                expected = {"base." + key for key, _ in source.named_parameters() if key.startswith(prefixes)}
                trainable = {key for key, parameter in model.named_parameters() if parameter.requires_grad}
                self.assertEqual(trainable, expected)
                # Compare the inference surface with identical mode and grad context.
                with torch.no_grad():
                    rows = model.row_features(self.histories).clone()
                    weights = model.column_weights(self.futures).clone()
                    self.assertTrue(torch.equal(rows, source.row_features(self.histories)))
                    self.assertTrue(torch.equal(weights, source.column_weights(self.futures)))
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
                columns = column_features(model.column_weights(self.futures), self.tokens)
                block = torch.randn(len(self.histories), columns.shape[0])
                loss = NestedLoRASVDLoss(3)(model.row_features(self.histories), columns, block)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    self.assertTrue(torch.equal(rows, model.row_features(self.histories)))
                    self.assertFalse(torch.equal(weights, model.column_weights(self.futures)))

    def test_wide_keeps_rows_and_copies_student_trunk(self) -> None:
        for name in ("exp6", "exp11"):
            with self.subTest(source=name):
                source = self.make_source(name)
                model = self.make_refit(name, "wide", source)
                with torch.no_grad():
                    self.assertTrue(torch.equal(
                        source.row_features(self.histories), model.row_features(self.histories)
                    ))
                if name == "exp11":
                    source_trunk = source.trunk.state_dict()
                    copied_trunk = model.column_trunk.state_dict()
                    self.assertEqual(source_trunk.keys(), copied_trunk.keys())
                    for key in source_trunk:
                        self.assertTrue(torch.equal(source_trunk[key], copied_trunk[key]), key)
                    with torch.no_grad():
                        self.assertTrue(torch.equal(
                            source.column_weights(self.futures)[:, 0],
                            model.column_weights(self.futures)[:, 0],
                        ))

    def test_rebuild_from_config_round_trip(self) -> None:
        for name in ("exp6", "exp11"):
            with self.subTest(source=name):
                source = self.make_source(name)
                model = self.make_refit(name, "wide", source)
                rebuilt = build_features(model.config, self.inference).eval()
                rebuilt.load_state_dict(model.state_dict(), strict=True)
                self.assertTrue(torch.equal(
                    model.column_weights(self.futures), rebuilt.column_weights(self.futures)
                ))


if __name__ == "__main__":
    unittest.main()
