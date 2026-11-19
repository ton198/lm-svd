import contextlib
import io
import json
import shutil
import unittest
from unittest.mock import patch

import torch
from test_exp5_gpt2_mlp_table import Exp5Fixture
from test_exp7_center import CenterFixture

# setUpModule builds the tiny models the fixtures read.
from test_exp7_independent_subnets import setUpModule  # noqa: F401
from test_exp7_joint import JointHelpers

from experiments.exp5_gpt2_mlp_table.cli import main as exp5_main
from experiments.exp5_gpt2_mlp_table.plateau import CONFIG_KEY, Plateau, continues
from experiments.exp5_gpt2_mlp_table.train import train_exp5
from experiments.exp7_independent_subnets import WARMUP_STEPS
from experiments.exp7_independent_subnets.cli import main as exp7_main

# Fires at every chance: after two evaluations at each learning rate.
ALWAYS = dict(window=1, gain=1e9, factor=0.5)


def record(energy, scale=None, table="val"):
    entry = {table: {"captured_energy": [0.0, energy]}}
    if scale is not None:
        entry["lr_scale"] = scale
    return entry


class PlateauRuleTests(unittest.TestCase):
    def test_the_rule_waits_for_two_windows(self):
        rule = Plateau("val", window=2, gain=0.01, factor=0.5)
        self.assertEqual(rule.scale([]), 1.0)
        self.assertEqual(rule.scale([record(0.1)] * 3), 1.0)
        self.assertEqual(rule.scale([record(0.1)] * 4), 0.5)

    def test_a_rising_energy_keeps_the_rate_and_a_flat_one_lowers_it(self):
        rule = Plateau("val", window=2, gain=0.01, factor=0.5)
        rising = [record(value) for value in (0.100, 0.100, 0.102, 0.102)]
        self.assertEqual(rule.scale(rising), 1.0)
        # Means 0.100 and 0.1005: half a percent, under the gain.
        flat = [record(value) for value in (0.100, 0.100, 0.102, 0.099)]
        self.assertEqual(rule.scale(flat), 0.5)
        # Only the largest rank and only the last two windows count.
        earlier = [record(0.5), record(0.5), *rising]
        self.assertEqual(rule.scale(earlier), 1.0)

    def test_a_falling_or_negative_energy_lowers_the_rate(self):
        rule = Plateau("val", window=1, gain=0.01, factor=0.5)
        self.assertEqual(rule.scale([record(0.2), record(0.1)]), 0.5)
        self.assertEqual(rule.scale([record(-0.2), record(-0.2)]), 0.5)
        self.assertEqual(rule.scale([record(-0.2), record(-0.1)]), 1.0)

    def test_after_lowering_only_evaluations_at_the_new_rate_count(self):
        rule = Plateau("val", window=2, gain=0.01, factor=0.5)
        history = [record(0.1)] * 4
        history += [record(0.1, 0.5)] * 3
        self.assertEqual(rule.scale(history), 0.5)
        history.append(record(0.1, 0.5))
        self.assertEqual(rule.scale(history), 0.25)
        # Records made before the rule have no scale: they count as the first rates.
        self.assertEqual(rule.scale([record(0.1), record(0.1), record(0.3, 1.0), record(0.3, 1.0)]),
                         1.0)

    def test_the_rule_reads_its_own_table(self):
        history = [{"val": {"captured_energy": [0.1]}, "train": {"captured_energy": [value]}}
                   for value in (0.1, 0.2)]
        self.assertEqual(Plateau("val", **ALWAYS).scale(history), 0.5)
        self.assertEqual(Plateau("train", window=1, gain=0.01, factor=0.5).scale(history), 1.0)

    def test_bad_numbers_are_rejected(self):
        for changes in (dict(window=0), dict(gain=-0.1), dict(factor=0.0), dict(factor=1.0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Plateau("val", **changes)

    def test_a_constant_run_may_take_up_the_rule_and_nothing_else_changes(self):
        constant = {"lr": 1e-4, "seed": 0}
        rule = {**constant, CONFIG_KEY: Plateau("val").config()}
        self.assertTrue(continues(constant, constant))
        self.assertTrue(continues(rule, rule))
        self.assertTrue(continues(constant, rule))
        self.assertFalse(continues(rule, constant))
        self.assertFalse(continues(rule, {**constant, CONFIG_KEY: Plateau("train").config()}))
        self.assertFalse(continues({**constant, "lr": 1e-3}, rule))


class Exp5PlateauTests(Exp5Fixture):
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

    def plateau(self, output, **changes):
        options = dict(steps=8, lr_schedule="plateau", plateau=Plateau("val", **ALWAYS))
        options.update(changes)
        return self.train(output, **options)

    def test_the_rates_fall_when_the_rule_fires(self):
        output = self.root / "run-plateau"
        self.plateau(output)
        records = self.records(output)
        self.assertEqual([entry["lr_scale"] for entry in records], [1.0, 1.0, 0.5, 0.5])
        self.assertEqual([entry["lr"] for entry in records], [1e-2, 1e-2, 5e-3, 5e-3])
        self.assertEqual(self.log.count("stopped rising"), 2)
        self.assertIn("0.25 of its first value", self.log)
        config = torch.load(output / "checkpoint.pt", weights_only=True)["config"]
        self.assertEqual(config[CONFIG_KEY], Plateau("val", **ALWAYS).config())
        self.assertNotIn("steps", config)
        self.assertEqual(json.loads((output / "metadata.json").read_text())[CONFIG_KEY]["table"],
                         "val")

    def test_the_rule_keeps_the_rates_until_it_has_two_windows(self):
        output = self.root / "run-plateau-waits"
        self.plateau(output, plateau=Plateau("val", window=3, gain=1e9, factor=0.5))
        self.assertEqual([entry["lr_scale"] for entry in self.records(output)], [1.0] * 4)
        self.assertNotIn("stopped rising", self.log)

    def test_a_continued_run_follows_the_uninterrupted_rates(self):
        whole, parts = self.root / "run-plateau-whole", self.root / "run-plateau-parts"
        self.plateau(whole)
        self.plateau(parts, steps=4)
        self.plateau(parts)
        self.assertIn("Resuming from step 4", self.log)
        self.assertEqual(
            [(entry["step"], entry["lr_scale"], entry["lr"]) for entry in self.records(parts)],
            [(entry["step"], entry["lr_scale"], entry["lr"]) for entry in self.records(whole)])

    def test_a_copy_of_a_constant_run_takes_up_the_rule(self):
        constant, copy = self.root / "run-constant", self.root / "run-constant-plateau"
        self.train(constant, steps=4)
        shutil.copytree(constant, copy)
        self.plateau(copy)
        self.assertIn("Resuming from step 4", self.log)
        records = self.records(copy)
        self.assertEqual(records[:2], self.records(constant))
        self.assertNotIn("lr_scale", records[0])
        # The two constant evaluations are the two windows: the rate falls at once.
        self.assertEqual([entry["lr_scale"] for entry in records[2:]], [0.5, 0.5])
        # It is a plateau run from then on.
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.train(copy, steps=12)
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.plateau(copy, steps=12, plateau=Plateau("train", **ALWAYS))
        self.assertEqual(self.train(constant, steps=6)["step"], 6)

    def test_the_schedule_and_the_rule_go_together(self):
        with self.assertRaisesRegex(ValueError, "go together"):
            self.train(self.root / "run-no-rule", lr_schedule="plateau")
        with self.assertRaisesRegex(ValueError, "go together"):
            self.train(self.root / "run-no-schedule", plateau=Plateau("val"))
        with self.assertRaisesRegex(ValueError, "train or val"):
            self.plateau(self.root / "run-bad-table", plateau=Plateau("report"))


class Exp5PlateauCLITests(unittest.TestCase):
    def run_cli(self, *extra):
        exp5_main(["train", "--device", "cpu", *extra])

    def test_the_schedule_passes_the_rule(self):
        with patch("experiments.exp5_gpt2_mlp_table.train.train_exp5") as train:
            train.return_value = {"output_dir": "x", "step": 0, "evaluations": 0}
            with contextlib.redirect_stdout(io.StringIO()):
                self.run_cli("--lr-schedule", "plateau", "--plateau-window", "5")
                self.run_cli()
        first, second = (call.kwargs for call in train.call_args_list)
        self.assertEqual(first["plateau"], Plateau("val", window=5, gain=0.01, factor=0.3))
        self.assertEqual(first["lr_schedule"], "plateau")
        self.assertIsNone(second["plateau"])

    def test_the_run_directory_gets_the_suffix(self):
        with patch("experiments.exp5_gpt2_mlp_table.train.train_exp5") as train, \
                patch("experiments.exp7_independent_subnets.center.read_pool_config",
                      return_value={"histories": 64}):
            train.return_value = {"output_dir": "x", "step": 0, "evaluations": 0}
            with contextlib.redirect_stdout(io.StringIO()):
                self.run_cli("--lr-schedule", "plateau")
                self.run_cli("--lr-schedule", "plateau", "--center-pool", "pool")
                self.run_cli()
        self.assertEqual(
            [call.kwargs["output_dir"].name for call in train.call_args_list],
            ["train-plateau", "train-center64-plateau", "train"])
        # The tables are those of the constant run: the same results root.
        self.assertEqual(
            len({call.kwargs["inference_dir"] for call in train.call_args_list}), 1)

    def test_a_bad_rule_is_a_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            self.run_cli("--lr-schedule", "plateau", "--plateau-factor", "2")
        self.assertEqual(raised.exception.code, 2)


class JointPlateauTests(JointHelpers, CenterFixture):
    def plateau(self, output_dir, **changes):
        options = dict(steps=8, plateau=Plateau("report", **ALWAYS))
        options.update(changes)
        return self.joint(output_dir, **options)

    def rates(self, output_dir):
        saved = torch.load(output_dir / "checkpoint.pt", weights_only=True)
        return [group["lr"] for group in saved["optimizer_state_dict"]["param_groups"]]

    def test_the_rates_fall_with_the_warmup_of_a_scratch_run(self):
        output_dir = self.root / "joint-plateau"
        self.plateau(output_dir)
        records = self.records(output_dir)
        self.assertEqual([entry["lr_scale"] for entry in records], [1.0, 1.0, 0.5, 0.5])
        self.assertEqual(self.log.count("the report energy stopped rising"), 2)
        # The rate of the next step: the warmup at step 8 and two halvings.
        self.assertEqual(len(self.rates(output_dir)), 1)
        self.assertAlmostEqual(self.rates(output_dir)[0], 1e-2 * 9 / WARMUP_STEPS * 0.25)
        saved = torch.load(output_dir / "checkpoint.pt", weights_only=True)
        self.assertEqual(saved["config"][CONFIG_KEY]["table"], "report")
        metadata = json.loads((output_dir / "metadata.json").read_text())
        self.assertIn("multiplied by 0.5", metadata["lr_rule"])
        self.assertEqual(metadata["tables"]["report"]["role"], "watched by the learning-rate rule")
        self.assertEqual(metadata["tables"]["selection"]["role"], "reported only")

    def test_both_rates_of_a_pretrained_kind_fall(self):
        output_dir = self.root / "joint-plateau-gpt2"
        self.plateau(output_dir, subnet="gpt2", steps=4)
        self.assertEqual(self.rates(output_dir), [5e-3, 5e-4])

    def test_a_run_without_the_rule_has_no_scale(self):
        output_dir = self.root / "joint-no-plateau"
        self.joint(output_dir)
        self.assertNotIn("lr_scale", self.records(output_dir)[0])
        metadata = json.loads((output_dir / "metadata.json").read_text())
        self.assertNotIn("lr_rule", metadata)
        self.assertNotIn(CONFIG_KEY, metadata)

    def test_a_continued_run_follows_the_uninterrupted_rates(self):
        whole, parts = self.root / "joint-plateau-whole", self.root / "joint-plateau-parts"
        self.plateau(whole)
        self.plateau(parts, steps=4)
        self.assertAlmostEqual(self.rates(parts)[0], 1e-2 * 5 / WARMUP_STEPS * 0.5)
        self.plateau(parts)
        self.assertEqual(
            [(entry["step"], entry["lr_scale"]) for entry in self.records(parts)],
            [(entry["step"], entry["lr_scale"]) for entry in self.records(whole)])
        self.assertEqual(self.rates(parts), self.rates(whole))

    def test_a_copy_of_a_constant_run_takes_up_the_rule(self):
        constant, copy = self.root / "joint-constant", self.root / "joint-constant-plateau"
        self.joint(constant)
        shutil.copytree(constant, copy)
        self.plateau(copy)
        self.assertIn("Resuming from step 4", self.log)
        records = self.records(copy)
        self.assertEqual(records[:2], self.records(constant))
        self.assertEqual([entry["lr_scale"] for entry in records[2:]], [0.5, 0.5])
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.joint(copy, steps=12)
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.plateau(copy, steps=12, plateau=Plateau("selection", **ALWAYS))

    def test_the_rule_watches_one_of_the_two_tables(self):
        with self.assertRaisesRegex(ValueError, "selection"):
            self.plateau(self.root / "joint-bad-table", plateau=Plateau("val"))


class JointPlateauCLITests(CenterFixture):
    def arguments(self, results_root, *extra):
        return [
            "train", "--steps", "2", "--results-root", str(results_root),
            "--dataset", "tiny", "--model-path", str(self.model_path),
            "--center-pool", str(self.pool_dir), "--subnet", "scratch", "--modes", "2",
            "--eval-every", "2", "--scratch-layers", "1", "--scratch-width", "4",
            "--scratch-heads", "2", "--batch-size", "8", "--device", "cpu", *extra,
        ]

    def test_the_run_gets_the_suffix_and_the_rule(self):
        results_root = self.root / "joint-plateau-results"
        shutil.copytree(
            self.inference_dir,
            results_root / "exp4_table_neural_svd" / "tiny__model" / "inference",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            exp7_main(self.arguments(
                results_root, "--joint", "--lr-plateau", "--plateau-table", "selection",
                "--plateau-window", "7"))
        run = (results_root / "exp7_independent_subnets" / "tiny__model"
               / "train-scratch-center3-joint-plateau")
        metadata = json.loads((run / "metadata.json").read_text())
        self.assertEqual(
            metadata[CONFIG_KEY], {"table": "selection", "window": 7, "gain": 0.01, "factor": 0.3})

    def test_the_rule_needs_a_joint_run(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors, \
                self.assertRaises(SystemExit) as raised:
            exp7_main(self.arguments(self.root / "unused", "--lr-plateau"))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--joint", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
