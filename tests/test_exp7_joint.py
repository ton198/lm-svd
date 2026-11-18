import contextlib
import io
import json
import shutil
import unittest
from unittest.mock import patch

import torch
from test_exp5_gpt2_mlp_table import FUTURE_TOKENS, HISTORY_TOKENS
from test_exp7_center import CenterFixture

# setUpModule silences the progress bars of the models the fixtures load.
from test_exp7_independent_subnets import SCRATCH, setUpModule  # noqa: F401

from experiments.exp5_gpt2_mlp_table.loss import paired_loss
from experiments.exp5_gpt2_mlp_table.train import sampler_seed
from experiments.exp7_independent_subnets import SUBNETS
from experiments.exp7_independent_subnets.cli import main
from experiments.exp7_independent_subnets.joint import joint_outputs, train_exp7_joint
from experiments.exp7_independent_subnets.model import token_vectors
from experiments.exp7_independent_subnets.stage import (
    joint_loss,
    nesting_weights,
    stage_loss,
)
from experiments.exp7_independent_subnets.train import MODE_SEED_OFFSET
from utils.nestedlora import NestedLoRA


def rank_loss(left, right, target, rank):
    """The loss of the first `rank` modes, written out from its definition."""
    left, right = left[:, :rank], right[:, :, :rank]
    fit = torch.einsum("bl,bjl->bj", left, right)
    operator = -2 * (target * fit).mean()
    # E over independent histories and columns of the squared fit.
    metric = torch.einsum("al,bjl->abj", left, right).square().mean()
    return operator + metric


class JointLossTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.left = torch.randn(6, 4, generator=generator, dtype=torch.float64)
        self.right = torch.randn(6, 5, 4, generator=generator, dtype=torch.float64)
        self.target = torch.randn(6, 5, generator=generator, dtype=torch.float64)

    def test_weights_count_the_ranks_a_mode_is_part_of(self):
        weights, pair_weights = nesting_weights(4)
        self.assertEqual(weights.tolist(), [1.0, 0.75, 0.5, 0.25])
        self.assertEqual(pair_weights[1].tolist(), [0.75, 0.75, 0.5, 0.25])

    def test_the_loss_is_the_mean_of_the_rank_losses(self):
        weights, pair_weights = (x.double() for x in nesting_weights(4))
        loss = joint_loss(self.left, self.right, self.target, weights, pair_weights)
        expected = sum(
            rank_loss(self.left, self.right, self.target, rank) for rank in range(1, 5)) / 4
        torch.testing.assert_close(loss, expected)

    def test_one_mode_gives_the_stage_loss(self):
        weights, pair_weights = (x.double() for x in nesting_weights(1))
        torch.testing.assert_close(
            joint_loss(self.left[:, :1], self.right[:, :, :1], self.target, weights, pair_weights),
            stage_loss(self.left[:, 0], self.right[:, :, 0], self.target))

    def test_the_loss_and_its_gradients_are_exp5s(self):
        left = self.left.float().requires_grad_()
        right = self.right.float().requires_grad_()
        target = self.target.float()
        loss = joint_loss(left, right, target, *nesting_weights(4))
        ours = torch.autograd.grad(loss, (left, right))
        other_left = self.left.float().requires_grad_()
        other_right = self.right.float().requires_grad_()
        model = NestedLoRA(model=torch.nn.Identity(), neigs=4)
        theirs_loss = paired_loss(model, other_left, other_right.reshape(30, 4), target)
        theirs = torch.autograd.grad(theirs_loss, (other_left, other_right))
        torch.testing.assert_close(loss, theirs_loss, rtol=1e-5, atol=1e-6)
        for mine, other in zip(ours, theirs):
            torch.testing.assert_close(mine, other, rtol=1e-4, atol=1e-6)

    def test_an_earlier_mode_gets_gradient_from_a_later_one(self):
        left = self.left.clone().requires_grad_()
        weights, pair_weights = (x.double() for x in nesting_weights(2))
        loss = joint_loss(
            left[:, :2], self.right[:, :, :2], self.target, weights, pair_weights)
        gradient, = torch.autograd.grad(loss, left)
        alone = self.left.clone().requires_grad_()
        own = stage_loss(alone[:, 0], self.right[:, :, 0], self.target)
        own_gradient, = torch.autograd.grad(own, alone)
        self.assertFalse(torch.allclose(gradient[:, 0], own_gradient[:, 0]))


class JointHelpers:
    def joint(self, output_dir, **changes):
        options = dict(
            inference_dir=self.inference_dir, stream_path=self.stream_path,
            model_path=self.model_path, output_dir=output_dir, device="cpu", subnet="scratch",
            modes=2, steps=4, **SCRATCH, batch_size=8, eval_every=2, lr=1e-2, encoder_lr=1e-3,
            seed=0, overwrite=False, roberta_path=self.roberta_dir, center_pool=self.pool_dir,
        )
        options.update(changes)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            try:
                return train_exp7_joint(**options)
            finally:
                self.log = output.getvalue()

    def records(self, output_dir):
        lines = (output_dir / "eval_history.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]


class JointTrainingTests(JointHelpers, CenterFixture):
    def test_a_joint_run_trains_every_mode_and_compares_all_ranks(self):
        output_dir = self.root / "joint-run"
        summary = self.joint(output_dir)
        self.assertEqual(summary, {
            "output_dir": str(output_dir), "step": 4, "modes": 2, "evaluations": 2})
        records = self.records(output_dir)
        self.assertEqual([record["step"] for record in records], [2, 4])
        for record in records:
            self.assertNotIn("stage", record)
            for key in ("selection", "report"):
                self.assertEqual(record[key]["ranks"], [1, 2])
        self.assertIn("Training 2 modes together", self.log)
        metadata = json.loads((output_dir / "metadata.json").read_text())
        self.assertEqual(metadata["training"], "joint")
        self.assertEqual(metadata["modes"], 2)
        self.assertEqual(metadata["center_pool"], self.pool.config)
        self.assertNotIn("store", metadata)
        self.assertNotIn("stage_steps", metadata)
        # Both modes were trained: neither still has the weights it was seeded with.
        for number in (1, 2):
            torch.manual_seed(sampler_seed(0, MODE_SEED_OFFSET + number))
            first = self.build("scratch").state_dict()
            saved = torch.load(
                output_dir / "modes" / f"mode-{number:02d}.pt", weights_only=True)
            self.assertEqual(saved["step"], 4)
            changed = [
                name for name, value in saved["state_dict"].items()
                if value.is_floating_point() and not torch.equal(value, first[name])]
            self.assertTrue(any("left_encoder" in name for name in changed))
            self.assertTrue(any("right_head" in name for name in changed))

    def test_every_kind_trains_and_a_pretrained_kind_has_two_rates(self):
        for subnet in SUBNETS:
            with self.subTest(subnet=subnet):
                output_dir = self.root / f"joint-{subnet}"
                seen = []
                real = torch.optim.Adam

                def recording(params, **kwargs):
                    optimizer = real(params, **kwargs)
                    seen.append([group["lr"] for group in optimizer.param_groups])
                    return optimizer

                with patch("torch.optim.Adam", recording):
                    self.joint(output_dir, subnet=subnet, steps=2)
                self.assertEqual(seen, [[1e-2] if subnet == "scratch" else [1e-2, 1e-3]])
                self.assertEqual(len(self.records(output_dir)), 1)

    def test_the_target_minus_g_trains_from_the_stream(self):
        output_dir = self.root / "joint-minus-g"
        self.joint(output_dir, center_pool=None, steps=2)
        config = torch.load(output_dir / "checkpoint.pt", weights_only=True)["config"]
        self.assertEqual(config["stream_path"], str(self.stream_path))
        self.assertNotIn("center_pool", config)

    def test_a_run_continues_to_more_steps_and_rejects_other_settings(self):
        output_dir = self.root / "joint-continued"
        self.joint(output_dir, steps=2)
        mode_path = output_dir / "modes" / "mode-02.pt"
        self.assertEqual(torch.load(mode_path, weights_only=True)["step"], 2)
        saved = torch.load(output_dir / "checkpoint.pt", weights_only=True)
        summary = self.joint(output_dir, steps=4)
        self.assertIn("Resuming from step 2", self.log)
        self.assertEqual(summary["step"], 4)
        records = self.records(output_dir)
        self.assertEqual([record["step"] for record in records], [2, 4])
        self.assertEqual(records[0], saved["history"][0])
        self.assertEqual(torch.load(mode_path, weights_only=True)["step"], 4)
        # Finished: the same command trains no further step.
        before = (output_dir / "checkpoint.pt").stat().st_mtime_ns
        self.assertEqual(self.joint(output_dir, steps=4)["evaluations"], 2)
        self.assertEqual((output_dir / "checkpoint.pt").stat().st_mtime_ns, before)
        for changes in (dict(modes=3), dict(lr=1e-3), dict(center_pool=None)):
            with self.assertRaisesRegex(ValueError, "--overwrite"):
                self.joint(output_dir, steps=6, **changes)

    def test_an_interrupted_run_continues_from_its_last_evaluation(self):
        output_dir = self.root / "joint-interrupted"
        calls = []
        real = torch.optim.Adam.step

        def failing(optimizer, *args, **kwargs):
            calls.append(1)
            if len(calls) == 4:
                raise RuntimeError("interrupted")
            return real(optimizer, *args, **kwargs)

        with patch("torch.optim.Adam.step", failing), \
                self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.joint(output_dir, steps=6)
        self.assertEqual([record["step"] for record in self.records(output_dir)], [2])
        self.joint(output_dir, steps=6)
        self.assertIn("Resuming from step 2", self.log)
        self.assertEqual([record["step"] for record in self.records(output_dir)], [2, 4, 6])

    def test_a_scratch_run_warms_up_once_and_keeps_its_place_when_continued(self):
        output_dir = self.root / "joint-warmup"
        self.joint(output_dir, steps=2)
        saved = torch.load(output_dir / "checkpoint.pt", weights_only=True)
        self.assertEqual(saved["scheduler_state_dict"]["last_epoch"], 2)
        self.joint(output_dir, steps=4)
        saved = torch.load(output_dir / "checkpoint.pt", weights_only=True)
        self.assertEqual(saved["scheduler_state_dict"]["last_epoch"], 4)

    def test_overwrite_starts_again_and_an_orphan_history_is_rejected(self):
        output_dir = self.root / "joint-overwrite"
        self.joint(output_dir, steps=4)
        self.joint(output_dir, steps=2, overwrite=True)
        self.assertEqual([record["step"] for record in self.records(output_dir)], [2])
        self.assertEqual(
            torch.load(output_dir / "modes" / "mode-01.pt", weights_only=True)["step"], 2)
        (output_dir / "checkpoint.pt").unlink()
        with self.assertRaisesRegex(FileExistsError, "--overwrite"):
            self.joint(output_dir, steps=2)

    def test_more_modes_than_exact_vectors_and_a_missing_pool_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact singular vectors"):
            self.joint(self.root / "joint-too-many", modes=1000)
        with self.assertRaisesRegex(ValueError, "no finished center pool"):
            self.joint(self.root / "joint-no-pool", center_pool=self.root / "no-pool")
        self.assertFalse((self.root / "joint-too-many").exists())

    def test_outputs_stack_the_modes_and_keep_their_gradients(self):
        modes = torch.nn.ModuleList(self.build("scratch") for _ in range(3))
        vectors = token_vectors(self.inference.model.get_output_embeddings().weight)
        histories = torch.zeros(4, HISTORY_TOKENS, dtype=torch.long)
        futures = torch.zeros(4, FUTURE_TOKENS, dtype=torch.long)
        header_ids = torch.arange(8).reshape(4, 2)
        left, right = joint_outputs(modes, vectors, histories, futures, header_ids)
        self.assertEqual(tuple(left.shape), (4, 3))
        self.assertEqual(tuple(right.shape), (4, 2, 3))
        torch.testing.assert_close(left[:, 1], modes[1].left_features(histories))
        self.assertTrue(left.requires_grad and right.requires_grad)


class JointCLITests(CenterFixture):
    def test_joint_writes_a_run_with_the_suffix(self):
        results_root = self.root / "joint-results"
        shutil.copytree(
            self.inference_dir,
            results_root / "exp4_table_neural_svd" / "tiny__model" / "inference",
        )
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main([
                "train", "--joint", "--steps", "2", "--results-root", str(results_root),
                "--dataset", "tiny", "--model-path", str(self.model_path),
                "--center-pool", str(self.pool_dir), "--subnet", "scratch", "--modes", "2",
                "--eval-every", "2", "--scratch-layers", "1", "--scratch-width", "4",
                "--scratch-heads", "2", "--batch-size", "8", "--device", "cpu",
            ])
        run = (results_root / "exp7_independent_subnets" / "tiny__model"
               / "train-scratch-center3-joint")
        self.assertTrue((run / "modes" / "mode-02.pt").is_file())
        self.assertIn(f"output_dir: {run}", output.getvalue())
        self.assertIn("step: 2", output.getvalue())

    def test_joint_without_a_pool_reads_the_stream(self):
        results_root = self.root / "joint-stream-results"
        shutil.copytree(
            self.inference_dir,
            results_root / "exp4_table_neural_svd" / "tiny__model" / "inference",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            main([
                "train", "--joint", "--steps", "2", "--results-root", str(results_root),
                "--dataset", "tiny", "--model-path", str(self.model_path),
                "--stream-path", str(self.stream_path), "--subnet", "scratch", "--modes", "1",
                "--eval-every", "2", "--scratch-layers", "1", "--scratch-width", "4",
                "--scratch-heads", "2", "--batch-size", "8", "--device", "cpu",
            ])
        run = results_root / "exp7_independent_subnets" / "tiny__model" / "train-scratch-joint"
        self.assertTrue((run / "checkpoint.pt").is_file())

    def test_a_missing_stream_is_a_usage_error(self):
        results_root = self.root / "joint-no-stream-results"
        shutil.copytree(
            self.inference_dir,
            results_root / "exp4_table_neural_svd" / "tiny__model" / "inference",
        )
        with contextlib.redirect_stderr(io.StringIO()) as error, \
                self.assertRaises(SystemExit) as raised:
            main(["train", "--joint", "--stream-path", str(self.root / "no-stream.npy"),
                  "--results-root", str(results_root), "--dataset", "tiny",
                  "--model-path", str(self.model_path), "--device", "cpu"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("no token stream", error.getvalue())


if __name__ == "__main__":
    unittest.main()
