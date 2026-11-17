import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.exp4_table_neural_svd.train import load_table
from experiments.exp8_hypotheses.center import (
    centered_loss,
    landmark_mean,
    load_centered_table,
    span_energy,
    train_centered,
)
from experiments.exp8_hypotheses.grid import Teacher, Windows
from experiments.exp8_hypotheses.mean import load_mean_table, mean_comparison, train_mean
from experiments.exp8_hypotheses.model import MeanStudent, build_network, parameter_groups
from experiments.exp8_hypotheses.runs import FINAL_LR_SHARE, RunFiles, lr_factor
from experiments.exp8_hypotheses.spectrum import (
    block_prediction,
    run_spectrum,
    split_report,
    subspace_transfer,
    thin_svd,
)
from utils.nestedlora import NestedLoRA
from test_exp5_gpt2_mlp_table import FUTURE_TOKENS, HISTORY_TOKENS, TOP_K, Exp5Fixture


def quiet(function, **arguments):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(**arguments)


class SpectrumTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.low_rank = (
            torch.randn(40, 3, generator=generator, dtype=torch.float64)
            @ torch.randn(3, 60, generator=generator, dtype=torch.float64))
        self.noise = torch.randn(40, 60, generator=generator, dtype=torch.float64)

    def test_thin_svd_matches_the_exact_svd_for_both_orientations(self):
        for matrix in (self.noise, self.noise.T):
            left, values, right = thin_svd(matrix)
            self.assertTrue(torch.allclose(values, torch.linalg.svdvals(matrix), atol=1e-8))
            self.assertTrue(torch.allclose((left * values) @ right.T, matrix, atol=1e-6))

    def test_modes_of_a_low_rank_matrix_carry_over_completely(self):
        transfer = subspace_transfer(self.low_rank[:, :30], self.low_rank[:, 30:], (1, 3))
        self.assertAlmostEqual(transfer["captured"][1], 1.0, places=8)
        self.assertLess(transfer["captured"][0], transfer["optimum"][0] + 1e-9)
        rows, columns = torch.arange(40), torch.arange(60)
        block = block_prediction(
            self.low_rank, rows[:20], rows[20:], columns[:30], columns[30:], (1, 3))
        self.assertAlmostEqual(block["captured"][1], 1.0, places=6)

    def test_modes_of_noise_do_not_predict_a_held_out_block(self):
        rows, columns = torch.arange(40), torch.arange(60)
        block = block_prediction(
            self.noise, rows[:20], rows[20:], columns[:30], columns[30:], (8,))
        self.assertLess(block["captured"][0], 0.0)
        self.assertGreater(block["optimum"][0], 0.5)

    def test_split_report_keeps_a_future_together_and_centers_each_half_apart(self):
        report = split_report(self.low_rank + 5.0, 4, center=True, seed=1, ranks=(1, 2, 3))
        self.assertTrue(report["centered"])
        # The shift of 5 is a column mean, so centering removes it without a trace.
        self.assertAlmostEqual(report["block_prediction"]["captured"][2], 1.0, places=6)
        with self.assertRaisesRegex(ValueError, "multiple of top_k"):
            split_report(self.noise, 7, center=False)


class RunFileTests(unittest.TestCase):
    def test_lr_factor_warms_up_and_decays_to_its_final_share(self):
        self.assertAlmostEqual(lr_factor(0, 20000), 1 / 1000)
        self.assertAlmostEqual(lr_factor(999, 20000), 1.0)
        self.assertAlmostEqual(lr_factor(20000, 20000), FINAL_LR_SHARE)
        self.assertAlmostEqual(lr_factor(3, 40), 1.0)  # a short run warms up over a tenth


class Exp8Fixture(Exp5Fixture):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.stream_path = cls.root / "stream.npy"
        cls.teacher = Teacher(cls.inference, chunk=3)

    def table_tokens(self, name: str) -> dict:
        tokens = np.load(self.inference_dir / name / "tokens.npz")
        return {key: torch.from_numpy(tokens[key]).long() for key in
                ("histories", "futures", "header_token_ids")}


class GridTests(Exp8Fixture):
    def test_grid_entries_equal_the_stored_table(self):
        tokens = self.table_tokens("train")
        stored = torch.from_numpy(np.load(self.inference_dir / "train" / "matrix.npy"))
        entries = self.teacher.grid_entries(
            tokens["histories"], tokens["futures"], tokens["header_token_ids"])
        self.assertEqual(entries.shape, (4, 3, TOP_K))
        self.assertTrue(torch.allclose(entries.flatten(1), stored, atol=1e-4))

    def test_pair_entries_are_the_grid_diagonal_and_headers_are_the_stored_ones(self):
        tokens = self.table_tokens("train")
        histories, futures = tokens["histories"][:3], tokens["futures"]
        headers, g = self.teacher.headers(futures, TOP_K)
        self.assertTrue(torch.equal(headers.cpu(), tokens["header_token_ids"]))
        self.assertTrue(torch.allclose(g, self.teacher.future_only(futures, headers), atol=1e-5))
        grid = self.teacher.grid_entries(histories, futures, headers)
        pairs = self.teacher.pair_entries(histories, futures, headers)
        self.assertTrue(torch.allclose(pairs, grid[range(3), range(3)], atol=1e-5))

    def test_windows_have_the_requested_shapes_and_reject_another_vocabulary(self):
        windows = Windows(self.stream_path, HISTORY_TOKENS, FUTURE_TOKENS,
                          self.inference.vocab_size, seed=0)
        history_ids, future_ids = windows.draw(5, 2)
        self.assertEqual(history_ids.shape, (5, HISTORY_TOKENS))
        self.assertEqual(future_ids.shape, (2, FUTURE_TOKENS))
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            Windows(self.stream_path, HISTORY_TOKENS, FUTURE_TOKENS, 7, seed=0)


class CenterTests(Exp8Fixture):
    def network(self, layout="shared", modes=3):
        torch.manual_seed(0)
        return build_network(self.inference, layout=layout, encoder="scratch", modes=modes)

    def test_loss_ignores_column_constants_and_constant_left_features(self):
        torch.manual_seed(0)
        wrapper = NestedLoRA(model=torch.nn.Identity(), neigs=3)
        left, right = torch.randn(5, 3), torch.randn(4, TOP_K, 3)
        entries = torch.randn(5, 4, TOP_K)
        loss = centered_loss(wrapper, left, right, entries)
        shifted = centered_loss(
            wrapper, left + torch.randn(3), right, entries + torch.randn(4, TOP_K))
        self.assertAlmostEqual(loss.item(), shifted.item(), places=5)
        manual = wrapper.compute_loss_kernel(
            left - left.mean(0), right.flatten(0, 1), (entries - entries.mean(0)).flatten(1))[0]
        self.assertAlmostEqual(loss.item(), manual.item(), places=6)

    def test_both_layouts_give_features_of_the_same_shapes(self):
        tokens = self.table_tokens("train")
        for layout in ("shared", "independent"):
            with self.subTest(layout=layout):
                network = self.network(layout)
                self.assertEqual(network.left_features(tokens["histories"]).shape, (4, 3))
                self.assertEqual(
                    network.right_features(tokens["futures"], tokens["header_token_ids"]).shape,
                    (3, TOP_K, 3))
        groups = parameter_groups(network, "scratch", 1e-4, 2e-5)
        self.assertEqual(len(groups), 1)

    def test_pretrained_encoders_get_their_own_learning_rate(self):
        network = build_network(self.inference, layout="shared", encoder="gpt2", modes=2)
        encoder_group, head_group = parameter_groups(network, "gpt2", 3e-4, 2e-5)
        self.assertEqual(encoder_group["lr"], 2e-5)
        self.assertEqual(head_group["lr"], 3e-4)
        heads = {id(p) for head in (network.left_head, network.right_head)
                 for p in head.parameters()}
        self.assertEqual({id(p) for p in head_group["params"]}, heads)

    def test_the_right_side_can_have_another_pretrained_encoder(self):
        from experiments.exp7_independent_subnets.model import LastToken, MeanPool
        network = build_network(
            self.inference, layout="shared", encoder="gpt2-mean", modes=2, right_encoder="gpt2")
        self.assertIsInstance(network.left_encoder, MeanPool)
        self.assertIsInstance(network.right_encoder, LastToken)
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            build_network(self.inference, layout="shared", encoder="scratch", modes=2,
                          right_encoder="gpt2")
        with self.assertRaisesRegex(ValueError, "shared layout"):
            build_network(self.inference, layout="independent", encoder="gpt2", modes=2,
                          right_encoder="gpt2")

    def test_span_energy_of_the_exact_vectors_is_the_exact_energy(self):
        table, _, _ = load_centered_table(self.inference_dir / "train", "cpu")
        ranks = [1, 2]
        spans = span_energy(
            table.left_vectors[:, :2], table.right_vectors[:, :2], table.matrix, ranks)
        exact = table.singular_values.square()
        for index, rank in enumerate(ranks):
            share = (exact[:rank].sum() / exact.sum()).item()
            self.assertAlmostEqual(spans["left_span_energy"][index], share, places=6)
            self.assertAlmostEqual(spans["right_span_energy"][index], share, places=6)
        self.assertLess(abs(table.matrix.mean(dim=0)).max().item(), 1e-5)
        self.assertEqual(
            table.matrix.shape, load_table(self.inference_dir / "train", "cpu").matrix.shape)

    def run_center(self, name="run", **changes):
        arguments = dict(
            inference_dir=self.inference_dir, stream_path=self.stream_path,
            model_path=self.model_path, output_dir=self.root / name, device="cpu",
            layout="shared", encoder="scratch", nesting="joint", modes=2, history_batch=4,
            future_batch=3, steps=4, eval_every=2, lr=1e-3, encoder_lr=1e-4, seed=0,
            overwrite=False)
        arguments.update(changes)
        return quiet(train_centered, **arguments)

    def test_training_writes_its_records_and_a_repeated_command_continues(self):
        summary = self.run_center("center-run")
        self.assertEqual(summary["step"], 4)
        lines = (self.root / "center-run" / "eval_history.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["step"] for line in lines], [2, 4])
        record = json.loads(lines[-1])
        self.assertEqual(record["val"]["ranks"], [1, 2])
        self.assertEqual(len(record["val"]["left_span_energy"]), 2)
        # The same command finds the finished run and adds nothing.
        self.assertEqual(self.run_center("center-run")["step"], 4)
        self.assertEqual(
            len((self.root / "center-run" / "eval_history.jsonl").read_text().splitlines()), 2)
        with self.assertRaisesRegex(ValueError, "other settings \\(modes\\)"):
            self.run_center("center-run", modes=3)
        self.assertEqual(self.run_center("center-run", modes=3, overwrite=True)["step"], 4)

    def test_a_stopped_run_resumes_from_its_checkpoint(self):
        self.run_center("center-stopped", steps=4)
        files = RunFiles(self.root / "center-stopped", {}, overwrite=False)
        state = torch.load(files.checkpoint_path, weights_only=False)
        # Rewind the record to the first evaluation, as if the job had stopped there.
        state["step"], state["history"] = 2, state["history"][:1]
        torch.save(state, files.checkpoint_path)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            summary = train_centered(
                inference_dir=self.inference_dir, stream_path=self.stream_path,
                model_path=self.model_path, output_dir=self.root / "center-stopped",
                device="cpu", layout="shared", encoder="scratch", nesting="joint", modes=2,
                history_batch=4, future_batch=3, steps=4, eval_every=2, lr=1e-3,
                encoder_lr=1e-4, seed=0, overwrite=False)
        self.assertIn("Resuming at step 2", output.getvalue())
        self.assertEqual(summary["step"], 4)
        lines = (self.root / "center-stopped" / "eval_history.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["step"] for line in lines], [2, 4])

    def test_independent_sequential_training_runs_and_bad_settings_are_rejected(self):
        summary = self.run_center(
            "center-independent", layout="independent", nesting="sequential", steps=2)
        self.assertEqual(len(summary["val_energy"]), 2)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            self.run_center("center-bad", history_batch=1)
        with self.assertRaisesRegex(ValueError, "unknown layout"):
            self.run_center("center-bad", nesting="both")


class LandmarkTests(Exp8Fixture):
    def network(self, modes=2):
        tokens = self.table_tokens("train")
        torch.manual_seed(0)
        return build_network(
            self.inference, layout="landmark", encoder="gpt2-mean", modes=modes,
            teacher=self.teacher, landmark_ids=tokens["histories"][:3]), tokens

    def test_right_functions_start_as_the_landmark_rows_minus_their_mean(self):
        network, tokens = self.network()
        futures, headers = tokens["futures"], tokens["header_token_ids"]
        rows = self.teacher.grid_entries(tokens["histories"][:3], futures, headers)
        expected = (rows - rows.mean(dim=0))[:2].permute(1, 2, 0)
        self.assertTrue(torch.allclose(network.right_features(futures, headers), expected, atol=1e-5))
        self.assertEqual(network.left_features(tokens["histories"]).shape, (4, 2))
        trained = {name for name, p in network.named_parameters() if p.requires_grad}
        self.assertIn("mix", trained)
        self.assertFalse(any(name.startswith("teacher") for name in trained))
        with self.assertRaisesRegex(ValueError, "at least as many landmarks"):
            self.network(modes=4)

    def test_all_histories_as_landmarks_give_the_tables_own_column_means(self):
        tokens = self.table_tokens("train")
        network = build_network(
            self.inference, layout="landmark", encoder="gpt2-mean", modes=2,
            teacher=self.teacher, landmark_ids=tokens["histories"])
        table = load_mean_table(self.inference_dir / "train", self.teacher, "cpu")
        parts = landmark_mean(network, table, chunk=2)
        self.assertAlmostEqual(parts["mean_energy"], parts["ceiling"], places=5)
        self.assertAlmostEqual(parts["ceiling"] + parts["centered_share"], 1.0, places=9)
        self.assertAlmostEqual(parts["exact"][0], table.exact_rank1, places=9)

    def test_training_reports_the_energy_on_the_table_minus_g(self):
        summary = quiet(
            train_centered, inference_dir=self.inference_dir, stream_path=self.stream_path,
            model_path=self.model_path, output_dir=self.root / "landmark-run", device="cpu",
            layout="landmark", encoder="gpt2-mean", nesting="joint", modes=2, history_batch=4,
            future_batch=3, steps=2, eval_every=1, lr=1e-3, encoder_lr=1e-4, seed=0,
            overwrite=False, landmarks=3)
        self.assertEqual(summary["step"], 2)
        record = json.loads(
            (self.root / "landmark-run" / "eval_history.jsonl").read_text().splitlines()[-1])
        self.assertEqual(len(record["val"]["minus_g_energy"]), 2)
        self.assertEqual(len(record["val"]["minus_g_exact_energy"]), 2)


class PrototypeTests(Exp8Fixture):
    def test_prototypes_start_as_the_landmarks_and_are_trained(self):
        tokens = self.table_tokens("train")
        futures, headers = tokens["futures"], tokens["header_token_ids"]
        networks = {}
        for layout in ("landmark", "prototype"):
            torch.manual_seed(0)
            networks[layout] = build_network(
                self.inference, layout=layout, encoder="gpt2-mean", modes=2,
                teacher=self.teacher, landmark_ids=tokens["histories"][:3])
        prototype = networks["prototype"]
        self.assertTrue(torch.allclose(
            prototype.right_features(futures, headers),
            networks["landmark"].right_features(futures, headers), atol=1e-4))
        prototype.eval()  # the chunked pass without gradients gives the same entries
        prototype.chunk = 3
        self.assertTrue(torch.allclose(
            prototype.entries(futures, headers),
            self.teacher.grid_entries(tokens["histories"][:3], futures, headers), atol=1e-4))
        trained = {name for name, p in prototype.named_parameters() if p.requires_grad}
        self.assertIn("prototypes", trained)
        self.assertFalse(any(name.startswith("transformer.") for name in trained))
        copied = build_network(
            self.inference, layout="prototype", encoder="gpt2-mean", modes=2,
            landmark_ids=tokens["histories"][:3], train_copy=True)
        self.assertTrue(any(
            name.startswith("transformer.") and p.requires_grad
            for name, p in copied.named_parameters()))

    def test_training_moves_the_prototypes_and_reports_the_mean(self):
        summary = quiet(
            train_centered, inference_dir=self.inference_dir, stream_path=self.stream_path,
            model_path=self.model_path, output_dir=self.root / "prototype-run", device="cpu",
            layout="prototype", encoder="gpt2-mean", nesting="joint", modes=2, history_batch=4,
            future_batch=3, steps=2, eval_every=1, lr=1e-2, encoder_lr=1e-4, seed=0,
            overwrite=False, landmarks=3)
        self.assertEqual(summary["step"], 2)
        records = [json.loads(line) for line in
                   (self.root / "prototype-run" / "eval_history.jsonl").read_text().splitlines()]
        self.assertNotEqual(records[0]["val"]["mean_energy"], records[1]["val"]["mean_energy"])
        self.assertEqual(len(records[1]["val"]["minus_g_energy"]), 2)


class MeanTests(Exp8Fixture):
    def test_without_training_the_student_gives_the_frozen_models_entries(self):
        tokens = self.table_tokens("train")
        futures, headers = tokens["futures"], tokens["header_token_ids"]
        plain = MeanStudent(self.inference, None)
        self.assertTrue(torch.allclose(
            plain(futures, headers), self.teacher.future_only(futures, headers), atol=1e-5))
        prefix = tokens["histories"][0]
        student = MeanStudent(self.inference, prefix)
        row = self.teacher.grid_entries(prefix[None], futures, headers)[0]
        self.assertTrue(torch.allclose(student(futures, headers), row, atol=1e-4))
        self.assertEqual(student.prefix.shape, (HISTORY_TOKENS, 8))
        self.assertTrue(student.prefix.requires_grad)

    def test_the_generic_student_is_exp7s_right_side_plus_g(self):
        from experiments.exp7_independent_subnets.model import token_vectors
        from experiments.exp8_hypotheses.model import GenericMeanStudent, build_encoder
        tokens = self.table_tokens("train")
        futures, headers = tokens["futures"], tokens["header_token_ids"]
        vectors = token_vectors(self.inference.model.get_output_embeddings().weight)
        student = GenericMeanStudent(
            build_encoder(self.inference, "gpt2", None), vectors, self.teacher)
        w = student.head(student.encoder(futures))
        expected = torch.einsum("fe,fke->fk", w, vectors[headers]) / 8 ** 0.5
        self.assertTrue(torch.allclose(
            student(futures, headers) - self.teacher.future_only(futures, headers),
            expected, atol=1e-5))
        summary = quiet(
            train_mean, inference_dir=self.inference_dir, stream_path=self.stream_path,
            model_path=self.model_path, output_dir=self.root / "mean-generic", device="cpu",
            prefix_tokens=0, batch_size=4, steps=2, eval_every=1, lr=1e-3, encoder_lr=1e-4,
            seed=0, overwrite=False, generic="gpt2")
        self.assertEqual(summary["step"], 2)

    def test_the_tables_own_column_means_reach_the_ceiling(self):
        table = load_mean_table(self.inference_dir / "train", self.teacher, "cpu")

        class Oracle(torch.nn.Module):
            def forward(self, future_ids, header_ids):
                return (table.column_mean + table.g)[: len(future_ids)]

        comparison = mean_comparison(Oracle(), table, chunk=8)
        self.assertAlmostEqual(comparison["energy"], comparison["ceiling"], places=6)
        self.assertAlmostEqual(comparison["mean_r2"], 1.0, places=6)
        self.assertLessEqual(comparison["ceiling"], comparison["exact_rank1"] + 1e-9)

    def test_training_runs_with_and_without_a_prefix(self):
        for prefix_tokens in (0, HISTORY_TOKENS):
            with self.subTest(prefix_tokens=prefix_tokens):
                name = f"mean-{prefix_tokens}"
                summary = quiet(
                    train_mean, inference_dir=self.inference_dir, stream_path=self.stream_path,
                    model_path=self.model_path, output_dir=self.root / name, device="cpu",
                    prefix_tokens=prefix_tokens, batch_size=4, steps=2, eval_every=1, lr=1e-3,
                    encoder_lr=1e-4, seed=0, overwrite=False)
                self.assertEqual(summary["step"], 2)
                lines = (self.root / name / "eval_history.jsonl").read_text().splitlines()
                self.assertEqual([json.loads(line)["step"] for line in lines], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "do not fit"):
            quiet(
                train_mean, inference_dir=self.inference_dir, stream_path=self.stream_path,
                model_path=self.model_path, output_dir=self.root / "mean-long", device="cpu",
                prefix_tokens=8, batch_size=4, steps=2, eval_every=1, lr=1e-3,
                encoder_lr=1e-4, seed=0, overwrite=False)


class SpectrumRunTests(Exp8Fixture):
    def test_run_spectrum_reports_every_table_and_baseline(self):
        summary = quiet(
            run_spectrum, inference_dir=self.inference_dir, model_path=self.model_path,
            device="cpu", output_dir=self.root / "spectrum", seeds=1)
        self.assertEqual(summary["reports"], 6)
        reports = json.loads((self.root / "spectrum" / "spectrum.json").read_text())
        self.assertIn("val/minus-column-mean/seed0", reports)


if __name__ == "__main__":
    unittest.main()
