import contextlib
import io
import json
import math
import shutil
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch

from experiments.exp4_table_neural_svd.train import compare_with_exact_svd
from experiments.exp5_gpt2_mlp_table.features import (
    future_only_logits,
    header_tokens,
    last_hidden,
    pair_targets,
)
from experiments.exp5_gpt2_mlp_table.loss import paired_loss
from experiments.exp5_gpt2_mlp_table.train import draw_pairs
from experiments.exp5_gpt2_mlp_table.tables import load_eval_table
from experiments.exp7_independent_subnets.align import Moments, alignment, max_correlation
from experiments.exp7_independent_subnets.model import build_mode, token_vectors
from experiments.exp7_independent_subnets.stage import Selection, residual_target, stage_loss
from experiments.exp7_independent_subnets.train import (
    StagedRun,
    mode_table_features,
    selection_energy,
    train_exp7,
    train_step,
)
from utils.nestedlora import NestedLoRA
from test_exp5_gpt2_mlp_table import FUTURES, HISTORIES, TOP_K, VOCAB, Exp5Fixture

SCRATCH = dict(scratch_layers=1, scratch_width=4, scratch_heads=2)
SUBNETS = ("gpt2", "scratch")


def parameter_ids(module: torch.nn.Module) -> set[int]:
    return {id(p) for p in module.parameters()}


class ModeTests(Exp5Fixture):
    HEADER = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 16]])

    def vectors(self):
        return token_vectors(self.inference.model.get_output_embeddings().weight)

    def test_token_vectors_are_the_centered_unembedding_at_unit_scale(self):
        weight = self.inference.model.get_output_embeddings().weight
        vectors = token_vectors(weight)
        centered = weight.detach() - weight.detach().mean(dim=0)
        self.assertFalse(vectors.requires_grad)
        self.assertAlmostEqual(vectors.square().mean().item(), 1.0, places=5)
        torch.testing.assert_close(
            vectors * centered.square().mean().sqrt(), centered, atol=1e-6, rtol=0)

    def test_shapes_and_the_bilinear_right_side(self):
        for subnet in SUBNETS:
            mode = build_mode(self.inference, subnet, **SCRATCH)
            vectors = self.vectors()
            futures = torch.tensor(FUTURES)
            with torch.no_grad():
                left = mode.left_features(torch.tensor(HISTORIES))
                w = mode.right_vectors(futures)
                right = mode.right_features(futures, self.HEADER, vectors)
            self.assertEqual((left.shape, w.shape, right.shape), ((6,), (5, 8), (5, TOP_K)))
            expected = torch.einsum("fe,fke->fk", w, vectors[self.HEADER]) / 8 ** 0.5
            torch.testing.assert_close(right, expected, atol=1e-6, rtol=0)

    def test_gpt2_mode_starts_at_the_teacher_and_scratch_does_not(self):
        histories = torch.tensor(HISTORIES)
        with torch.no_grad():
            teacher = self.inference.model.base_model(
                input_ids=histories, use_cache=False).last_hidden_state[:, -1]
            gpt2 = build_mode(self.inference, "gpt2", **SCRATCH)
            for encoder in (gpt2.left_encoder, gpt2.right_encoder):
                hidden = encoder(input_ids=histories, use_cache=False).last_hidden_state[:, -1]
                torch.testing.assert_close(hidden, teacher, atol=1e-6, rtol=0)
            scratch = build_mode(self.inference, "scratch", **SCRATCH)
        teacher_ids = parameter_ids(self.inference.model)
        for encoder in (scratch.left_encoder, scratch.right_encoder):
            self.assertEqual(encoder.config.n_embd, 4)
            self.assertEqual(encoder.config.n_layer, 1)
            self.assertEqual(encoder.config.n_head, 2)
            self.assertEqual(encoder.config.vocab_size, VOCAB)
            self.assertEqual(encoder.config.n_positions, 8)
            self.assertEqual(
                (encoder.config.resid_pdrop, encoder.config.embd_pdrop, encoder.config.attn_pdrop),
                (0.0, 0.0, 0.0))
            self.assertFalse(parameter_ids(encoder) & teacher_ids)
            self.assertEqual(
                encoder.wte.weight.shape, self.inference.model.base_model.wte.weight.shape[:1] + (4,))
            self.assertFalse(torch.equal(
                encoder.wte.weight[:, :4], self.inference.model.base_model.wte.weight[:, :4]))
        self.assertEqual(scratch.left_head[1].in_features, 4)
        self.assertEqual(scratch.right_head[1].out_features, 8)

    def test_the_four_parts_of_a_mode_and_two_modes_share_no_parameter(self):
        for subnet in SUBNETS:
            mode_a = build_mode(self.inference, subnet, **SCRATCH)
            mode_b = build_mode(self.inference, subnet, **SCRATCH)
            groups = [
                parameter_ids(getattr(mode, name))
                for mode in (mode_a, mode_b)
                for name in ("left_encoder", "right_encoder", "left_head", "right_head")
            ] + [parameter_ids(self.inference.model)]
            for i, first in enumerate(groups):
                self.assertTrue(first)
                for second in groups[i + 1:]:
                    self.assertFalse(first & second, subnet)

    def test_each_side_reads_its_own_input_only(self):
        for subnet in SUBNETS:
            mode = build_mode(self.inference, subnet, **SCRATCH)
            vectors = self.vectors()
            mode.left_features(torch.tensor(HISTORIES)).sum().backward()
            for part in (mode.right_encoder, mode.right_head):
                self.assertTrue(all(p.grad is None for p in part.parameters()))
            self.assertTrue(any(p.grad is not None for p in mode.left_head.parameters()))
            mode.zero_grad()
            mode.right_features(torch.tensor(FUTURES), self.HEADER, vectors).sum().backward()
            for part in (mode.left_encoder, mode.left_head):
                self.assertTrue(all(p.grad is None for p in part.parameters()))
            self.assertTrue(any(p.grad is not None for p in mode.right_head.parameters()))
            # A header token enters only through vectors[z].
            vectors = vectors.clone()
            vectors[2] = 2 * vectors[0] - 3 * vectors[1]
            header = torch.tensor([[0, 1, 2]] * len(FUTURES))
            with torch.no_grad():
                right = mode.right_features(torch.tensor(FUTURES), header, vectors)
            torch.testing.assert_close(
                right[:, 2], 2 * right[:, 0] - 3 * right[:, 1], atol=1e-5, rtol=0)

    def test_unknown_subnet_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "subnet"):
            build_mode(self.inference, "mlp", **SCRATCH)


class StageTests(unittest.TestCase):
    def test_residual_removes_the_frozen_modes(self):
        generator = torch.Generator().manual_seed(0)
        target = torch.randn(4, 3, generator=generator)
        empty = residual_target(target, torch.zeros(4, 0), torch.zeros(4, 3, 0))
        torch.testing.assert_close(empty, target, atol=0, rtol=0)
        left, right = torch.randn(4, 2, generator=generator), torch.randn(4, 3, 2, generator=generator)
        expected = target.clone()
        for b in range(4):
            for j in range(3):
                for i in range(2):
                    expected[b, j] -= left[b, i] * right[b, j, i]
        residual = residual_target(target, left, right)
        self.assertEqual(residual.shape, (4, 3))
        torch.testing.assert_close(residual, expected, atol=1e-6, rtol=0)

    def test_stage_loss_is_the_rank_one_paired_loss(self):
        generator = torch.Generator().manual_seed(1)
        batch, top_k = 4, 3
        left = torch.randn(batch, generator=generator).requires_grad_()
        right = torch.randn(batch, top_k, generator=generator).requires_grad_()
        target = torch.randn(batch, top_k, generator=generator)
        ours = stage_loss(left, right, target)
        ours.backward()
        left_ref = left.detach().clone()[:, None].requires_grad_()
        right_ref = right.detach().clone().reshape(-1, 1).requires_grad_()
        model = NestedLoRA(torch.nn.Identity(), neigs=1)
        ref = paired_loss(model, left_ref, right_ref, target)
        ref.backward()
        torch.testing.assert_close(ours.detach(), ref.detach(), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(left.grad, left_ref.grad[:, 0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(right.grad, right_ref.grad.reshape(batch, top_k), atol=1e-6, rtol=1e-5)

    def test_stage_gradient_is_the_sequentially_nested_one(self):
        generator = torch.Generator().manual_seed(2)
        modes, top_k = 3, 5
        left = torch.randn(1, modes, generator=generator).requires_grad_()
        right = torch.randn(top_k, modes, generator=generator).requires_grad_()
        target = torch.randn(1, top_k, generator=generator)
        model = NestedLoRA(torch.nn.Identity(), neigs=modes, sequential=True)
        paired_loss(model, left, right, target).backward()
        for l in range(modes):
            left_l = left.detach().clone().requires_grad_()
            right_l = right.detach().clone().requires_grad_()
            residual = residual_target(target, left_l[:, :l], right_l[None, :, :l])
            stage_loss(left_l[:, l], right_l[:, l][None], residual).backward()
            torch.testing.assert_close(left.grad[:, l], left_l.grad[:, l], atol=1e-6, rtol=0)
            torch.testing.assert_close(right.grad[:, l], right_l.grad[:, l], atol=1e-6, rtol=0)

    def test_single_pair_and_single_header_are_finite(self):
        left = torch.tensor([0.7], requires_grad=True)
        right = torch.tensor([[-1.3]], requires_grad=True)
        loss = stage_loss(left, right, torch.tensor([[0.4]]))
        loss.backward()
        for value in (loss, left.grad, right.grad):
            self.assertTrue(torch.isfinite(value).all())

    def test_selection_keeps_the_best_and_counts_stale_evaluations(self):
        s = Selection(max_steps=100, min_steps=30, patience=2)
        self.assertEqual([s.record(m, t) for m, t in ((0.5, 10), (0.9, 20), (0.1, 30), (0.1, 40))],
                         [True, True, False, False])
        self.assertEqual((s.best, s.best_step, s.stale), (0.9, 20, 2))
        self.assertEqual(s.reason(40), "patience")

    def test_selection_waits_for_min_steps_and_stops_at_max_steps(self):
        s = Selection(max_steps=100, min_steps=50, patience=1)
        s.record(0.9, 10)
        s.record(0.1, 20)
        self.assertEqual(s.stale, 1)
        self.assertIsNone(s.reason(49))
        self.assertEqual(s.reason(50), "patience")
        self.assertTrue(s.record(0.95, 99))
        self.assertIsNone(s.reason(99))
        self.assertEqual(s.reason(100), "steps")

    def test_patience_zero_is_a_fixed_number_of_steps(self):
        s = Selection(max_steps=4, min_steps=0, patience=0)
        self.assertFalse(s.keeps_best)
        self.assertFalse(s.record(0.9, 2))
        self.assertEqual((s.reason(2), s.reason(4)), (None, "steps"))


class AlignmentTests(unittest.TestCase):
    N, C = 12, 20
    S = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5], dtype=torch.float64)

    def setUp(self):
        torch.manual_seed(0)
        f64 = dict(dtype=torch.float64)
        self.U = torch.linalg.qr(torch.randn(self.N, 5, **f64))[0]
        self.V = torch.linalg.qr(torch.randn(self.C, 5, **f64))[0]
        self.A = (self.U * self.S) @ self.V.T

    def moments(self, F, G):
        return (F.T @ F / self.N, G.T @ G / self.C, F.T @ self.A @ G / (self.N * self.C))

    def projected(self, F, G):
        def projector(X):
            return X @ torch.linalg.pinv(X)
        return projector(F) @ self.A @ projector(G).T

    def test_alignment_recovers_the_truncated_svd_from_a_mixed_basis(self):
        s = self.S
        F = self.U[:, :3] @ torch.randn(3, 3, dtype=torch.float64)
        G = self.V[:, :3] @ torch.randn(3, 3, dtype=torch.float64)
        T_f, T_g, S = alignment(*self.moments(F, G))
        torch.testing.assert_close(
            (F @ T_f) @ (G @ T_g).T, (self.U[:, :3] * s[:3]) @ self.V[:, :3].T, atol=1e-8, rtol=0)
        torch.testing.assert_close(S, s[:3] / (self.N * self.C) ** 0.5, atol=1e-10, rtol=0)
        torch.testing.assert_close(
            (F @ T_f).T @ (F @ T_f) / self.N, torch.diag(S), atol=1e-10, rtol=0)
        torch.testing.assert_close(
            (G @ T_g).T @ (G @ T_g) / self.C, torch.diag(S), atol=1e-10, rtol=0)

    def test_alignment_is_the_least_squares_fit_in_the_span(self):
        F = torch.randn(self.N, 3, dtype=torch.float64)
        G = torch.randn(self.C, 3, dtype=torch.float64)
        T_f, T_g, _ = alignment(*self.moments(F, G))
        torch.testing.assert_close(
            (F @ T_f) @ (G @ T_g).T, self.projected(F, G), atol=1e-8, rtol=0)

    def test_one_mode_is_rescaled(self):
        F = torch.randn(self.N, 1, dtype=torch.float64)
        G = torch.randn(self.C, 1, dtype=torch.float64)
        lam_f, lam_g, cross = self.moments(F, G)
        T_f, T_g, _ = alignment(lam_f, lam_g, cross)
        torch.testing.assert_close(T_f * T_g, cross / (lam_f * lam_g), atol=1e-10, rtol=1e-10)

    def test_collinear_and_zero_modes_give_finite_transforms(self):
        F = torch.randn(self.N, 3, dtype=torch.float64)
        F[:, 1] = 2 * F[:, 0]
        G = torch.randn(self.C, 3, dtype=torch.float64)
        G[:, 2] = 0
        T_f, T_g, S = alignment(*self.moments(F, G))
        for out in (T_f, T_g, S):
            self.assertTrue(torch.isfinite(out).all())
        self.assertAlmostEqual(S[-1].item(), 0.0, delta=1e-12)
        torch.testing.assert_close(
            (F @ T_f) @ (G @ T_g).T, self.projected(F, G), atol=1e-8, rtol=0)

    def test_moments_match_their_definitions_in_any_chunking(self):
        P, K, L = 6, 3, 2
        left = torch.randn(P, L)
        right = torch.randn(P, K, L)
        target = torch.randn(P, K)
        l64, r64, t64 = left.double(), right.double(), target.double()
        lam_f = torch.stack([torch.outer(f, f) for f in l64]).mean(0)
        lam_g = torch.stack([torch.outer(g, g) for p in r64 for g in p]).mean(0)
        cross = torch.zeros(L, L, dtype=torch.float64)
        for p in range(P):
            for i in range(L):
                for j in range(L):
                    cross[i, j] += l64[p, i] * (t64[p] * r64[p, :, j]).mean() / P
        whole = Moments(L)
        whole.add(left, right, target)
        chunked = Moments(L)
        chunked.add(left[:4], right[:4], target[:4])
        chunked.add(left[4:], right[4:], target[4:])
        for moments in (whole, chunked):
            self.assertEqual(moments.lam_f.dtype, torch.float64)
            torch.testing.assert_close(moments.lam_f, lam_f, atol=1e-12, rtol=0)
            torch.testing.assert_close(moments.lam_g, lam_g, atol=1e-12, rtol=0)
            torch.testing.assert_close(moments.cross, cross, atol=1e-12, rtol=0)

    def test_max_correlation(self):
        self.assertEqual(max_correlation(torch.tensor([[4.0]])), 0.0)
        self.assertAlmostEqual(
            max_correlation(torch.tensor([[4.0, -1.0], [-1.0, 1.0]])), 0.5)

    def test_a_zero_mode_has_correlation_zero(self):
        self.assertEqual(max_correlation(torch.tensor([[0.0, 0.0], [0.0, 1.0]])), 0.0)
        gram = torch.tensor([[4.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]])
        self.assertAlmostEqual(max_correlation(gram), 0.5)


class TrainingHelpers:
    """Runs train_exp7 on the Exp5Fixture and reads its output files."""

    def train(self, output_dir, **changes):
        options = dict(
            inference_dir=self.inference_dir, store_dir=self.store_dir,
            model_path=self.model_path, output_dir=output_dir, device="cpu", subnet="gpt2",
            modes=2, stage_steps=4, min_stage_steps=0, patience=0, align_pairs=32,
            **SCRATCH, batch_size=8, eval_every=2, lr=1e-2, encoder_lr=1e-3, seed=0,
            overwrite=False,
        )
        options.update(changes)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            try:
                return train_exp7(**options)
            finally:
                self.log = output.getvalue()

    def records(self, output_dir):
        lines = (output_dir / "eval_history.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]

    def stages(self, output_dir):
        lines = (output_dir / "stages.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]


class TrainingTests(TrainingHelpers, Exp5Fixture):
    def vectors(self):
        return token_vectors(self.inference.model.get_output_embeddings().weight)

    def load_mode(self, path, subnet="gpt2"):
        mode = build_mode(self.inference, subnet, **SCRATCH)
        mode.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
        return mode

    def batch(self):
        histories, futures = torch.tensor(HISTORIES[:5]), torch.tensor(FUTURES)
        future_hidden = last_hidden(self.inference, futures)
        header_ids, g = header_tokens(future_only_logits(self.inference, future_hidden), TOP_K)
        pair_logits = self.inference.next_token_logits_batch(
            torch.cat([histories, futures], dim=1).tolist())
        return histories, futures, header_ids, pair_targets(pair_logits, header_ids, g)

    def test_train_step_updates_only_the_current_mode(self):
        histories, futures, header_ids, target = self.batch()
        vectors = self.vectors()
        for subnet in SUBNETS:
            with self.subTest(subnet=subnet):
                frozen = [build_mode(self.inference, subnet, **SCRATCH) for _ in range(2)]
                for mode in frozen:
                    mode.requires_grad_(False).eval()
                current = build_mode(self.inference, subnet, **SCRATCH)
                frozen_before = [
                    {name: p.detach().clone() for name, p in mode.named_parameters()}
                    for mode in frozen]
                current_before = {
                    name: p.detach().clone() for name, p in current.named_parameters()}
                optimizer = torch.optim.Adam(current.parameters(), lr=1e-2)
                loss, residual_energy = train_step(
                    current, frozen, vectors, optimizer, histories, futures, header_ids, target)
                self.assertTrue(math.isfinite(loss))
                for mode, before in zip(frozen, frozen_before):
                    for name, parameter in mode.named_parameters():
                        self.assertTrue(torch.equal(parameter, before[name]), name)
                        self.assertIsNone(parameter.grad, name)
                for part in ("left_encoder", "right_encoder", "left_head", "right_head"):
                    self.assertTrue(any(
                        not torch.equal(p, current_before[f"{part}.{name}"])
                        for name, p in getattr(current, part).named_parameters()), part)
                with torch.no_grad():
                    frozen_left = torch.stack([m.left_features(histories) for m in frozen], 1)
                    frozen_right = torch.stack(
                        [m.right_features(futures, header_ids, vectors) for m in frozen], 2)
                expected = residual_target(target, frozen_left, frozen_right).square().mean()
                self.assertAlmostEqual(residual_energy, expected.item(), places=6)
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    train_step(current, frozen, vectors, optimizer, histories, futures,
                               header_ids, torch.full_like(target, float("nan")))

    def test_writes_outputs_for_two_stages(self):
        output = self.root / "exp7-outputs"
        summary = self.train(output)
        self.assertEqual(
            summary, {"output_dir": str(output), "step": 8, "modes": 2, "evaluations": 4})
        records = self.records(output)
        self.assertEqual([(r["stage"], r["stage_step"], r["step"]) for r in records],
                         [(1, 2, 2), (1, 4, 4), (2, 2, 6), (2, 4, 8)])
        self.assertEqual(set(records[0]), {
            "step", "stage", "stage_step", "train_loss", "kernel_energy", "residual_energy",
            "seconds_per_step", "selection", "report"})
        self.assertEqual([r["selection"]["ranks"] for r in records], [[1], [1], [1, 2], [1, 2]])
        # Stage 1 trains on the target itself.
        self.assertAlmostEqual(records[0]["residual_energy"], records[0]["kernel_energy"], places=6)
        stages = self.stages(output)
        self.assertEqual([(s["stage"], s["steps"], s["best_step"], s["reason"]) for s in stages],
                         [(1, 4, 4, "steps"), (2, 4, 4, "steps")])
        self.assertEqual(set(stages[1]), {
            "stage", "steps", "best_step", "reason", "selection_energy", "selection", "report",
            "aligned", "singular_values", "left_correlation", "right_correlation"})
        self.assertEqual(set(stages[1]["aligned"]), {"selection", "report"})
        self.assertEqual(stages[1]["aligned"]["selection"]["ranks"], [1, 2])
        self.assertEqual(len(stages[1]["singular_values"]), 2)
        # With patience 0 the frozen weights are the last ones, so the stage line repeats
        # the stage's last evaluation.
        self.assertEqual(stages[1]["selection"], records[-1]["selection"])
        self.assertEqual(stages[1]["selection_energy"], selection_energy(records[-1]))
        for stage in stages:
            for key in ("left_correlation", "right_correlation"):
                self.assertTrue(0.0 <= stage[key] <= 1.0 + 1e-9)
        self.assertEqual(sorted(path.name for path in output.iterdir()), [
            "alignment.pt", "checkpoint.pt", "eval_history.jsonl", "metadata.json", "modes",
            "stages.jsonl"])
        self.assertEqual(
            sorted(path.name for path in (output / "modes").iterdir()),
            ["mode-01.pt", "mode-02.pt"])
        mode_file = torch.load(output / "modes" / "mode-02.pt", weights_only=True)
        self.assertEqual((mode_file["stage"], mode_file["best_step"]), (2, 4))
        self.assertEqual(
            set(mode_file["state_dict"]),
            set(build_mode(self.inference, "gpt2", **SCRATCH).state_dict()))
        aligned = torch.load(output / "alignment.pt", weights_only=True)
        self.assertEqual(aligned["T_f"].shape, (2, 2))
        self.assertEqual(aligned["T_g"].shape, (2, 2))
        self.assertEqual(aligned["modes"], 2)
        self.assertEqual(aligned["S"].tolist(), stages[1]["singular_values"])
        checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
        self.assertEqual(
            set(checkpoint), {"config", "completed_modes", "step", "history", "stage"})
        self.assertEqual(
            (checkpoint["completed_modes"], checkpoint["step"], checkpoint["stage"]), (2, 8, None))
        self.assertEqual(checkpoint["history"], records)
        for key in ("modes", "device", "overwrite", "encode_batch", "eval_chunk"):
            self.assertNotIn(key, checkpoint["config"])
        self.assertEqual(
            (checkpoint["config"]["subnet"], checkpoint["config"]["stage_steps"]), ("gpt2", 4))
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["table_shapes"], {"selection": [4, 9], "report": [2, 6]})
        self.assertEqual(metadata["ranks"], "1..l at stage l")
        self.assertEqual(
            {key: metadata["tables"][key]["exp4_table"] for key in ("selection", "report")},
            {"selection": "train", "report": "val"})
        self.assertEqual(metadata["top_k"], TOP_K)
        self.assertEqual(
            metadata["trainable_parameters"],
            sum(p.numel() for p in build_mode(self.inference, "gpt2", **SCRATCH).parameters()))
        # No file holds the teacher's token vectors or the frozen model.
        vectors = self.vectors()
        teacher = set(self.inference.model.state_dict())
        for path in [*output.glob("*.pt"), *(output / "modes").glob("*.pt")]:
            stored = torch.load(path, weights_only=True)
            tensors = [stored] if isinstance(stored, torch.Tensor) else []
            pending = [stored]
            while pending:
                value = pending.pop()
                if isinstance(value, dict):
                    self.assertFalse(teacher & set(value), path)
                    self.assertFalse(any("vectors" in str(key) for key in value), path)
                    pending.extend(value.values())
                elif isinstance(value, (list, tuple)):
                    pending.extend(value)
                elif isinstance(value, torch.Tensor):
                    tensors.append(value)
            for tensor in tensors:
                self.assertFalse(
                    tensor.shape == vectors.shape and torch.allclose(tensor, vectors), path)
        self.assertIn("stage 1", self.log)
        self.assertIn("stage 2 finished", self.log)

    def test_selection_table_is_exp4_train_and_report_table_is_exp4_val(self):
        output = self.root / "exp7-tables"
        self.train(output)
        record = self.records(output)[-1]
        modes = [self.load_mode(output / "modes" / f"mode-0{stage}.pt") for stage in (1, 2)]
        vectors = self.vectors()
        for exp4_name, key in (("train", "selection"), ("val", "report")):
            table = load_eval_table(self.inference_dir / exp4_name, self.inference, "cpu")
            features = [mode_table_features(mode, table, vectors, chunk=2) for mode in modes]
            left = torch.stack([f[0] for f in features], dim=1)
            right = torch.stack([f[1] for f in features], dim=1)
            self.assertEqual((left.dtype, right.dtype), (torch.float64, torch.float64))
            self.assertEqual(
                (left.shape, right.shape), ((len(table.table.histories), 2),
                                            (len(table.table.columns), 2)))
            expected = compare_with_exact_svd(left, right, table.table, ranks=(1, 2))
            self.assertEqual(set(expected), set(record[key]))
            for name, values in expected.items():
                # chunk 2 here and 256 in the run: equal up to float32 rounding.
                np.testing.assert_allclose(record[key][name], values, rtol=1e-6, atol=1e-6)

    def test_mode_table_features_follow_the_column_order(self):
        table = load_eval_table(self.inference_dir / "train", self.inference, "cpu")
        mode = build_mode(self.inference, "gpt2", **SCRATCH)
        vectors = self.vectors()
        left, right = mode_table_features(mode, table, vectors, chunk=2)
        self.assertFalse(mode.training)
        futures = table.future_ids
        with torch.no_grad():
            expected_left = mode.left_features(table.table.histories)
            for column, tokens in enumerate(table.table.columns):
                f, j = divmod(column, TOP_K)
                self.assertTrue(torch.equal(tokens[:-1], futures[f]))
                expected = mode.right_features(futures[f:f + 1], tokens[None, -1:], vectors)
                self.assertAlmostEqual(right[column].item(), expected.item(), places=5)
        torch.testing.assert_close(left.float(), expected_left, atol=1e-6, rtol=0)

    def test_third_stage_selects_on_rank_three(self):
        output = self.root / "exp7-three"
        self.train(output, modes=3)
        records = self.records(output)
        third = [r for r in records if r["stage"] == 3]
        self.assertEqual(len(third), 2)
        for record in third:
            self.assertEqual(record["selection"]["ranks"], [1, 2, 3])
            self.assertEqual(selection_energy(record), record["selection"]["captured_energy"][2])
            # The frozen modes 1 and 2 do not move during stage 3.
            np.testing.assert_allclose(
                record["selection"]["captured_energy"][:2],
                [r for r in records if r["stage"] == 2][-1]["selection"]["captured_energy"],
                rtol=0, atol=1e-9)
        # The selection metric follows the mode being trained.
        self.assertNotEqual(selection_energy(third[0]), selection_energy(third[1]))

    def test_last_step_of_a_stage_is_evaluated(self):
        output = self.root / "exp7-last-a"
        self.train(output, modes=1, stage_steps=3, eval_every=2)
        self.assertEqual([r["stage_step"] for r in self.records(output)], [2, 3])
        output = self.root / "exp7-last-b"
        self.train(output, modes=1, stage_steps=3, eval_every=5)
        self.assertEqual([r["stage_step"] for r in self.records(output)], [3])
        self.assertEqual(self.stages(output)[0]["steps"], 3)

    def test_patience_freezes_the_best_weights_and_patience_zero_the_last(self):
        a, b, c = (self.root / f"exp7-patience-{name}" for name in "abc")
        self.train(a, modes=1, stage_steps=2)
        self.train(b, modes=1, stage_steps=4)
        with patch("experiments.exp7_independent_subnets.train.selection_energy",
                   side_effect=[0.9, 0.1]):
            summary = self.train(c, modes=1, stage_steps=6, patience=1)
        self.assertEqual(summary["step"], 4)
        self.assertEqual([r["stage_step"] for r in self.records(c)], [2, 4])
        stage = self.stages(c)[0]
        self.assertEqual(
            (stage["steps"], stage["best_step"], stage["reason"], stage["selection_energy"]),
            (4, 2, "patience", 0.9))
        weights = {
            name: torch.load(path / "modes" / "mode-01.pt", weights_only=True)
            for name, path in (("a", a), ("b", b), ("c", c))}
        self.assertEqual(weights["c"]["best_step"], 2)
        self.assertEqual(weights["b"]["best_step"], 4)
        for key, value in weights["a"]["state_dict"].items():
            self.assertTrue(torch.equal(weights["c"]["state_dict"][key], value), key)
        self.assertTrue(any(
            not torch.equal(weights["b"]["state_dict"][key], value)
            for key, value in weights["a"]["state_dict"].items()))

    def test_scratch_subnet_trains_with_its_own_small_transformers(self):
        output = self.root / "exp7-scratch"
        summary = self.train(output, subnet="scratch")
        self.assertEqual(
            summary, {"output_dir": str(output), "step": 8, "modes": 2, "evaluations": 4})
        self.assertEqual([(s["stage"], s["reason"]) for s in self.stages(output)],
                         [(1, "steps"), (2, "steps")])
        state = torch.load(output / "modes" / "mode-01.pt", weights_only=True)["state_dict"]
        for side in ("left_encoder", "right_encoder"):
            self.assertEqual(state[f"{side}.wte.weight"].shape, (VOCAB, 4))
            self.assertEqual(state[f"{side}.wpe.weight"].shape[1], 4)
        self.assertEqual(state["left_head.1.weight"].shape, (1, 4))
        self.assertEqual(state["right_head.1.weight"].shape, (8, 4))
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(
            {key: metadata[key] for key in (
                "subnet", "scratch_layers", "scratch_width", "scratch_heads")},
            {"subnet": "scratch", "scratch_layers": 1, "scratch_width": 4, "scratch_heads": 2})
        self.assertIn("linear warmup over the first 1000 steps", self.log)

    def test_two_stages_learn_and_the_second_starts_from_a_smaller_residual(self):
        output = self.root / "exp7-learns"
        self.train(output, stage_steps=300, eval_every=100, batch_size=16, align_pairs=256)
        records = self.records(output)
        first = [r for r in records if r["stage"] == 1][-1]
        # An untrained mode scores at or below zero; the exact value is 1.
        for key in ("selection", "report"):
            self.assertGreater(first[key]["captured_energy"][0], 0.5, key)
        second = [r for r in records if r["stage"] == 2]
        self.assertEqual(len(second), 3)
        for record in second:
            self.assertLess(record["residual_energy"], record["kernel_energy"])
        aligned = self.stages(output)[1]["aligned"]["selection"]
        self.assertTrue(math.isfinite(aligned["captured_energy"][1]))
        self.assertLessEqual(
            aligned["captured_energy"][1], aligned["exact_captured_energy"][1] + 1e-6)

    def test_rejects_missing_tables_a_negative_seed_and_another_store(self):
        with self.assertRaisesRegex(ValueError, "exp4 inference"):
            self.train(self.root / "exp7-none", inference_dir=self.root / "absent")
        with self.assertRaises(ValueError):
            self.train(self.root / "exp7-seed", seed=-1)
        store = self.root / "exp7-store-other-model"
        shutil.copytree(self.store_dir, store)
        config = json.loads((store / "store.json").read_text())
        config["model_files"][next(iter(config["model_files"]))]["size"] += 1
        (store / "store.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "model"):
            self.train(self.root / "exp7-other-model", store_dir=store)
        with self.assertRaisesRegex(ValueError, "subnet"):
            self.train(self.root / "exp7-subnet", subnet="mlp")
        for change in ({"modes": 0}, {"stage_steps": 0}, {"eval_every": 0},
                       {"align_pairs": 0}, {"patience": -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.train(self.root / "exp7-bad", **change)
        self.assertFalse((self.root / "exp7-bad").exists())

    def test_a_finished_stage_releases_its_optimizer_and_best_weights(self):
        # Adam's moments of a gpt2 mode are as large as two copies of the model: they
        # must be gone before the next mode is built.
        finished = []
        released = []
        finish_stage, start_stage = StagedRun.finish_stage, StagedRun.start_stage

        def recording_finish(run, state, reason):
            finished.append([
                weakref.ref(state.optimizer),
                weakref.ref(next(iter(state.best_state_dict.values())))])
            finish_stage(run, state, reason)

        def checking_start(run, stage, *args, **kwargs):
            released.append([ref() is None for refs in finished for ref in refs])
            return start_stage(run, stage, *args, **kwargs)

        with patch.object(StagedRun, "finish_stage", recording_finish), \
                patch.object(StagedRun, "start_stage", checking_start):
            self.train(self.root / "exp7-release", modes=3, patience=2)
        self.assertEqual(released, [[], [True, True], [True, True, True, True]])


class ResumeTests(TrainingHelpers, Exp5Fixture):
    def interrupted(self, output_dir, records):
        """Raise KeyboardInterrupt at the first draw of pairs once `records` evaluations exist.

        Counting the draws would be fragile: the alignment draws pairs too.
        """
        history = output_dir / "eval_history.jsonl"

        def draw(sampler, count):
            if history.exists() and len(history.read_text().splitlines()) >= records:
                raise KeyboardInterrupt
            return draw_pairs(sampler, count)

        stack = contextlib.ExitStack()
        stack.enter_context(
            patch("experiments.exp7_independent_subnets.train.draw_pairs", side_effect=draw))
        stack.enter_context(self.assertRaises(KeyboardInterrupt))
        return stack

    def assert_same_weights(self, first, second):
        first = torch.load(first, weights_only=True)["state_dict"]
        second = torch.load(second, weights_only=True)["state_dict"]
        self.assertEqual(set(first), set(second))
        for key, value in first.items():
            self.assertTrue(torch.equal(second[key], value), key)

    def mtimes(self, output_dir):
        return {path: path.stat().st_mtime_ns for path in output_dir.rglob("*")}

    def test_resume_in_the_middle_of_a_stage(self):
        output = self.root / "exp7-resume-middle"
        options = dict(modes=2, stage_steps=6, eval_every=2, patience=3)
        with self.interrupted(output, 5):
            self.train(output, **options)
        checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
        self.assertEqual(
            (checkpoint["completed_modes"], checkpoint["step"], checkpoint["stage"]["stage_step"]),
            (1, 10, 4))
        first_mode = (output / "modes" / "mode-01.pt").read_bytes()
        summary = self.train(output, **options)
        self.assertIn("Resuming stage 2 at stage step 4 (step 10)", self.log)
        self.assertNotIn("stage 1", self.log.lower())
        self.assertEqual((output / "modes" / "mode-01.pt").read_bytes(), first_mode)
        records = self.records(output)
        self.assertEqual([r["step"] for r in records], [2, 4, 6, 8, 10, 12])
        self.assertEqual([(r["stage"], r["stage_step"]) for r in records[3:]],
                         [(2, 2), (2, 4), (2, 6)])
        self.assertEqual([(s["stage"], s["steps"]) for s in self.stages(output)],
                         [(1, 6), (2, 6)])
        self.assertEqual(
            summary, {"output_dir": str(output), "step": 12, "modes": 2, "evaluations": 6})
        checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
        self.assertEqual((checkpoint["completed_modes"], checkpoint["stage"]), (2, None))
        self.assertEqual(checkpoint["history"], records)

    def test_a_resumed_scratch_stage_continues_its_warmup(self):
        output = self.root / "exp7-resume-scratch"
        options = dict(modes=1, subnet="scratch", stage_steps=8)
        with self.interrupted(output, 2):
            self.train(output, **options)
        with self.interrupted(output, 3):
            self.train(output, **options)
        self.assertIn("Resuming stage 1 at stage step 4 (step 4)", self.log)
        stage = torch.load(output / "checkpoint.pt", weights_only=True)["stage"]
        self.assertEqual(stage["stage_step"], 6)
        self.assertEqual(stage["scheduler_state_dict"]["last_epoch"], 6)
        self.assertAlmostEqual(
            stage["optimizer_state_dict"]["param_groups"][0]["lr"], 1e-2 * 7 / 1000)
        self.assertEqual(self.train(output, **options)["step"], 8)

    def test_resume_restores_the_best_weights_and_the_stale_count(self):
        output = self.root / "exp7-resume-best"
        plain = self.root / "exp7-resume-best-plain"
        options = dict(modes=1, stage_steps=8, eval_every=2, patience=2)
        with patch("experiments.exp7_independent_subnets.train.selection_energy",
                   side_effect=[0.9, 0.1]), self.interrupted(output, 2):
            self.train(output, **options)
        with patch("experiments.exp7_independent_subnets.train.selection_energy",
                   return_value=0.1):
            summary = self.train(output, **options)
        self.assertIn("Resuming stage 1 at stage step 4 (step 4)", self.log)
        self.assertEqual(summary["step"], 6)
        self.assertEqual([r["stage_step"] for r in self.records(output)], [2, 4, 6])
        stage = self.stages(output)[0]
        self.assertEqual(
            (stage["steps"], stage["best_step"], stage["reason"], stage["selection_energy"]),
            (6, 2, "patience", 0.9))
        self.train(plain, modes=1, stage_steps=2, eval_every=2, patience=0)
        self.assert_same_weights(plain / "modes" / "mode-01.pt", output / "modes" / "mode-01.pt")
        self.assertEqual(
            torch.load(output / "modes" / "mode-01.pt", weights_only=True)["best_step"], 2)

    def test_resume_after_the_last_evaluation_of_a_stage(self):
        # Interrupted while the stage is being frozen: in the alignment, or while the
        # checkpoint after the stage line is written. The stage is not trained further.
        plain = self.root / "exp7-resume-last-plain"
        self.train(plain, modes=1)
        write_checkpoint = StagedRun.write_checkpoint

        def failing_write(run, state):
            if state is None:
                raise KeyboardInterrupt
            write_checkpoint(run, state)

        for name in ("alignment", "checkpoint"):
            with self.subTest(name=name):
                output = self.root / f"exp7-resume-last-{name}"
                if name == "alignment":
                    with self.interrupted(output, 2):
                        self.train(output, modes=1)
                else:
                    with patch.object(StagedRun, "write_checkpoint", failing_write), \
                            self.assertRaises(KeyboardInterrupt):
                        self.train(output, modes=1)
                summary = self.train(output, modes=1)
                self.assertIn("Resuming stage 1 at stage step 4 (step 4)", self.log)
                self.assertEqual(
                    summary, {"output_dir": str(output), "step": 4, "modes": 1, "evaluations": 2})
                self.assertEqual([r["step"] for r in self.records(output)], [2, 4])
                self.assertEqual([(s["stage"], s["steps"]) for s in self.stages(output)],
                                 [(1, 4)])
                self.assert_same_weights(
                    plain / "modes" / "mode-01.pt", output / "modes" / "mode-01.pt")

    def test_adding_modes_keeps_the_frozen_ones(self):
        output = self.root / "exp7-add"
        self.train(output, modes=1)
        first_mode = (output / "modes" / "mode-01.pt").read_bytes()
        summary = self.train(output, modes=3)
        self.assertEqual((output / "modes" / "mode-01.pt").read_bytes(), first_mode)
        self.assertNotIn("stage 1", self.log.lower())
        self.assertNotIn("Resuming", self.log)
        self.assertIn("Stage 2: mode 2 on the target minus 1 frozen mode(s)", self.log)
        records = self.records(output)
        self.assertEqual([r["step"] for r in records], [2, 4, 6, 8, 10, 12])
        self.assertEqual([r["stage"] for r in records], [1, 1, 2, 2, 3, 3])
        self.assertEqual([s["stage"] for s in self.stages(output)], [1, 2, 3])
        self.assertEqual(
            summary, {"output_dir": str(output), "step": 12, "modes": 3, "evaluations": 6})
        self.assertEqual(
            sorted(path.name for path in (output / "modes").iterdir()),
            ["mode-01.pt", "mode-02.pt", "mode-03.pt"])
        self.assertEqual(json.loads((output / "metadata.json").read_text())["modes"], 3)

    def test_modes_below_and_equal_to_the_finished_count(self):
        output = self.root / "exp7-done"
        summary = self.train(output, modes=2)
        with self.assertRaisesRegex(ValueError, "holds 2 finished modes.*--modes 1"):
            self.train(output, modes=1)
        before = self.mtimes(output)
        # Nothing expensive either: the model and the tables are not loaded.
        with patch("experiments.exp7_independent_subnets.train.prepare_run",
                   side_effect=AssertionError("prepare_run")):
            self.assertEqual(self.train(output, modes=2), summary)
        self.assertEqual(self.mtimes(output), before)

    def test_modes_equal_to_the_finished_count_with_a_stage_in_progress(self):
        output = self.root / "exp7-in-progress"
        with self.interrupted(output, 3):
            self.train(output, modes=2)
        before = self.mtimes(output)
        with self.assertRaisesRegex(ValueError, "stage 2 is in progress.*--modes 2"):
            self.train(output, modes=1)
        self.assertEqual(self.mtimes(output), before)

    def test_resume_needs_the_same_settings(self):
        output = self.root / "exp7-settings"
        self.train(output, modes=1)
        before = self.mtimes(output)
        for change in ({"subnet": "scratch"}, {"stage_steps": 6}, {"patience": 1},
                       {"lr": 1e-3}, {"align_pairs": 16}, {"scratch_width": 8}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "--overwrite"):
                self.train(output, modes=2, **change)
        self.assertEqual(self.mtimes(output), before)
        self.train(output, modes=2)
        summary = self.train(output, modes=1, stage_steps=2, overwrite=True)
        self.assertIn("Stage 1: mode 1", self.log)
        self.assertEqual(
            summary, {"output_dir": str(output), "step": 2, "modes": 1, "evaluations": 1})
        self.assertEqual([path.name for path in (output / "modes").iterdir()], ["mode-01.pt"])
        self.assertEqual([r["step"] for r in self.records(output)], [2])
        self.assertEqual(len(self.stages(output)), 1)
        # Without a checkpoint, an existing eval_history.jsonl is not resumed.
        history_only = self.root / "exp7-history-only"
        history_only.mkdir()
        shutil.copy(output / "eval_history.jsonl", history_only)
        with self.assertRaises(FileExistsError):
            self.train(history_only, modes=1, stage_steps=2)

    def test_a_missing_mode_file_is_reported(self):
        output = self.root / "exp7-missing"
        self.train(output, modes=2)
        (output / "modes" / "mode-01.pt").unlink()
        with self.assertRaisesRegex(ValueError, "mode-01.pt"):
            self.train(output, modes=3)
