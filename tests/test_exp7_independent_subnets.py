import unittest

import torch

from experiments.exp5_gpt2_mlp_table.loss import paired_loss
from experiments.exp7_independent_subnets.model import build_mode, token_vectors
from experiments.exp7_independent_subnets.stage import Selection, residual_target, stage_loss
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
