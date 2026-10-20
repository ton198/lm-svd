import tempfile
import unittest
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from utils import CausalLMInference


class CausalLMInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls._temporary_directory.name) / "model"
        config = GPT2Config(
            vocab_size=11,
            n_positions=6,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=None,
            eos_token_id=None,
        )
        GPT2LMHeadModel(config).save_pretrained(cls.model_path)
        cls.inference = CausalLMInference(cls.model_path, "cpu")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_model_properties_and_eval_mode(self) -> None:
        self.assertEqual(self.inference.vocab_size, 11)
        self.assertEqual(self.inference.max_length, 6)
        self.assertEqual(self.inference.device.type, "cpu")
        self.assertFalse(self.inference.model.training)

    def test_maximum_length_and_highest_token_return_cpu_float_logits(self) -> None:
        logits = self.inference.next_token_logits([10] * 6)
        self.assertEqual(logits.shape, (11,))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(logits.device.type, "cpu")

    def test_next_token_logits_batch_matches_individual_sequences(self) -> None:
        sequences = [[1, 2], [3, 4], [5, 6]]
        expected = torch.stack(
            [self.inference.next_token_logits(sequence) for sequence in sequences]
        )
        logits = self.inference.next_token_logits_batch(sequences)
        self.assertEqual(logits.shape, (3, 11))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(logits.device, self.inference.device)
        torch.testing.assert_close(logits, expected, rtol=0, atol=1e-6)

    def test_next_token_logits_batch_rejects_unequal_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have equal length"):
            self.inference.next_token_logits_batch([[1, 2], [3]])

    def test_input_longer_than_maximum_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds model maximum 6"):
            self.inference.next_token_logits([0] * 7)

    def test_out_of_vocabulary_ids_are_rejected(self) -> None:
        for token_id in (11, -1):
            with self.subTest(token_id=token_id):
                with self.assertRaisesRegex(ValueError, r"outside \[0, 11\)"):
                    self.inference.next_token_logits([token_id])

    def test_empty_sequence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.inference.next_token_logits([])

    def test_non_integer_ids_are_rejected(self) -> None:
        for token_id in (True, 1.5):
            with self.subTest(token_id=token_id):
                with self.assertRaisesRegex(ValueError, "must be integers"):
                    self.inference.next_token_logits([token_id])

    def test_continuation_log_probs_match_tokenwise_scores_and_batch_sizes(self) -> None:
        context = [1, 2, 3]
        continuations = [[7, 8], [9], [2, 4]]
        expected = []
        for continuation in continuations:
            prefix = list(context)
            score = 0.0
            for token_id in continuation:
                logits = self.inference.next_token_logits(prefix)
                score += torch.log_softmax(logits, dim=-1)[token_id].item()
                prefix.append(token_id)
            expected.append(score)

        scores_one = self.inference.continuation_log_probs(
            context, continuations, batch_size=1
        )
        scores_three = self.inference.continuation_log_probs(
            context, continuations, batch_size=3
        )
        self.assertEqual(scores_one.shape, (3,))
        self.assertEqual(scores_one.dtype, torch.float64)
        self.assertEqual(scores_one.device.type, "cpu")
        torch.testing.assert_close(
            scores_one, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=1e-5
        )
        torch.testing.assert_close(scores_one, scores_three, rtol=0, atol=1e-7)

    def test_continuation_log_probs_validates_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            self.inference.continuation_log_probs([1], [[2]], batch_size=0)
        with self.assertRaisesRegex(ValueError, "token_ids must not be empty"):
            self.inference.continuation_log_probs([], [[2]])
        with self.assertRaisesRegex(ValueError, "continuations must not be empty"):
            self.inference.continuation_log_probs([1], [])
        with self.assertRaisesRegex(
            ValueError,
            "context plus continuation length 7 exceeds model maximum 6",
        ):
            self.inference.continuation_log_probs([1, 2, 3, 4, 5], [[6, 7]])

    def test_returned_logits_can_be_modified_in_place(self) -> None:
        logits = self.inference.next_token_logits([1, 2, 3])
        logits.add_(1)

    def test_missing_model_directory_is_rejected(self) -> None:
        missing_path = Path(self._temporary_directory.name) / "missing"
        with self.assertRaisesRegex(ValueError, "model directory does not exist"):
            CausalLMInference(missing_path, "cpu")


if __name__ == "__main__":
    unittest.main()
