import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.exp11_stream_student.cli import load_student_checkpoint
from experiments.exp11_stream_student.distill import distill_student
from experiments.exp11_stream_student.features import StudentLM
from utils.inference import CausalLMInference
from utils.stream_neural_svd import TokenStream, centered_unembedding


class _InterruptingStream(TokenStream):
    def __init__(self, path: Path, *, fail_on_call: int) -> None:
        super().__init__(path)
        self.fail_on_call = fail_on_call
        self.calls = 0

    def sample_starts(
        self, count: int, window: int, generator: torch.Generator
    ) -> torch.Tensor:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("interrupted")
        return super().sample_starts(count, window, generator)


class StudentFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        temporary_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary_directory.cleanup)
        cls.root = Path(temporary_directory.name)
        cls.model_path = cls.root / "model"
        config = GPT2Config(
            vocab_size=11,
            n_positions=12,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=None,
            eos_token_id=None,
        )
        GPT2LMHeadModel(config).save_pretrained(cls.model_path)
        cls.stream_path = cls.root / "stream.npy"
        stream = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1], dtype=np.uint16)
        np.save(cls.stream_path, stream)
        cls.stream_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "config": "tiny",
                    "split": "train",
                    "tokenizer_path": str(cls.model_path),
                    "vocab_size": 11,
                    "stream_tokens": len(stream),
                    "dtype": str(stream.dtype),
                }
            ),
            encoding="utf-8",
        )
        cls.stream = TokenStream(cls.stream_path)
        cls.inference = CausalLMInference(cls.model_path, "cpu")
        cls.token_matrix = centered_unembedding(cls.inference)
        cls.student_config = {"model_dim": 8, "layers": 1, "heads": 2, "max_positions": 5}

    def test_student_states_are_causal(self) -> None:
        torch.manual_seed(0)
        student = StudentLM(self.token_matrix, **self.student_config)
        ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]])
        with torch.no_grad():
            states = student(ids)
        torch.testing.assert_close(states[0, :4], states[1, :4], rtol=0, atol=1e-6)
        self.assertFalse(torch.allclose(states[0, 4], states[1, 4]))

    def test_distillation_resume_matches_uninterrupted_run(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            uninterrupted = root / "uninterrupted"
            resumed = root / "resumed"
            kwargs = dict(
                inference=self.inference,
                stream_path=self.stream_path,
                model_path=self.model_path,
                student_config=self.student_config,
                device="cpu",
                batch=2,
                steps=6,
                lr=1e-3,
                min_lr=1e-5,
                probe_windows=2,
                probe_seed=1,
                eval_every=3,
                checkpoint_every=3,
                log_every=3,
                seed=0,
                overwrite=False,
            )
            distill_student(stream=self.stream, output_dir=uninterrupted, **kwargs)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                distill_student(
                    stream=_InterruptingStream(self.stream_path, fail_on_call=5),
                    output_dir=resumed,
                    **kwargs,
                )
            self.assertTrue((resumed / "checkpoint.pt").is_file())
            distill_student(stream=TokenStream(self.stream_path), output_dir=resumed, **kwargs)

            expected = torch.load(uninterrupted / "student.pt", map_location="cpu", weights_only=True)
            actual = torch.load(resumed / "student.pt", map_location="cpu", weights_only=True)
            self.assertEqual(expected.keys(), actual.keys())
            for name in expected:
                with self.subTest(parameter=name):
                    self.assertTrue(torch.equal(expected[name], actual[name]))
            for output_dir in (uninterrupted, resumed):
                with (output_dir / "metadata.json").open(encoding="utf-8") as file:
                    metadata = json.load(file)
                self.assertIs(metadata["complete"], True)
                self.assertEqual(metadata["steps"], 6)

    def test_load_student_checkpoint_rejects_missing_incomplete_and_mismatched(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            path = Path(directory) / "student.pt"
            with self.assertRaisesRegex(ValueError, "student distillation does not exist"):
                load_student_checkpoint(path, self.student_config)
            torch.save({}, path)
            metadata_path = path.with_name("metadata.json")
            metadata = {"complete": False, "steps": 1, "steps_requested": 2}
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "student distillation is incomplete"):
                load_student_checkpoint(path, self.student_config)
            metadata.update(complete=True, student=self.student_config)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            requested = {**self.student_config, "model_dim": 16}
            with self.assertRaisesRegex(ValueError, "student checkpoint config"):
                load_student_checkpoint(path, requested)
            self.assertEqual(load_student_checkpoint(path, self.student_config), path)


if __name__ == "__main__":
    unittest.main()
