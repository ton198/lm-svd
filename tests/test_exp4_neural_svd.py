import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.exp4_neural_svd import (
    analyze_exp4,
    centered_logit_block,
    train_exp4,
)
from utils import CausalLMInference, load_samples


class Exp4NeuralSVDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        previous_rng_state = torch.get_rng_state()
        cls.addClassCleanup(torch.set_rng_state, previous_rng_state)
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
        cls.samples_path = cls.root / "samples.parquet"
        schema = pa.schema(
            [
                ("sample_id", pa.int32()),
                ("group", pa.string()),
                ("token_count", pa.int32()),
                ("start_token", pa.int64()),
                ("token_ids", pa.list_(pa.int32())),
                ("text", pa.string()),
            ],
            metadata={b"lm_svd": json.dumps({"fixture": True}).encode("utf-8")},
        )
        token_ids = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 0, 1],
            [7, 8],
            [9, 10],
            [2, 4],
        ]
        table = pa.Table.from_arrays(
            [
                pa.array(range(7), type=pa.int32()),
                pa.array(["history"] * 4 + ["future"] * 3, type=pa.string()),
                pa.array([3] * 4 + [2] * 3, type=pa.int32()),
                pa.array([0, 3, 6, 9, 12, 14, 16], type=pa.int64()),
                pa.array(token_ids, type=pa.list_(pa.int32())),
                pa.array(["h0", "h1", "h2", "h3", "f0", "f1", "f2"], type=pa.string()),
            ],
            schema=schema,
        )
        pq.write_table(table, cls.samples_path)
        cls.histories, cls.futures = load_samples(cls.samples_path, None, None)
        cls.inference = CausalLMInference(cls.model_path, "cpu")

    def _train(self, output_dir: Path, *, history_batch: int = 4) -> dict:
        return train_exp4(
            samples_path=self.samples_path,
            model_path=self.model_path,
            device="cpu",
            history_count=None,
            future_count=None,
            rank=2,
            nesting="sequential",
            history_batch=history_batch,
            future_batch=3,
            # The second mode is ~85x smaller; full-batch probes need >10000 steps.
            steps=20000,
            lr=1e-2,
            seed=0,
            log_every=20000,
            output_dir=output_dir,
            overwrite=False,
        )

    def test_centered_logit_block_matches_every_full_vocabulary_entry(self) -> None:
        block = centered_logit_block(self.inference, self.histories, self.futures)
        self.assertEqual(block.shape, (4, 33))
        for history_index, history in enumerate(self.histories):
            for future_index, future in enumerate(self.futures):
                logits = self.inference.next_token_logits(
                    history.token_ids + future.token_ids
                )
                expected = logits - logits.mean()
                with self.subTest(history=history_index, future=future_index):
                    torch.testing.assert_close(
                        block[history_index, future_index * 11 : (future_index + 1) * 11],
                        expected,
                        rtol=0,
                        atol=1e-6,
                    )

    def test_training_and_analysis_recover_top_two_singular_values(self) -> None:
        result_dir = self.root / "convergence"
        result = self._train(result_dir / "train")
        self.assertEqual((result["rows"], result["columns"]), (4, 33))
        self.assertEqual(result["steps"], 20000)
        self.assertEqual(result["pairs_evaluated"], 20000 * 4 * 3)
        analyze_exp4(result_dir=result_dir, device="cpu", overwrite=False)
        matrix = centered_logit_block(
            self.inference, self.histories, self.futures
        ).double().numpy()
        expected = np.linalg.svd(matrix, compute_uv=False)[:2]
        learned = np.load(result_dir / "analysis" / "singular_values.npy")
        self.assertEqual(learned.shape, (2,))
        np.testing.assert_allclose(learned, expected, rtol=1e-2, atol=0)
        self.assertTrue((result_dir / "analysis" / "neural_svd_spectrum.png").is_file())

    def test_existing_training_metadata_requires_overwrite(self) -> None:
        output_dir = self.root / "existing" / "train"
        output_dir.mkdir(parents=True)
        (output_dir / "metadata.json").touch()
        with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
            self._train(output_dir)

    def test_oversize_history_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds history count 4"):
            self._train(self.root / "oversize" / "train", history_batch=5)


if __name__ == "__main__":
    unittest.main()
