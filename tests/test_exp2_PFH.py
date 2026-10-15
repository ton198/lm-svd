import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.exp2_PFH import build_exp2_pfh
from utils import CausalLMInference


class Exp2PFHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary_directory.name)
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
        table = pa.Table.from_arrays(
            [
                pa.array([0, 1, 2, 3], type=pa.int32()),
                pa.array(["history", "history", "future", "future"], type=pa.string()),
                pa.array([3, 3, 2, 2], type=pa.int32()),
                pa.array([0, 3, 6, 8], type=pa.int64()),
                pa.array(
                    [[1, 2, 3], [4, 5, 6], [7, 8], [9, 10]],
                    type=pa.list_(pa.int32()),
                ),
                pa.array(["h0", "h1", "f0", "f1"], type=pa.string()),
            ],
            schema=schema,
        )
        pq.write_table(table, cls.samples_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_matrix_matches_brute_force_scores_and_metadata(self) -> None:
        output = self.root / "out"
        summary = build_exp2_pfh(
            samples_path=self.samples_path,
            model_path=self.model_path,
            device="cpu",
            history_count=None,
            future_count=None,
            batch_size=1,
            output_dir=output,
            overwrite=False,
        )
        matrix = np.load(output / "feature_log_prob.npy")
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["columns"], 2)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix.dtype, np.float64)
        self.assertEqual(metadata["history_sample_ids"], [0, 1])
        self.assertEqual(metadata["future_sample_ids"], [2, 3])
        self.assertEqual(metadata["source_metadata"], {"fixture": True})

        inference = CausalLMInference(self.model_path, "cpu")
        histories = ([1, 2, 3], [4, 5, 6])
        futures = ([7, 8], [9, 10])
        for history_index, history in enumerate(histories):
            for future_index, future in enumerate(futures):
                prefix = list(history)
                expected = 0.0
                for token_id in future:
                    logits = inference.next_token_logits(prefix)
                    expected += torch.log_softmax(logits, dim=-1)[token_id].item()
                    prefix.append(token_id)
                self.assertAlmostEqual(
                    matrix[history_index, future_index], expected, delta=1e-5
                )

    def test_existing_output_requires_overwrite_before_model_load(self) -> None:
        output = self.root / "existing"
        output.mkdir(exist_ok=True)
        (output / "metadata.json").touch()
        with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
            build_exp2_pfh(
                samples_path=self.samples_path,
                model_path=self.root / "missing-model",
                device="cpu",
                history_count=None,
                future_count=None,
                batch_size=64,
                output_dir=output,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
