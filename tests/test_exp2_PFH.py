import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.exp2_PFH.feature_logprob import (
    build_exp2_pfh,
    parquet_inputs,
    stream_inputs,
)
from utils.inference import CausalLMInference


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
            bos_token_id=0,
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
                pa.array([0, 1, 2, 3, 4], type=pa.int32()),
                pa.array(
                    ["history", "history", "future", "future", "future"],
                    type=pa.string(),
                ),
                pa.array([3, 3, 2, 2, 1], type=pa.int32()),
                pa.array([0, 3, 6, 8, 10], type=pa.int64()),
                pa.array(
                    [[1, 2, 3], [4, 5, 6], [7, 8], [9, 10], [5]],
                    type=pa.list_(pa.int32()),
                ),
                pa.array(["h0", "h1", "f0", "f1", "f2"], type=pa.string()),
            ],
            schema=schema,
        )
        pq.write_table(table, cls.samples_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_matrix_matches_brute_force_pmi(self) -> None:
        output = self.root / "out"
        summary = build_exp2_pfh(
            inputs=parquet_inputs(self.samples_path, None, None),
            model_path=self.model_path,
            device="cpu",
            batch_size=2,
            output_dir=output,
            overwrite=False,
        )
        matrix = np.load(output / "pmi.npy")
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["columns"], 3)
        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(matrix.dtype, np.float64)
        self.assertEqual(metadata["source"]["history_sample_ids"], [0, 1])
        self.assertEqual(metadata["source"]["future_sample_ids"], [2, 3, 4])
        self.assertEqual(metadata["source"]["source_metadata"], {"fixture": True})

        inference = CausalLMInference(self.model_path, "cpu")
        histories = ([1, 2, 3], [4, 5, 6])
        futures = ([7, 8], [9, 10], [5])

        def lp(tokens: list[int]) -> float:
            return sum(
                torch.log_softmax(inference.next_token_logits(tokens[:t]), dim=-1)[
                    tokens[t]
                ].item()
                for t in range(1, len(tokens))
            )

        for history_index, history in enumerate(histories):
            for future_index, future in enumerate(futures):
                expected = lp(history + future) - lp(history) - lp(future)
                self.assertAlmostEqual(
                    matrix[history_index, future_index], expected, delta=1e-5
                )

    def test_stream_inputs_are_seeded_windows_at_recorded_starts(self) -> None:
        stream_path = self.root / "stream.npy"
        tokens = (np.arange(40) % 11).astype(np.uint16)
        np.save(stream_path, tokens)
        stream_path.with_suffix(".json").write_text(
            json.dumps({"vocab_size": 11}), encoding="utf-8"
        )
        kwargs = dict(
            history_count=3, future_count=2, history_tokens=4, future_tokens=3, seed=5
        )
        inputs = stream_inputs(stream_path, **kwargs)
        self.assertEqual(inputs, stream_inputs(stream_path, **kwargs))
        for history, start in zip(inputs.histories, inputs.source["history_starts"]):
            self.assertEqual(history, tokens[start : start + 4].tolist())
        for future, start in zip(inputs.futures, inputs.source["future_starts"]):
            self.assertEqual(future, tokens[start : start + 3].tolist())
        with self.assertRaisesRegex(ValueError, "seed must be non-negative"):
            stream_inputs(stream_path, 3, 2, 4, 3, -1)

    def test_existing_output_requires_overwrite_before_model_load(self) -> None:
        output = self.root / "existing"
        output.mkdir(exist_ok=True)
        (output / "metadata.json").touch()
        with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
            build_exp2_pfh(
                inputs=parquet_inputs(self.samples_path, None, None),
                model_path=self.root / "missing-model",
                device="cpu",
                batch_size=64,
                output_dir=output,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
