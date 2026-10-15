import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.exp1_L_M_table import build_exp1_table
from experiments.exp1_L_M_table.svd import compute_svd
from utils import CausalLMInference


class Exp1TableTests(unittest.TestCase):
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
                pa.array([[1, 2, 3], [4, 5, 6], [7, 8], [9, 10]], type=pa.list_(pa.int32())),
                pa.array(["h0", "h1", "f0", "f1"], type=pa.string()),
            ],
            schema=schema,
        )
        pq.write_table(table, cls.samples_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_table_matches_future_headers_and_centered_full_vocabulary_logits(self) -> None:
        output = self.root / "out"
        build_exp1_table(
            samples_path=self.samples_path,
            model_path=self.model_path,
            device="cpu",
            history_count=None,
            future_count=None,
            top_k=3,
            output_dir=output,
            overwrite=False,
        )
        matrix = np.load(output / "matrix.npy")
        rows = pq.read_table(output / "rows.parquet")
        columns = pq.read_table(output / "columns.parquet")
        self.assertEqual(matrix.shape, (2, 6))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(columns["future_sample_id"].to_pylist(), [2, 2, 2, 3, 3, 3])
        self.assertEqual(columns["rank"].to_pylist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(rows["history_sample_id"].to_pylist(), [0, 1])
        self.assertEqual(columns.schema.field("column_index").type, pa.int32())
        self.assertEqual(columns.schema.field("future_only_centered_logit").type, pa.float32())
        inference = CausalLMInference(self.model_path, "cpu")
        histories = ([1, 2, 3], [4, 5, 6])
        futures = ([7, 8], [9, 10])
        tokens = columns["token_id"].to_numpy()
        for future_index, future in enumerate(futures):
            logits = inference.next_token_logits(future)
            values = (logits - logits.mean()).numpy()
            expected_tokens = np.lexsort((np.arange(len(values)), -values))[:3]
            actual_tokens = tokens[future_index * 3 : (future_index + 1) * 3]
            np.testing.assert_array_equal(actual_tokens, expected_tokens)
            for history_index, history in enumerate(histories):
                logits = inference.next_token_logits(history + future)
                centered = logits - logits.mean()
                np.testing.assert_allclose(
                    matrix[history_index, future_index * 3 : (future_index + 1) * 3],
                    centered[actual_tokens].numpy(),
                    rtol=0,
                    atol=1e-6,
                )

    def test_existing_output_requires_overwrite(self) -> None:
        output = self.root / "existing"
        output.mkdir(exist_ok=True)
        (output / "matrix.npy").touch()
        with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
            build_exp1_table(
                samples_path=self.samples_path,
                model_path=self.model_path,
                device="cpu",
                history_count=None,
                future_count=None,
                top_k=3,
                output_dir=output,
                overwrite=False,
            )

    def test_insufficient_history_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "requested 3 history samples but only 2 exist"
        ):
            build_exp1_table(
                samples_path=self.samples_path,
                model_path=self.model_path,
                device="cpu",
                history_count=3,
                future_count=None,
                top_k=3,
                output_dir=self.root / "insufficient",
                overwrite=False,
            )


class Exp1SvdTests(unittest.TestCase):
    def test_outputs_are_written_and_consistent(self) -> None:
        a = (
            np.outer([1, 2, 3, 4], [1, 0, 2, 0, 1, 3])
            + np.outer([0, 1, 0, -1], [2, 1, 0, 1, 0, 1])
        ).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            np.save(tmp / "matrix.npy", a)
            summary = compute_svd(input_dir=tmp, output_dir=tmp, overwrite=False)
            self.assertEqual(summary["rank"], 2)
            metadata = json.loads((tmp / "svd_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["rank"], 2)
            U = np.load(tmp / "svd_U.npy")
            S = np.load(tmp / "svd_S.npy")
            Vt = np.load(tmp / "svd_Vt.npy")
            np.testing.assert_allclose((U * S) @ Vt, a, atol=1e-5)
            self.assertIsNone(metadata["table_metadata"])

    def test_existing_output_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            np.save(tmp / "matrix.npy", np.zeros((3, 5), dtype=np.float32))
            (tmp / "svd_S.npy").touch()
            with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
                compute_svd(input_dir=tmp, output_dir=tmp, overwrite=False)


if __name__ == "__main__":
    unittest.main()
