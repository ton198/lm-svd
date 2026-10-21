import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from utils.analysis import compute_svd
from utils.common import load_samples
from experiments.exp1_L_M_table.table import centered_logits, select_header_tokens
from experiments.exp3_L_M_diff.table import build_exp3_diff
from experiments.exp3_L_M_diff.cli import main as exp3_main
from utils.inference import CausalLMInference


class Exp3LMDiffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary_directory.name)
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
        config = GPT2Config(
            vocab_size=11,
            n_positions=12,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=None,
            eos_token_id=None,
        )
        cls.model_a_path = cls.root / "model-a"
        torch.manual_seed(0)
        GPT2LMHeadModel(config).save_pretrained(cls.model_a_path)
        cls.model_b_path = cls.root / "model-b"
        torch.manual_seed(1)
        GPT2LMHeadModel(config).save_pretrained(cls.model_b_path)
        mismatch_config = GPT2Config(
            vocab_size=13,
            n_positions=12,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=None,
            eos_token_id=None,
        )
        cls.mismatch_model_path = cls.root / "model-mismatch"
        torch.manual_seed(2)
        GPT2LMHeadModel(mismatch_config).save_pretrained(cls.mismatch_model_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def build(self, output: Path, *, model_a: Path | None = None, model_b: Path | None = None, top_k: int = 3) -> None:
        build_exp3_diff(
            samples_path=self.samples_path,
            model_a_path=model_a or self.model_a_path,
            model_b_path=model_b or self.model_b_path,
            device="cpu",
            history_count=None,
            future_count=None,
            top_k=top_k,
            output_dir=output,
            overwrite=False,
        )

    def test_diff_uses_model_a_headers_for_both_models(self) -> None:
        output = self.root / "shared-headers"
        self.build(output)
        matrix = np.load(output / "matrix.npy")
        matrix_a = np.load(output / "matrix_a.npy")
        matrix_b = np.load(output / "matrix_b.npy")
        rows = pq.read_table(output / "rows.parquet")
        columns = pq.read_table(output / "columns.parquet")
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(matrix.shape, (2, 6))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(metadata["header_model"], "model_a")
        self.assertEqual(metadata["source_metadata"], {"fixture": True})
        self.assertEqual(rows["history_sample_id"].to_pylist(), [0, 1])

        histories, futures = load_samples(self.samples_path, None, None)
        inference_a = CausalLMInference(self.model_a_path, "cpu")
        header_ids, future_values = select_header_tokens(inference_a, futures, 3)
        tokens = columns["token_id"].to_numpy()
        np.testing.assert_array_equal(tokens, header_ids.reshape(-1))
        np.testing.assert_array_equal(
            columns["future_only_centered_logit"].to_numpy(), future_values.reshape(-1)
        )
        self.assertEqual(columns["future_sample_id"].to_pylist(), [2, 2, 2, 3, 3, 3])
        self.assertEqual(columns["rank"].to_pylist(), [0, 1, 2, 0, 1, 2])
        for matrix_part, model_path in (
            (matrix_a, self.model_a_path),
            (matrix_b, self.model_b_path),
        ):
            inference = CausalLMInference(model_path, "cpu")
            for history_index, history in enumerate(histories):
                for future_index, future in enumerate(futures):
                    expected = centered_logits(
                        inference.next_token_logits(history.token_ids + future.token_ids)
                    )[header_ids[future_index]].numpy()
                    np.testing.assert_allclose(
                        matrix_part[
                            history_index, future_index * 3 : (future_index + 1) * 3
                        ],
                        expected,
                        rtol=0,
                        atol=1e-6,
                    )
        np.testing.assert_array_equal(matrix, matrix_a - matrix_b)

    def test_identical_models_give_zero_rank(self) -> None:
        output = self.root / "identical"
        self.build(output, model_b=self.model_a_path)
        matrix = np.load(output / "matrix.npy")
        self.assertFalse(matrix.any())
        svd, _ = compute_svd(matrix, centering="none", device="cpu")
        self.assertEqual(svd.rank, 0)

    def test_vocab_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "model vocab sizes differ"):
            self.build(self.root / "mismatch", model_b=self.mismatch_model_path)

    def test_existing_output_requires_overwrite(self) -> None:
        output = self.root / "existing"
        output.mkdir()
        (output / "matrix_b.npy").touch()
        with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
            self.build(output)

    def test_cli_runs_inference_and_analysis(self) -> None:
        results_root = self.root / "cli"
        output = (
            results_root / "exp3_L_M_diff"
            / f"samples__{self.model_a_path.name}__minus__{self.model_b_path.name}"
        )
        arguments = [
            "inference",
            "--dataset",
            "samples",
            "--model-a-path",
            str(self.model_a_path),
            "--model-b-path",
            str(self.model_b_path),
            "--device",
            "cpu",
            "--top-k",
            "2",
            "--results-root",
            str(results_root),
        ]
        with (
            mock.patch("utils.experiment_paths.SAMPLES_DIR", self.root),
            redirect_stdout(StringIO()),
        ):
            exp3_main(arguments)
            exp3_main([
                "analysis", "--dataset", "samples",
                "--results-root", str(results_root),
                "--model-a-path", str(self.model_a_path),
                "--model-b-path", str(self.model_b_path),
                "--device", "cpu",
            ])
        self.assertTrue((output / "inference" / "matrix.npy").is_file())
        analysis = output / "analysis" / "none"
        metadata = json.loads((analysis / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["input"]["matrix_shape"], [2, 4])
        self.assertTrue(
            (analysis / "svd_spectrum.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        )


if __name__ == "__main__":
    unittest.main()
