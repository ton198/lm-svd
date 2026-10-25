import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_pipline.token_stream import build_token_stream


class TokenStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset_dir = self.root / "dataset"
        self.config_dir = self.dataset_dir / "tiny"
        self.config_dir.mkdir(parents=True)
        self.tokenizer_path = self.root / "tokenizer"
        self.tokenizer_path.mkdir()
        self.output = self.root / "streams" / "train.npy"
        self.tokenizer = MagicMock()
        self.tokenizer.__len__.return_value = 11
        self.tokenizer.side_effect = lambda texts, **kwargs: {
            "input_ids": [[int(token) for token in text.split()] for text in texts]
        }
        self.loader = self.enterContext(
            patch(
                "data_pipline.token_stream.AutoTokenizer.from_pretrained",
                return_value=self.tokenizer,
            )
        )
        self.write_split("train-00001.parquet", ["4 5", "6"])
        self.write_split("train-00000.parquet", ["1 2", "", "3"])
        self.write_split("validation-00000.parquet", ["10"])

    def write_split(self, name: str, texts: list[str]) -> None:
        pq.write_table(pa.table({"text": texts}), self.config_dir / name)

    def build(self, **overrides) -> dict[str, int | str]:
        arguments = {
            "dataset_dir": self.dataset_dir,
            "config": "tiny",
            "split": "train",
            "tokenizer_path": self.tokenizer_path,
            "output": self.output,
            "overwrite": False,
        }
        arguments.update(overrides)
        return build_token_stream(**arguments)

    def test_build_preserves_full_stream_order_and_writes_metadata_last(self) -> None:
        from utils.common import save_json

        def save_metadata(path, metadata):
            np.testing.assert_array_equal(np.load(self.output), [1, 2, 3, 4, 5, 6])
            self.assertFalse(self.output.with_suffix(".json").exists())
            save_json(path, metadata)

        with patch("data_pipline.token_stream.save_json", side_effect=save_metadata):
            summary = self.build()

        self.assertEqual(summary, {"output": str(self.output), "stream_tokens": 6})
        stream = np.load(self.output, mmap_mode="r")
        self.assertEqual(stream.dtype, np.dtype("uint16"))
        self.assertEqual(stream.shape, (6,))
        self.assertEqual(
            json.loads(self.output.with_suffix(".json").read_text()),
            {
                "config": "tiny",
                "split": "train",
                "tokenizer_path": str(self.tokenizer_path),
                "vocab_size": 11,
                "stream_tokens": 6,
                "dtype": "uint16",
            },
        )
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])

    def test_large_vocabulary_uses_uint32(self) -> None:
        self.tokenizer.__len__.return_value = 70_000
        self.write_split("train-00000.parquet", ["65536 69999"])
        self.build()
        stream = np.load(self.output)
        self.assertEqual(stream.dtype, np.dtype("uint32"))
        np.testing.assert_array_equal(stream, [65536, 69999, 4, 5, 6])
        metadata = json.loads(self.output.with_suffix(".json").read_text())
        self.assertEqual(metadata["dtype"], "uint32")
        self.assertEqual(metadata["vocab_size"], 70_000)

    def test_existing_output_is_rejected_before_loading_tokenizer(self) -> None:
        self.output.parent.mkdir()
        self.output.write_bytes(b"existing stream")
        with self.assertRaises(FileExistsError) as raised:
            self.build()
        self.assertEqual(
            str(raised.exception),
            f"output already exists: {self.output}; pass --overwrite to replace it",
        )
        self.loader.assert_not_called()
        self.assertEqual(self.output.read_bytes(), b"existing stream")

    def test_overwrite_replaces_array_and_metadata(self) -> None:
        self.build()
        self.write_split("train-00000.parquet", ["9 8"])
        self.build(overwrite=True)
        np.testing.assert_array_equal(np.load(self.output), [9, 8, 4, 5, 6])
        metadata = json.loads(self.output.with_suffix(".json").read_text())
        self.assertEqual(metadata["stream_tokens"], 5)

    def test_missing_tokenizer_reports_directory(self) -> None:
        missing = self.root / "missing-tokenizer"
        with self.assertRaises(ValueError) as raised:
            self.build(tokenizer_path=missing)
        self.assertEqual(str(raised.exception), f"tokenizer directory does not exist: {missing}")
        self.loader.assert_not_called()

    def test_tokenizer_load_failure_is_wrapped(self) -> None:
        self.loader.side_effect = OSError("invalid tokenizer")
        with self.assertRaises(ValueError) as raised:
            self.build()
        self.assertEqual(
            str(raised.exception),
            f"cannot load local tokenizer from {self.tokenizer_path}: invalid tokenizer",
        )
        self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_missing_split_reports_parquet_pattern(self) -> None:
        with self.assertRaises(ValueError) as raised:
            self.build(split="test")
        self.assertEqual(
            str(raised.exception), f"no parquet files match {self.config_dir / 'test-*.parquet'}"
        )
        self.assertFalse(self.output.exists())

    def test_empty_split_does_not_write_outputs(self) -> None:
        self.write_split("empty-00000.parquet", ["", ""])
        with self.assertRaisesRegex(ValueError, "tokenized split is empty"):
            self.build(split="empty")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".json").exists())

    def test_metadata_write_failure_leaves_no_completion_marker_or_temporary(self) -> None:
        def fail_metadata(path, metadata):
            path.write_text("partial metadata")
            raise OSError("write failed")

        with patch("data_pipline.token_stream.save_json", side_effect=fail_metadata):
            with self.assertRaisesRegex(OSError, "write failed"):
                self.build()
        np.testing.assert_array_equal(np.load(self.output), [1, 2, 3, 4, 5, 6])
        self.assertFalse(self.output.with_suffix(".json").exists())
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
