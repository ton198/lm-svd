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
    sample_store_inputs,
)
from utils.inference import CausalLMInference
from utils.sample_store import produce_blocks, read_store_config, slot_bytes, slot_path


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

    def test_sample_store_inputs_use_recorded_block_windows(self) -> None:
        stream_path = self.root / "stream.npy"
        store_dir = self.root / "sample-store"
        tokens = (np.arange(40) % 11).astype(np.uint16)
        np.save(stream_path, tokens)
        stream_path.with_suffix(".json").write_text(
            json.dumps({"vocab_size": 11}), encoding="utf-8"
        )
        produce_blocks(
            store_dir=store_dir, model_path=self.model_path, stream_path=stream_path,
            device="cpu", history_tokens=4, future_tokens=3, block_histories=2,
            block_futures=2, max_bytes=2 * slot_bytes(2, 2, 8), shards=1, shard=0,
            seed=11, pair_batch=4, max_blocks=2, log_every=100, overwrite=False,
        )
        kwargs = dict(
            store_dir=store_dir, stream_path=stream_path, model_path=self.model_path,
            history_count=5, future_count=3, history_tokens=4, future_tokens=3, seed=5,
            pairing="independent",
        )
        inputs = sample_store_inputs(**kwargs)
        self.assertEqual(inputs, sample_store_inputs(**kwargs))
        source = inputs.source
        self.assertEqual(source["kind"], "sample_store")
        self.assertEqual(source["store_dir"], str(store_dir))
        self.assertEqual(source["store_config"], read_store_config(store_dir))
        self.assertEqual(source["seed"], 5)
        self.assertEqual(len(inputs.histories), 5)
        self.assertEqual(len(inputs.futures), 3)
        self.assertEqual(len(source["blocks"]), 3)

        generator = torch.Generator().manual_seed(5)
        history_starts, future_starts = [], []
        for block in source["blocks"]:
            slot = int(torch.randint(2, (1,), generator=generator))
            rows = torch.randperm(2, generator=generator).tolist()
            cols = torch.randperm(2, generator=generator).tolist()
            history_begin, future_begin = len(history_starts), len(future_starts)
            rows = rows[:min(2, 5 - history_begin)]
            cols = cols[:min(2, 3 - future_begin)]
            with np.load(slot_path(store_dir, slot)) as data:
                self.assertEqual(block["slot"], slot)
                self.assertEqual(block["sequence"], int(data["sequence"]))
                self.assertEqual(block["shard"], int(data["shard"]))
                self.assertEqual(block["history_indices"], rows)
                self.assertEqual(block["future_indices"], cols)
                history_starts.extend(data["history_starts"][rows].tolist())
                future_starts.extend(data["future_starts"][cols].tolist())
            self.assertEqual(block["history_range"], [history_begin, len(history_starts)])
            self.assertEqual(block["future_range"], [future_begin, len(future_starts)])

        self.assertEqual(source["history_starts"], history_starts)
        self.assertEqual(source["future_starts"], future_starts)
        self.assertEqual(inputs.histories, [tokens[start : start + 4].tolist()
                                            for start in history_starts])
        self.assertEqual(inputs.futures, [tokens[start : start + 3].tolist()
                                          for start in future_starts])
        with self.assertRaisesRegex(ValueError, "seed must be non-negative"):
            sample_store_inputs(**{**kwargs, "seed": -1})

    def test_adjacent_pairing_uses_store_history_continuations(self) -> None:
        stream_path = self.root / "adjacent-stream.npy"
        store_dir = self.root / "adjacent-store"
        tokens = (np.arange(40) % 11).astype(np.uint16)
        np.save(stream_path, tokens)
        stream_path.with_suffix(".json").write_text(
            json.dumps({"vocab_size": 11}), encoding="utf-8"
        )
        produce_blocks(
            store_dir=store_dir, model_path=self.model_path, stream_path=stream_path,
            device="cpu", history_tokens=4, future_tokens=3, block_histories=2,
            block_futures=2, max_bytes=2 * slot_bytes(2, 2, 8), shards=1, shard=0,
            seed=11, pair_batch=4, max_blocks=2, log_every=100, overwrite=False,
        )
        kwargs = dict(
            store_dir=store_dir, stream_path=stream_path, model_path=self.model_path,
            history_count=3, future_count=3, history_tokens=4, future_tokens=3,
            seed=5, pairing="adjacent",
        )
        inputs = sample_store_inputs(**kwargs)
        self.assertEqual(inputs, sample_store_inputs(**kwargs))
        source = inputs.source
        self.assertEqual(source["pairing"], "adjacent")
        generator = torch.Generator().manual_seed(5)
        starts = []
        for block in source["blocks"]:
            slot = int(torch.randint(2, (1,), generator=generator))
            rows = torch.randperm(2, generator=generator).tolist()[:3 - len(starts)]
            torch.randperm(2, generator=generator)
            begin = len(starts)
            with np.load(slot_path(store_dir, slot)) as data:
                self.assertEqual(block["slot"], slot)
                self.assertEqual(block["history_indices"], rows)
                starts.extend(data["history_starts"][rows].tolist())
            self.assertEqual(block["future_indices"], [])
            self.assertEqual(block["history_range"], [begin, len(starts)])
            self.assertEqual(block["future_range"], block["history_range"])
        self.assertEqual(source["history_starts"], starts)
        self.assertEqual(source["future_starts"], [start + 4 for start in starts])
        self.assertEqual(inputs.histories, [tokens[start:start + 4].tolist() for start in starts])
        self.assertEqual(inputs.futures, [tokens[start + 4:start + 7].tolist() for start in starts])
        with self.assertRaisesRegex(ValueError, "adjacent pairing requires history_count == future_count"):
            sample_store_inputs(**{**kwargs, "future_count": 2})
        with self.assertRaisesRegex(ValueError, "unknown pairing: bogus"):
            sample_store_inputs(**{**kwargs, "pairing": "bogus"})

    def test_adjacent_pairing_rejects_history_without_continuation(self) -> None:
        stream_path = self.root / "short-adjacent-stream.npy"
        store_dir = self.root / "short-adjacent-store"
        np.save(stream_path, (np.arange(11) % 11).astype(np.uint16))
        stream_path.with_suffix(".json").write_text(
            json.dumps({"vocab_size": 11}), encoding="utf-8"
        )
        produce_blocks(
            store_dir=store_dir, model_path=self.model_path, stream_path=stream_path,
            device="cpu", history_tokens=4, future_tokens=8, block_histories=2,
            block_futures=2, max_bytes=slot_bytes(2, 2, 8), shards=1, shard=0,
            seed=11, pair_batch=4, max_blocks=1, log_every=100, overwrite=False,
        )
        with self.assertRaisesRegex(ValueError, "has no 8-token continuation"):
            sample_store_inputs(
                store_dir=store_dir, stream_path=stream_path, model_path=self.model_path,
                history_count=1, future_count=1, history_tokens=4, future_tokens=8,
                seed=0, pairing="adjacent",
            )

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
