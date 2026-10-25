import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.exp1_L_M_table.analysis import run_analysis as run_exp1_analysis
from experiments.exp5_L_M_neural_svd.analysis import run_analysis


class Exp5LMNeuralSVDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        previous_rng_state = torch.get_rng_state()
        cls.addClassCleanup(torch.set_rng_state, previous_rng_state)
        temporary_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary_directory.cleanup)
        cls.root = Path(temporary_directory.name)
        cls.exp1_result_dir = cls.root / "exp1_L_M_table" / "fixture__model"
        inference_dir = cls.exp1_result_dir / "inference"
        inference_dir.mkdir(parents=True)

        generator = torch.Generator().manual_seed(1)
        u, _ = torch.linalg.qr(
            torch.randn(6, 6, generator=generator, dtype=torch.float64)
        )
        v, _ = torch.linalg.qr(
            torch.randn(10, 10, generator=generator, dtype=torch.float64)
        )
        matrix = (
            u[:, :6]
            @ torch.diag(torch.tensor([4, 2, 1, 0.3, 0.1, 0.05], dtype=torch.float64))
            @ v[:, :6].T
        )
        np.save(inference_dir / "matrix.npy", matrix.numpy().astype(np.float32))
        (inference_dir / "metadata.json").write_text(
            json.dumps({"future_count": 5, "top_k": 2, "matrix_shape": [6, 10]}),
            encoding="utf-8",
        )
        for centering in ("none", "deflate1"):
            run_exp1_analysis(
                result_dir=cls.exp1_result_dir,
                centering=centering,
                device="cpu",
                overwrite=False,
            )

    def _run_analysis(
        self, name: str, *, centering: str = "none", future_batch: int = 5
    ) -> Path:
        result_dir = self.root / "exp5" / name
        run_analysis(
            exp1_result_dir=self.exp1_result_dir,
            result_dir=result_dir,
            centering=centering,
            device="cpu",
            rank=2,
            history_batch=6,
            future_batch=future_batch,
            steps=2000,
            lr=1e-2,
            min_lr=1e-4,
            seed=0,
            log_every=1000,
            overwrite=False,
        )
        return result_dir / "analysis" / centering

    def test_none_matches_exact_top_components(self) -> None:
        output_dir = self._run_analysis("none")
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        np.testing.assert_allclose(
            metadata["spectrum"]["singular_values"], [4, 2], rtol=1e-3
        )
        for key in ("left_abs_cosine", "right_abs_cosine"):
            with self.subTest(alignment=key):
                self.assertTrue(np.all(np.asarray(metadata["comparison"][key]) >= 0.999))
        expected_error = math.sqrt(1 + 0.09 + 0.01 + 0.0025) / math.sqrt(
            16 + 4 + 1 + 0.09 + 0.01 + 0.0025
        )
        np.testing.assert_allclose(
            metadata["svd"]["reconstruction_relative_error"], expected_error, rtol=1e-2
        )
        for filename, shape in (
            ("svd_U.npy", (6, 2)),
            ("svd_S.npy", (2,)),
            ("svd_Vt.npy", (2, 10)),
            ("loss_history.npy", (2000,)),
        ):
            with self.subTest(filename=filename):
                self.assertEqual(np.load(output_dir / filename).shape, shape)

    def test_deflate1_drops_learned_leading_component(self) -> None:
        output_dir = self._run_analysis("deflate1", centering="deflate1")
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        np.testing.assert_allclose(
            metadata["spectrum"]["singular_values"], [2, 1], rtol=1e-3
        )
        for key in ("left_abs_cosine", "right_abs_cosine"):
            with self.subTest(alignment=key):
                self.assertTrue(np.all(np.asarray(metadata["comparison"][key]) >= 0.999))
        self.assertEqual(metadata["neural_svd"]["trained_rank"], 3)

    def test_missing_exp1_analysis_is_rejected_before_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "exp1 analysis output does not exist"):
            self._run_analysis("missing_exp1", centering="row")
        self.assertFalse((self.root / "exp5" / "missing_exp1").exists())

    def test_batch_larger_than_counts_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "future_batch 6 exceeds future count 5"):
            self._run_analysis("oversized_future_batch", future_batch=6)


if __name__ == "__main__":
    unittest.main()
