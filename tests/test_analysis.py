import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from utils.analysis import CENTERING_CHOICES, compute_rank, compute_svd, run_analysis, visualize_svd
from experiments.exp2_PFH.cli import main as exp2_main


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.matrix = (
            0.25 * np.outer([1, 2, 3, 4], [1, 0, 2, 0, 1, 3])
            + np.outer([0, 1, 0, -1], [2, 1, 0, 1, 0, 1])
        ).astype(np.float32)
        self.matrix_path = self.root / "feature_log_prob.npy"
        np.save(self.matrix_path, self.matrix)
        (self.root / "metadata.json").write_text(
            json.dumps({"fixture": True}), encoding="utf-8"
        )

    def compute_svd(self) -> None:
        compute_svd(
            matrix_path=self.matrix_path,
            output_dir=self.root,
            centering="none",
            device="cpu",
            overwrite=False,
        )

    def test_full_pipeline_reconstructs_matrix_and_finds_rank(self) -> None:
        self.compute_svd()
        U = np.load(self.root / "svd_U.npy")
        S = np.load(self.root / "svd_S.npy")
        Vt = np.load(self.root / "svd_Vt.npy")
        np.testing.assert_allclose((U * S) @ Vt, self.matrix, atol=1e-5)
        metadata = json.loads((self.root / "svd_metadata.json").read_text())
        for key in ("rank", "rank_tolerance", "rank_tolerance_rule"):
            self.assertNotIn(key, metadata)
        self.assertEqual(metadata["matrix_metadata"], {"fixture": True})
        summary = compute_rank(svd_dir=self.root, overwrite=False)
        rank = json.loads((self.root / "rank.json").read_text())
        self.assertEqual(summary["rank"], 2)
        self.assertEqual(rank["rank"], 2)
        self.assertAlmostEqual(
            rank["rank_tolerance"], float(S.max()) * 6 * np.finfo(np.float64).eps
        )
        path = visualize_svd(svd_dir=self.root, overwrite=False)
        self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_existing_outputs_require_overwrite(self) -> None:
        self.compute_svd()
        with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
            self.compute_svd()
        compute_rank(svd_dir=self.root, overwrite=False)
        with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
            compute_rank(svd_dir=self.root, overwrite=False)
        visualize_svd(svd_dir=self.root, overwrite=False)
        with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
            visualize_svd(svd_dir=self.root, overwrite=False)
        np.save(self.matrix_path, np.eye(3, dtype=np.float32))
        compute_svd(
            matrix_path=self.matrix_path,
            output_dir=self.root,
            centering="none",
            device="cpu",
            overwrite=True,
        )
        self.assertEqual(compute_rank(svd_dir=self.root, overwrite=True)["rank"], 3)
        visualize_svd(svd_dir=self.root, overwrite=True)
        rank = json.loads((self.root / "rank.json").read_text())
        self.assertEqual(rank["matrix_shape"], [3, 3])

    def test_missing_stage_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist: .*svd_S.npy"):
            compute_rank(svd_dir=self.root, overwrite=False)
        self.compute_svd()
        with self.assertRaisesRegex(ValueError, "does not exist: .*rank.json"):
            visualize_svd(svd_dir=self.root, overwrite=False)
        (self.root / "svd_metadata.json").unlink()
        with self.assertRaisesRegex(ValueError, "does not exist: .*svd_metadata.json"):
            compute_rank(svd_dir=self.root, overwrite=False)

    def test_zero_matrix_has_rank_zero_and_no_spectrum(self) -> None:
        np.save(self.matrix_path, np.zeros((3, 5), dtype=np.float32))
        self.compute_svd()
        summary = compute_rank(svd_dir=self.root, overwrite=False)
        self.assertEqual(summary["rank"], 0)
        self.assertEqual(summary["rank_tolerance"], 0)
        with self.assertRaisesRegex(ValueError, "all singular values are zero"):
            visualize_svd(svd_dir=self.root, overwrite=False)
        self.assertFalse((self.root / "svd_spectrum.png").exists())

    def test_exp2_analysis_writes_each_centering_to_its_own_directory(self) -> None:
        run = self.root / "exp2_PFH" / "wikitext__gpt2-small"
        inference = run / "inference"
        inference.mkdir(parents=True)
        matrix_path = inference / "feature_log_prob.npy"
        np.save(matrix_path, self.matrix)
        (inference / "metadata.json").write_text(
            json.dumps({"fixture": True}), encoding="utf-8"
        )
        matrix = self.matrix.astype(np.float64)
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        expected = {
            "none": matrix,
            "global": matrix - matrix.mean(),
            "row": matrix - matrix.mean(axis=1, keepdims=True),
            "deflate1": matrix - s[0] * np.outer(u[:, 0], vt[0]),
        }
        formulas = {
            "none": "M",
            "global": "M - mean(M)",
            "row": "M - mean(M, axis=1, keepdims=True)",
            "deflate1": "M - S[0] * outer(U[:, 0], Vt[0, :]), where M = U @ diag(S) @ Vt",
        }
        for centering in CENTERING_CHOICES:
            with self.subTest(centering=centering):
                with redirect_stdout(StringIO()):
                    exp2_main([
                        "analysis", "--results-root", str(self.root),
                        "--centering", centering,
                        "--device", "cpu",
                    ])
                output = run / "analysis" / centering
                U = np.load(output / "svd_U.npy")
                S = np.load(output / "svd_S.npy")
                Vt = np.load(output / "svd_Vt.npy")
                reconstructed = (U * S) @ Vt
                np.testing.assert_allclose(
                    reconstructed, expected[centering], atol=1e-5
                )
                metadata = json.loads((output / "svd_metadata.json").read_text())
                self.assertEqual(metadata["centering"], centering)
                self.assertEqual(metadata["centering_formula"], formulas[centering])
                self.assertEqual(metadata["input_dtype"], "float32")
                self.assertEqual(metadata["matrix_metadata"], {"fixture": True})
                self.assertTrue((output / "rank.json").is_file())
                self.assertTrue(
                    (output / "svd_spectrum.png").read_bytes().startswith(
                        b"\x89PNG\r\n\x1a\n"
                    )
                )
                if centering == "row":
                    self.assertLess(np.abs(reconstructed.sum(axis=1)).max(), 1e-5)
                elif centering == "global":
                    self.assertLess(abs(reconstructed.mean()), 1e-6)
                elif centering == "deflate1":
                    np.testing.assert_allclose(S[:-1], s[1:], atol=1e-5)
                    rank = json.loads((output / "rank.json").read_text())
                    self.assertEqual(rank["rank"], 1)

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            exp2_main([
                "analysis", "--results-root", str(self.root), "--centering", "row",
                "--device", "cpu",
            ])
        with self.assertRaisesRegex(ValueError, "unknown centering"):
            compute_svd(
                matrix_path=matrix_path,
                output_dir=self.root / "bad",
                centering="bogus",
                device="cpu",
                overwrite=False,
            )
        self.assertFalse((self.root / "bad").exists())

    def test_analysis_preflights_rank_and_plot_collisions_before_writing_svd(self) -> None:
        for existing_name in ("rank.json", "svd_spectrum.png"):
            with self.subTest(existing_name=existing_name):
                result_dir = self.root / existing_name
                inference = result_dir / "inference"
                inference.mkdir(parents=True)
                np.save(inference / "feature_log_prob.npy", self.matrix)
                output = result_dir / "analysis" / "none"
                output.mkdir(parents=True)
                existing = output / existing_name
                existing.write_bytes(b"existing output")

                with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
                    run_analysis(
                        result_dir=result_dir,
                        matrix_name="feature_log_prob.npy",
                        centering="none",
                        device="cpu",
                        overwrite=False,
                    )

                self.assertEqual(existing.read_bytes(), b"existing output")
                self.assertEqual(set(output.iterdir()), {existing})

    def test_rank_ignores_input_dtype_and_stale_rank_metadata(self) -> None:
        tolerance = 1e12 * 5 * float(np.finfo(np.float64).eps)
        singular_values = np.array(
            [1e12, 1e5, tolerance, tolerance / 2, 0], dtype=np.float64
        )
        np.save(self.root / "svd_S.npy", singular_values)
        metadata_path = self.root / "svd_metadata.json"
        for input_dtype, stale_rank, stale_tolerance in (
            ("float32", 1, 1e6),
            ("float64", 99, 0),
            ("int64", 0, 1e12),
        ):
            with self.subTest(input_dtype=input_dtype):
                metadata = {
                    "matrix_shape": [5, 5],
                    "input_dtype": input_dtype,
                    "rank": stale_rank,
                    "rank_tolerance": stale_tolerance,
                }
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                summary = compute_rank(svd_dir=self.root, overwrite=True)
                rank = json.loads((self.root / "rank.json").read_text())
                self.assertEqual(summary["rank"], 2)
                self.assertEqual(rank["rank"], 2)
                self.assertEqual(rank["rank_tolerance"], tolerance)
                self.assertEqual(rank["input_dtype"], input_dtype)


if __name__ == "__main__":
    unittest.main()
