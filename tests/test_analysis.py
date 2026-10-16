import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from utils import compute_rank, compute_svd, visualize_svd
from experiments.exp2_PFH.cli import main as exp2_main


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.matrix = (
            np.outer([1, 2, 3, 4], [1, 0, 2, 0, 1, 3])
            + np.outer([0, 1, 0, -1], [2, 1, 0, 1, 0, 1])
        ).astype(np.float32)
        self.matrix_path = self.root / "feature_log_prob.npy"
        np.save(self.matrix_path, self.matrix)
        (self.root / "metadata.json").write_text(
            json.dumps({"fixture": True}), encoding="utf-8"
        )

    def compute_svd(self) -> None:
        compute_svd(matrix_path=self.matrix_path, output_dir=self.root, overwrite=False)

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
            rank["rank_tolerance"], float(S.max()) * 6 * np.finfo(np.float32).eps
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
        compute_svd(matrix_path=self.matrix_path, output_dir=self.root, overwrite=True)
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

    def test_exp2_cli_reads_feature_log_probability_matrix(self) -> None:
        with redirect_stdout(StringIO()):
            exp2_main(["svd", "--input-dir", str(self.root)])
        singular_values = np.load(self.root / "svd_S.npy")
        np.testing.assert_allclose(
            singular_values, np.linalg.svd(self.matrix.astype(np.float64), compute_uv=False),
            atol=1e-12,
        )

    def test_rank_uses_input_precision_with_legacy_metadata(self) -> None:
        np.save(self.root / "svd_S.npy", np.array([1.0, 1e-8], dtype=np.float64))
        metadata_path = self.root / "svd_metadata.json"
        metadata = {
            "matrix_shape": [2, 2], "input_dtype": "float32",
            "rank": 99, "rank_tolerance": 0,
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.assertEqual(compute_rank(svd_dir=self.root, overwrite=False)["rank"], 1)
        metadata["input_dtype"] = "float64"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.assertEqual(compute_rank(svd_dir=self.root, overwrite=True)["rank"], 2)


if __name__ == "__main__":
    unittest.main()
