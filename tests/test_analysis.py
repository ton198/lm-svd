import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.analysis import (
    CENTERING_CHOICES,
    compute_svd,
    fit_log_log,
    plot_spectrum_panels,
    visualize_svd,
)
from experiments.exp1_L_M_table import analysis as exp1_analysis
from experiments.exp2_PFH import analysis as exp2_analysis
from experiments.exp2_PFH.analysis import run_analysis
from experiments.exp2_PFH.cli import main as exp2_main
from experiments.exp3_L_M_diff import analysis as exp3_analysis


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.matrix = (
            0.25 * np.outer([1, 2, 3, 4], [1, 0, 2, 0, 1, 3])
            + np.outer([0, 1, 0, -1], [2, 1, 0, 1, 0, 1])
        ).astype(np.float32)

    def test_full_pipeline_reconstructs_matrix_and_finds_rank(self) -> None:
        svd, summary = compute_svd(self.matrix, centering="none", device="cpu")
        np.testing.assert_allclose((svd.U * svd.S) @ svd.Vt, self.matrix, atol=1e-5)
        self.assertEqual(svd.rank, 2)
        self.assertEqual(
            svd.rank_tolerance, float(svd.S.max()) * 6 * np.finfo(np.float64).eps
        )
        self.assertEqual(summary["singular_value_count"], 4)
        plot_path = self.root / "plot.png"
        spectrum, fit = visualize_svd(
            svd.S,
            rank_tolerance=svd.rank_tolerance,
            matrix_shape=(4, 6),
            plot_path=plot_path,
        )
        self.assertTrue(plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(fit["point_count"], 2)
        self.assertEqual(len(spectrum["singular_values"]), 2)
        self.assertAlmostEqual(spectrum["cumulative_energy_fraction"][-1], 1.0)
        self.assertNotIn("loglog_fit", spectrum)

    def test_log_log_fit_recovers_power_law(self) -> None:
        _, fit = visualize_svd(
            10.0 * np.arange(1, 6, dtype=np.float64) ** -1.5,
            rank_tolerance=0.0,
            matrix_shape=(5, 5),
            plot_path=self.root / "plot.png",
        )
        self.assertAlmostEqual(fit["slope"], -1.5, places=10)
        self.assertAlmostEqual(fit["intercept"], 1.0, places=10)
        self.assertEqual(fit["point_count"], 5)
        self.assertEqual(fit["log_base"], 10)

    def test_spectrum_panels_draw_fit_and_optional_rank(self) -> None:
        values = 10.0 * np.arange(1, 6) ** -1.5
        legend_labels = [
            "singular values",
            "log-log fit: k = -1.5, b = 1",
            "rank = 5",
        ]
        for rank, energy_ylim, point_count in (
            (5, (0, 1.01), 5),
            (None, None, 5),
            (None, None, 1),
        ):
            with self.subTest(rank=rank, point_count=point_count):
                spectrum = values[:point_count]
                energy = np.cumsum(spectrum**2) / (spectrum**2).sum()
                fig, (value_ax, energy_ax) = plt.subplots(1, 2)
                self.addCleanup(plt.close, fig)
                plot_spectrum_panels(
                    value_ax,
                    energy_ax,
                    spectrum,
                    energy,
                    fit_log_log(spectrum),
                    index_label="index",
                    energy_title="Cumulative energy",
                    energy_ylim=energy_ylim,
                    rank=rank,
                )
                expected_labels = (
                    legend_labels if rank is not None else legend_labels[:2]
                )
                if point_count == 1:
                    expected_labels = legend_labels[:1]
                    self.assertEqual(len(value_ax.lines), 1)
                else:
                    np.testing.assert_allclose(
                        value_ax.lines[1].get_ydata(), spectrum, rtol=1e-9
                    )
                self.assertEqual(
                    [text.get_text() for text in value_ax.get_legend().get_texts()],
                    expected_labels,
                )
                self.assertEqual(value_ax.get_xscale(), "log")
                self.assertEqual(value_ax.get_yscale(), "log")
                if rank is not None:
                    self.assertEqual(energy_ax.get_ylim(), energy_ylim)
                    self.assertEqual(
                        [
                            text.get_text()
                            for text in energy_ax.get_legend().get_texts()
                        ],
                        ["rank = 5"],
                    )
                else:
                    self.assertIsNone(energy_ax.get_legend())

    def test_existing_outputs_require_overwrite(self) -> None:
        for existing_name in (
            "svd_U.npy", "svd_S.npy", "svd_Vt.npy", "metadata.json", "svd_spectrum.png"
        ):
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
                        centering="none",
                        device="cpu",
                        overwrite=False,
                    )
                self.assertEqual(existing.read_bytes(), b"existing output")
                self.assertEqual(set(output.iterdir()), {existing})

        result_dir = self.root / "overwrite"
        inference = result_dir / "inference"
        inference.mkdir(parents=True)
        matrix_path = inference / "feature_log_prob.npy"
        np.save(matrix_path, self.matrix)
        run_analysis(
            result_dir=result_dir, centering="none", device="cpu", overwrite=False
        )
        np.save(matrix_path, np.eye(3, dtype=np.float32))
        run_analysis(
            result_dir=result_dir, centering="none", device="cpu", overwrite=True
        )
        metadata = json.loads(
            (result_dir / "analysis" / "none" / "metadata.json").read_text()
        )
        self.assertEqual(metadata["rank"]["value"], 3)
        self.assertEqual(metadata["input"]["matrix_shape"], [3, 3])

    def test_missing_matrix_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"matrix file does not exist: .*feature_log_prob\.npy"
        ):
            run_analysis(
                result_dir=self.root, centering="none", device="cpu", overwrite=False
            )

    def test_zero_matrix_has_rank_zero_and_no_spectrum(self) -> None:
        matrix = np.zeros((3, 5), dtype=np.float32)
        svd, _ = compute_svd(matrix, centering="none", device="cpu")
        self.assertEqual(svd.rank, 0)
        self.assertEqual(svd.rank_tolerance, 0)
        plot_path = self.root / "zero" / "plot.png"
        with self.assertRaisesRegex(ValueError, "all singular values are zero"):
            visualize_svd(
                svd.S,
                rank_tolerance=svd.rank_tolerance,
                matrix_shape=(3, 5),
                plot_path=plot_path,
            )
        self.assertFalse(plot_path.parent.exists())
        result_dir = self.root / "zero-run"
        inference = result_dir / "inference"
        inference.mkdir(parents=True)
        np.save(inference / "feature_log_prob.npy", matrix)
        with self.assertRaisesRegex(ValueError, "all singular values are zero"):
            run_analysis(
                result_dir=result_dir, centering="none", device="cpu", overwrite=False
            )
        self.assertFalse((result_dir / "analysis").exists())

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
                self.assertEqual(
                    {path.name for path in output.iterdir()},
                    {"svd_U.npy", "svd_S.npy", "svd_Vt.npy", "metadata.json", "svd_spectrum.png"},
                )
                U = np.load(output / "svd_U.npy")
                S = np.load(output / "svd_S.npy")
                Vt = np.load(output / "svd_Vt.npy")
                reconstructed = (U * S) @ Vt
                np.testing.assert_allclose(
                    reconstructed, expected[centering], atol=1e-5
                )
                metadata = json.loads((output / "metadata.json").read_text())
                self.assertEqual(
                    list(metadata),
                    ["input", "centering", "svd", "rank", "spectrum", "loglog_fit"],
                )
                self.assertEqual(metadata["centering"]["strategy"], centering)
                self.assertEqual(metadata["centering"]["formula"], formulas[centering])
                self.assertEqual(metadata["input"]["matrix_dtype"], "float32")
                self.assertEqual(metadata["input"]["inference_metadata"], {"fixture": True})
                self.assertEqual(metadata["input"]["matrix_shape"], [4, 6])
                self.assertEqual(
                    metadata["svd"]["factors"],
                    {"U": "svd_U.npy", "S": "svd_S.npy", "Vt": "svd_Vt.npy"},
                )
                self.assertEqual(metadata["spectrum"]["plot"], "svd_spectrum.png")
                self.assertTrue(
                    (output / "svd_spectrum.png").read_bytes().startswith(
                        b"\x89PNG\r\n\x1a\n"
                    )
                )
                rank = metadata["rank"]
                fit = metadata["loglog_fit"]
                self.assertEqual(len(metadata["spectrum"]["singular_values"]), rank["value"])
                self.assertEqual(fit["point_count"], rank["value"])
                if centering == "deflate1":
                    self.assertIsNone(fit["slope"])
                    self.assertIsNone(fit["intercept"])
                    self.assertEqual(fit["point_count"], 1)
                else:
                    self.assertIsInstance(fit["slope"], float)
                if centering == "row":
                    self.assertLess(np.abs(reconstructed.sum(axis=1)).max(), 1e-5)
                elif centering == "global":
                    self.assertLess(abs(reconstructed.mean()), 1e-6)
                elif centering == "deflate1":
                    np.testing.assert_allclose(S[:-1], s[1:], atol=1e-5)
                    self.assertEqual(rank["value"], 1)

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            exp2_main([
                "analysis", "--results-root", str(self.root), "--centering", "row",
                "--device", "cpu",
            ])
        with self.assertRaisesRegex(ValueError, "unknown centering"):
            compute_svd(self.matrix, centering="bogus", device="cpu")

    def test_each_experiment_reads_its_own_matrix(self) -> None:
        for module, matrix_name in (
            (exp1_analysis, "matrix.npy"),
            (exp2_analysis, "feature_log_prob.npy"),
            (exp3_analysis, "matrix.npy"),
        ):
            with self.subTest(experiment=module.__name__):
                result_dir = self.root / module.__name__
                inference = result_dir / "inference"
                inference.mkdir(parents=True)
                np.save(inference / matrix_name, self.matrix)
                module.run_analysis(
                    result_dir=result_dir,
                    centering="none",
                    device="cpu",
                    overwrite=False,
                )
                metadata = json.loads(
                    (result_dir / "analysis" / "none" / "metadata.json").read_text()
                )
                self.assertTrue(metadata["input"]["matrix_path"].endswith(matrix_name))
                self.assertIsNone(metadata["input"]["inference_metadata"])
                self.assertEqual(metadata["rank"]["value"], 2)


if __name__ == "__main__":
    unittest.main()
