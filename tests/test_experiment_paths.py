import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.experiment_paths import dataset_samples_path


class ExperimentPathsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        (self.root / "b.parquet").touch()
        (self.root / "a.parquet").touch()
        patcher = mock.patch("utils.experiment_paths.SAMPLES_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_unknown_dataset_lists_available_names_in_sorted_order(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "unknown dataset: missing; available: a, b"
        ):
            dataset_samples_path("missing")

    def test_existing_dataset_resolves_to_its_sample_file(self) -> None:
        self.assertEqual(dataset_samples_path("a"), self.root / "a.parquet")

    def test_dataset_name_rejects_path_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid dataset name"):
            dataset_samples_path("../a")


if __name__ == "__main__":
    unittest.main()
