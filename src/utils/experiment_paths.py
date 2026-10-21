import argparse
from pathlib import Path
import re


SAMPLES_DIR = Path("data/history_future_samples")
DEFAULT_DATASET = "wikitext"
DEFAULT_RESULTS_ROOT = Path("data/experiment_results")
INFERENCE_DIR = "inference"
TRAIN_DIR = "train"
ANALYSIS_DIR = "analysis"
_DATASET_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")


def validate_dataset_name(dataset: str) -> str:
    if _DATASET_NAME.fullmatch(dataset) is None:
        raise ValueError(
            f"invalid dataset name: {dataset!r}; use lowercase letters, digits and hyphens"
        )
    return dataset


def dataset_samples_path(dataset: str) -> Path:
    validate_dataset_name(dataset)
    path = SAMPLES_DIR / f"{dataset}.parquet"
    if not path.is_file():
        names = ", ".join(sorted(p.stem for p in SAMPLES_DIR.glob("*.parquet"))) or "none"
        raise ValueError(
            f"unknown dataset: {dataset}; available: {names} (files in {SAMPLES_DIR})"
        )
    return path


def run_dir(results_root: Path, experiment: str, dataset: str, model_label: str) -> Path:
    validate_dataset_name(dataset)
    return results_root / experiment / f"{dataset}__{model_label}"


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Sample set name; loads data/history_future_samples/<dataset>.parquet",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Results root; run directory is <results-root>/<experiment>/<dataset>__<model>",
    )
