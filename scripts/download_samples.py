"""Download source WikiText Parquet shards for the history/future sampling pipeline."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Salesforce/wikitext"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CONFIGS = (
    "wikitext-103-raw-v1",
    "wikitext-103-v1",
    "wikitext-2-raw-v1",
    "wikitext-2-v1",
)
SPLITS = ("train", "validation", "test")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download pinned WikiText source data, not generated history/future windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs", nargs="+", choices=CONFIGS, default=["wikitext-103-raw-v1"],
        help="Dataset configurations to download",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
        help="Dataset splits to download",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "original" / "wikitext",
        help="Dataset root, passed to build-windows as --dataset-dir",
    )
    args = parser.parse_args(argv)
    configs = list(dict.fromkeys(args.configs))
    splits = list(dict.fromkeys(args.splits))
    patterns = [f"{config}/{split}-*.parquet" for config in configs for split in splits]
    print(f"Downloading {REPO_ID}@{REVISION} to {args.output_dir}", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=args.output_dir,
        allow_patterns=["README.md", *patterns],
    )
    for pattern in patterns:
        paths = sorted(args.output_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no downloaded Parquet files match {args.output_dir / pattern}")
        print(f"Ready: {pattern} ({len(paths)} shards)", flush=True)
    print("Next: uv run data-pipline build-windows --help", flush=True)


if __name__ == "__main__":
    main()
