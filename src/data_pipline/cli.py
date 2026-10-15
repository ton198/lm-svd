import argparse
from pathlib import Path

from .windows import build_windows_table


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="data-pipline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build-windows",
        help="Build independent, non-overlapping history/future windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build.add_argument("--dataset-dir", type=Path, default=Path("data/original/wikitext"), help="Local dataset root")
    build.add_argument("--config", default="wikitext-103-raw-v1", help="Dataset configuration directory")
    build.add_argument("--split", default="train", help="Parquet split prefix")
    build.add_argument("--tokenizer-path", type=Path, default=Path("models/gpt2-small"), help="Local tokenizer directory; no Hub downloads")
    build.add_argument("--history-count", type=positive_int, default=512, help="Number of history windows")
    build.add_argument("--history-tokens", type=positive_int, default=64, help="Tokens per history window")
    build.add_argument("--future-count", type=positive_int, default=512, help="Number of future windows")
    build.add_argument("--future-tokens", type=positive_int, default=32, help="Tokens per future window")
    build.add_argument("--seed", type=int, default=0, help="Random selection seed")
    build.add_argument(
        "--output", type=Path,
        default=Path("data/history_future_samples/wikitext-103-raw-v1-train-gpt2-history-future.parquet"),
        help="Output Parquet file",
    )
    build.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    args = parser.parse_args(argv)
    try:
        summary = build_windows_table(
            dataset_dir=args.dataset_dir,
            config=args.config,
            split=args.split,
            tokenizer_path=args.tokenizer_path,
            history_count=args.history_count,
            history_tokens=args.history_tokens,
            future_count=args.future_count,
            future_tokens=args.future_tokens,
            seed=args.seed,
            output=args.output,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"output: {summary['output']}\n"
        f"stream_tokens: {summary['stream_tokens']}\n"
        f"history: {summary['history_count']} x {summary['history_tokens']} tokens\n"
        f"future: {summary['future_count']} x {summary['future_tokens']} tokens"
    )
