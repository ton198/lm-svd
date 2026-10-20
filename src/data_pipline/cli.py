import argparse
from pathlib import Path

from .task_inputs import build_task_input_tables
from .windows import build_windows_table


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_window_arguments(
    parser: argparse.ArgumentParser, *, history_count: int, future_count: int, output: Path
) -> None:
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/original/wikitext"), help="Local dataset root")
    parser.add_argument("--config", default="wikitext-103-raw-v1", help="Dataset configuration directory")
    parser.add_argument("--split", default="train", help="Parquet split prefix")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("models/gpt2-small"), help="Local tokenizer directory; no Hub downloads")
    parser.add_argument("--history-count", type=positive_int, default=history_count, help="Number of history windows")
    parser.add_argument("--history-tokens", type=positive_int, default=64, help="Tokens per history window")
    parser.add_argument("--future-count", type=positive_int, default=future_count, help="Number of future windows")
    parser.add_argument("--future-tokens", type=positive_int, default=32, help="Tokens per future window")
    parser.add_argument("--seed", type=int, default=0, help="Random selection seed")
    parser.add_argument(
        "--output", type=Path,
        default=output,
        help="Output Parquet file",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="data-pipline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build-windows",
        help="Build independent, non-overlapping history/future windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_window_arguments(
        build, history_count=512, future_count=512,
        output=Path("data/history_future_samples/wikitext.parquet"),
    )
    large = subparsers.add_parser(
        "build-wikitext-large",
        help="Build the large WikiText window set used by exp4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_window_arguments(
        large, history_count=10000, future_count=10000,
        output=Path("data/history_future_samples/wikitext-large.parquet"),
    )
    task_inputs = subparsers.add_parser(
        "build-task-inputs",
        help="Build task histories and shared C4 document futures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    task_inputs.add_argument(
        "--task-catalog", type=Path,
        default=Path("data/original/task_sources/single_text.jsonl"),
        help="Frozen single-document task catalog",
    )
    task_inputs.add_argument(
        "--c4-paths", type=Path, nargs="+",
        default=[Path("data/original/c4/en/c4-validation.00000-of-00008.json.gz")],
        help="Local C4 JSON gzip shards",
    )
    task_inputs.add_argument(
        "--tokenizer-path", type=Path, default=Path("models/gpt2-small"),
        help="Local tokenizer directory; no Hub downloads",
    )
    task_inputs.add_argument(
        "--input-count", type=positive_int, default=1309, help="Number of shared documents",
    )
    task_inputs.add_argument(
        "--history-max-tokens", type=positive_int, default=256,
        help="Maximum rendered history length",
    )
    task_inputs.add_argument(
        "--passage-min-tokens", type=positive_int, default=96,
        help="Minimum document passage length",
    )
    task_inputs.add_argument(
        "--passage-max-tokens", type=positive_int, default=160,
        help="Maximum document passage length",
    )
    task_inputs.add_argument("--seed", type=int, default=0, help="Random selection seed")
    task_inputs.add_argument(
        "--output-dir", type=Path, default=Path("data/history_future_samples"),
        help="Output Parquet directory",
    )
    task_inputs.add_argument(
        "--overwrite", action="store_true", help="Replace existing outputs",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "build-task-inputs":
            build_task_input_tables(
                task_catalog=args.task_catalog,
                c4_paths=args.c4_paths,
                tokenizer_path=args.tokenizer_path,
                input_count=args.input_count,
                history_max_tokens=args.history_max_tokens,
                passage_min_tokens=args.passage_min_tokens,
                passage_max_tokens=args.passage_max_tokens,
                seed=args.seed,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
            return
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
