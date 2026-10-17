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
        default=Path("data/history_future_samples/wikitext.parquet"),
        help="Output Parquet file",
    )
    build.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    task_inputs = subparsers.add_parser(
        "build-task-inputs",
        help="Build task histories and shared C4 document futures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    task_inputs.add_argument(
        "--sni-headers", type=Path,
        default=Path("data/original/natural_instructions/task_headers.jsonl"),
        help="Natural Instructions task headers",
    )
    task_inputs.add_argument(
        "--flan-templates", type=Path,
        default=Path("data/original/flan/templates.py"),
        help="Local FLAN templates source",
    )
    task_inputs.add_argument(
        "--tasksource-parquet", type=Path,
        default=Path("data/original/tasksource-instruct/data/test-00000-of-00001.parquet"),
        help="Local tasksource instruction examples",
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
        "--input-count", type=positive_int, default=1000, help="Number of shared documents",
    )
    task_inputs.add_argument(
        "--all-formats-task-count", type=positive_int, default=1000,
        help="Number of tasks in the all-formats dataset",
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
        "--pool-dir", type=Path, default=Path("data/task_input_pool"),
        help="Task and input audit JSONL directory",
    )
    task_inputs.add_argument(
        "--overwrite", action="store_true", help="Replace existing outputs",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "build-task-inputs":
            build_task_input_tables(
                sni_headers=args.sni_headers,
                flan_templates=args.flan_templates,
                tasksource_parquet=args.tasksource_parquet,
                c4_paths=args.c4_paths,
                tokenizer_path=args.tokenizer_path,
                input_count=args.input_count,
                all_formats_task_count=args.all_formats_task_count,
                history_max_tokens=args.history_max_tokens,
                passage_min_tokens=args.passage_min_tokens,
                passage_max_tokens=args.passage_max_tokens,
                seed=args.seed,
                output_dir=args.output_dir,
                pool_dir=args.pool_dir,
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
