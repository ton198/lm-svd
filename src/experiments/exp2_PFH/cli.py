import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from utils.analysis import CENTERING_CHOICES
from utils.experiment_paths import (
    DEFAULT_DATASET,
    DEFAULT_RESULTS_ROOT,
    INFERENCE_DIR,
    dataset_samples_path,
    run_dir,
    sample_store_dir,
    token_stream_path,
)


EXPERIMENT = "exp2_PFH"


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Parquet sample set; loads data/history_future_samples/<dataset>.parquet",
    )
    group.add_argument(
        "--stream",
        default=None,
        help="Token stream name; selects windows from its shared sample store",
    )
    parser.add_argument(
        "--pairing", choices=("independent", "adjacent"), default="independent",
        help="With --stream: 'independent' uses the store's future starts; 'adjacent' uses the tokens right after each history as its future, so diagonal entries are matched pairs",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Store-selection seed with --stream (default: 0); a nonzero seed adds -seed<N> to the run directory name",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Results root; run directory is <results-root>/<experiment>/<dataset>__<model>",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inference_parser = subparsers.add_parser(
        "inference",
        help="Build the PFH matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_source_arguments(inference_parser)
    inference_parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
    inference_parser.add_argument("--device", default="cuda")
    inference_parser.add_argument("--history-count", type=positive_int, default=None)
    inference_parser.add_argument("--future-count", type=positive_int, default=None)
    inference_parser.add_argument(
        "--sample-store", type=Path, default=None,
        help="Shared sample store directory with --stream (default: derived from stream, model and window lengths)",
    )
    inference_parser.add_argument(
        "--history-tokens", type=positive_int, default=None,
        help="History window length with --stream (default: 64)",
    )
    inference_parser.add_argument(
        "--future-tokens", type=positive_int, default=None,
        help="Future window length with --stream (default: 32)",
    )
    inference_parser.add_argument(
        "--batch-size", type=positive_int, default=64, help="Inference batch size"
    )
    inference_parser.add_argument("--overwrite", action="store_true")
    analysis_parser = subparsers.add_parser(
        "analysis",
        help="Run SVD, rank, and visualization on the inference matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_source_arguments(analysis_parser)
    analysis_parser.add_argument(
        "--model-path", type=Path, default=Path("models/gpt2-small"),
        help="Model whose run to analyze; only its directory name is used",
    )
    analysis_parser.add_argument(
        "--centering", choices=CENTERING_CHOICES, default="none",
        help="Secondary centering applied before SVD",
    )
    analysis_parser.add_argument("--device", default="cuda", help="Torch device for the float64 SVD")
    analysis_parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    args = parser.parse_args(argv)
    if args.stream is None and args.pairing == "adjacent":
        parser.error("--pairing adjacent requires --stream")
    if args.stream is None and args.seed is not None:
        parser.error("--seed requires --stream")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.command == "inference":
        if args.stream is None:
            if args.sample_store is not None:
                parser.error("--sample-store requires --stream")
            if any(value is not None for value in (args.history_tokens, args.future_tokens)):
                parser.error("--history-tokens and --future-tokens require --stream")
        elif args.history_count is None or args.future_count is None:
            parser.error("--stream requires --history-count and --future-count")
        else:
            if args.pairing == "adjacent" and args.history_count != args.future_count:
                parser.error("--pairing adjacent requires --history-count equal to --future-count")
            history_tokens = args.history_tokens or 64
            future_tokens = args.future_tokens or 32
            seed = 0 if args.seed is None else args.seed
    label = args.stream if args.stream is not None else args.dataset
    if args.pairing == "adjacent":
        label = f"{label}-adjacent"
    if args.seed:
        label = f"{label}-seed{args.seed}"
    try:
        result_dir = run_dir(
            args.results_root, EXPERIMENT, label, args.model_path.name
        )
        if args.command == "analysis":
            from .analysis import run_analysis

            print(
                run_analysis(
                    result_dir=result_dir,
                    centering=args.centering,
                    device=args.device,
                    overwrite=args.overwrite,
                )
            )
            return
        from .feature_logprob import build_exp2_pfh, parquet_inputs, sample_store_inputs

        if args.stream is not None:
            inputs = sample_store_inputs(
                store_dir=(
                    args.sample_store
                    if args.sample_store is not None
                    else sample_store_dir(
                        args.stream, args.model_path.name, history_tokens, future_tokens
                    )
                ),
                stream_path=token_stream_path(args.stream),
                model_path=args.model_path,
                history_count=args.history_count,
                future_count=args.future_count,
                history_tokens=history_tokens,
                future_tokens=future_tokens,
                seed=seed,
                pairing=args.pairing,
            )
        else:
            inputs = parquet_inputs(
                dataset_samples_path(args.dataset), args.history_count, args.future_count
            )
        summary = build_exp2_pfh(
            inputs=inputs,
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            output_dir=result_dir / INFERENCE_DIR,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"output_dir: {summary['output_dir']}\n"
        f"matrix: {summary['rows']} x {summary['columns']}"
    )
