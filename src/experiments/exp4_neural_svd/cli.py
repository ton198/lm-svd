import argparse
import math
from pathlib import Path

from data_pipline.cli import positive_int
from utils.analysis import ANALYSIS_DIR
from utils.experiment_paths import add_run_arguments, dataset_samples_path, run_dir


EXPERIMENT = "exp4_neural_svd"
TRAIN_DIR = "train"


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive float") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive float")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser(
        "train",
        help="Learn the full-vocabulary centered-logit matrix with NeuralSVD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_arguments(train_parser)
    train_parser.set_defaults(dataset="wikitext-large")
    train_parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--history-count", type=positive_int, default=None)
    train_parser.add_argument("--future-count", type=positive_int, default=None)
    train_parser.add_argument("--rank", type=positive_int, default=32)
    train_parser.add_argument(
        "--nesting", choices=("sequential", "joint"), default="sequential"
    )
    train_parser.add_argument("--history-batch", type=positive_int, default=64)
    train_parser.add_argument("--future-batch", type=positive_int, default=64)
    train_parser.add_argument("--steps", type=positive_int, default=100000)
    train_parser.add_argument("--lr", type=positive_float, default=1e-2)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--log-every", type=positive_int, default=100)
    train_parser.add_argument("--overwrite", action="store_true")
    analysis_parser = subparsers.add_parser(
        "analysis",
        help=f"Analyze the learned spectrum and write {ANALYSIS_DIR} outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_arguments(analysis_parser)
    analysis_parser.set_defaults(dataset="wikitext-large")
    analysis_parser.add_argument(
        "--model-path", type=Path, default=Path("models/gpt2-small"),
        help="Model whose run to analyze; only its directory name is used",
    )
    analysis_parser.add_argument(
        "--device", default="cuda",
        help="Torch device for the float64 spectrum computation",
    )
    analysis_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result_dir = run_dir(
            args.results_root, EXPERIMENT, args.dataset, args.model_path.name
        )
        if args.command == "analysis":
            from .spectrum import analyze_exp4

            print(analyze_exp4(
                result_dir=result_dir, device=args.device, overwrite=args.overwrite
            ))
            return
        from .train import train_exp4

        summary = train_exp4(
            samples_path=dataset_samples_path(args.dataset),
            model_path=args.model_path,
            device=args.device,
            history_count=args.history_count,
            future_count=args.future_count,
            rank=args.rank,
            nesting=args.nesting,
            history_batch=args.history_batch,
            future_batch=args.future_batch,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed,
            log_every=args.log_every,
            output_dir=result_dir / TRAIN_DIR,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"output_dir: {summary['output_dir']}\n"
        f"matrix: {summary['rows']} x {summary['columns']}\n"
        f"steps: {summary['steps']}\n"
        f"pairs_evaluated: {summary['pairs_evaluated']}\n"
        f"final_loss: {summary['final_loss']}"
    )
