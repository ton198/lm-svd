import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from utils.experiment_paths import INFERENCE_DIR, add_run_arguments, dataset_samples_path, run_dir


EXPERIMENT = "exp4_table_neural_svd"
TRAIN_DIR = "train"


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not number > 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inference_parser = subparsers.add_parser(
        "inference",
        help="Build the training and validation tables and their exact SVDs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inference_parser.add_argument("--train-histories", type=positive_int, default=2048)
    inference_parser.add_argument("--train-futures", type=positive_int, default=1024)
    inference_parser.add_argument("--val-histories", type=positive_int, default=1024)
    inference_parser.add_argument("--val-futures", type=positive_int, default=512)
    inference_parser.add_argument("--top-k", type=positive_int, default=50)
    inference_parser.add_argument(
        "--batch-size", type=positive_int, default=1024,
        help="Sequences per language-model forward pass",
    )
    train_parser = subparsers.add_parser(
        "train",
        help="Train NeuralSVD on the training table and compare with the exact SVDs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument("--neigs", type=positive_int, default=128)
    train_parser.add_argument(
        "--history-batch", type=positive_int, default=32, help="Table rows per step"
    )
    train_parser.add_argument(
        "--column-batch", type=positive_int, default=512, help="Table columns per step"
    )
    train_parser.add_argument("--steps", type=positive_int, default=100000)
    train_parser.add_argument("--eval-every", type=positive_int, default=1000)
    train_parser.add_argument("--lr", type=positive_float, default=3e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    for subparser in (inference_parser, train_parser):
        add_run_arguments(subparser)
        # 3072 histories and 1536 futures need more than the 512 + 512 default set.
        subparser.set_defaults(dataset="wikitext-large")
        subparser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
        subparser.add_argument("--device", default="cuda")
        subparser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result_dir = run_dir(args.results_root, EXPERIMENT, args.dataset, args.model_path.name)
        if args.command == "train":
            from .train import train_exp4

            summary = train_exp4(
                inference_dir=result_dir / INFERENCE_DIR,
                output_dir=result_dir / TRAIN_DIR,
                device=args.device,
                neigs=args.neigs,
                history_batch=args.history_batch,
                column_batch=args.column_batch,
                steps=args.steps,
                eval_every=args.eval_every,
                lr=args.lr,
                seed=args.seed,
                overwrite=args.overwrite,
            )
            print(
                f"output_dir: {summary['output_dir']}\n"
                f"step: {summary['step']}\n"
                f"evaluations: {summary['evaluations']}"
            )
            return
        from .tables import build_exp4_tables

        summary = build_exp4_tables(
            samples_path=dataset_samples_path(args.dataset),
            model_path=args.model_path,
            device=args.device,
            train_histories=args.train_histories,
            train_futures=args.train_futures,
            val_histories=args.val_histories,
            val_futures=args.val_futures,
            top_k=args.top_k,
            batch_size=args.batch_size,
            output_dir=result_dir / INFERENCE_DIR,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"output_dir: {summary['output_dir']}")
    for name, table in summary["tables"].items():
        print(f"{name}: {table['matrix_shape'][0]} x {table['matrix_shape'][1]}")
