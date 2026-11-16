import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from experiments.exp4_table_neural_svd.cli import EXPERIMENT as TABLES_EXPERIMENT
from experiments.exp4_table_neural_svd.cli import positive_float
from experiments.exp5_gpt2_mlp_table.cli import DEFAULT_STORE_DIR
from utils.experiment_paths import INFERENCE_DIR, add_run_arguments, run_dir

from . import EXPERIMENT, TRAIN_DIR


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp6",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser(
        "train",
        help="Train two copies of GPT-2, the right one bilinear in the header token's "
             "unembedding row, with sample-store pairs and compare with the exact SVDs "
             "of the exp4 tables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument("--neigs", type=positive_int, default=128)
    train_parser.add_argument(
        "--head-rank", type=positive_int, default=128,
        help="d: the right head gives neigs x d numbers per future, and the unembedding "
             "rows are projected to d dimensions; at least the tables' top-k loses nothing",
    )
    train_parser.add_argument(
        "--batch-size", type=positive_int, default=128,
        help="(history, future) pairs per step; each gives every header token of its future",
    )
    train_parser.add_argument("--steps", type=positive_int, default=100000)
    train_parser.add_argument("--eval-every", type=positive_int, default=1000)
    train_parser.add_argument(
        "--lr", type=positive_float, default=3e-4,
        help="Learning rate of the heads and of the projection of the unembedding rows",
    )
    train_parser.add_argument(
        "--encoder-lr", type=positive_float, default=2e-5,
        help="Learning rate of the two trainable copies of the model's transformer",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--store-dir", type=Path, default=DEFAULT_STORE_DIR,
        help="Sample store to draw training pairs from; it must come from --model-path",
    )
    add_run_arguments(train_parser)
    # The evaluation tables are exp4's, which are built from this sample set.
    train_parser.set_defaults(dataset="wikitext-large")
    train_parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        from .train import train_exp6

        model_label = args.model_path.name
        summary = train_exp6(
            inference_dir=run_dir(
                args.results_root, TABLES_EXPERIMENT, args.dataset, model_label
            ) / INFERENCE_DIR,
            store_dir=args.store_dir,
            model_path=args.model_path,
            output_dir=run_dir(args.results_root, EXPERIMENT, args.dataset, model_label)
            / TRAIN_DIR,
            device=args.device,
            neigs=args.neigs,
            head_rank=args.head_rank,
            batch_size=args.batch_size,
            steps=args.steps,
            eval_every=args.eval_every,
            lr=args.lr,
            encoder_lr=args.encoder_lr,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"output_dir: {summary['output_dir']}\n"
        f"step: {summary['step']}\n"
        f"evaluations: {summary['evaluations']}"
    )
