import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from experiments.exp4_table_neural_svd.cli import EXPERIMENT as TABLES_EXPERIMENT
from experiments.exp4_table_neural_svd.cli import positive_float
from utils.experiment_paths import INFERENCE_DIR, add_run_arguments, run_dir

from . import EXPERIMENT, TRAIN_DIR

DEFAULT_STORE_DIR = Path("data/sample_stores/wikitext-103-train__gpt2-small__h64-f32")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser(
        "train",
        help="Train MLPs on frozen GPT-2 hidden states with sample-store pairs and "
             "compare with the exact SVDs of the exp4 tables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument("--neigs", type=positive_int, default=128)
    train_parser.add_argument(
        "--mlp-width", type=positive_int, default=4096, help="Size of each MLP's middle layer"
    )
    train_parser.add_argument(
        "--batch-size", type=positive_int, default=256,
        help="(history, future) pairs per step; each gives every header token of its future",
    )
    train_parser.add_argument("--steps", type=positive_int, default=100000)
    train_parser.add_argument("--eval-every", type=positive_int, default=1000)
    train_parser.add_argument("--lr", type=positive_float, default=3e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    encoders = train_parser.add_mutually_exclusive_group()
    encoders.add_argument(
        "--unfreeze-right", action="store_true",
        help="Train a copy of the model's transformer as the right side's encoder; "
             "results go to train-unfreeze-right/",
    )
    encoders.add_argument(
        "--finetune", action="store_true",
        help="Train a copy of the model's transformer per side with a linear head; the "
             "right side reads the future followed by each header token; --mlp-width is "
             "unused; results go to train-finetune/",
    )
    train_parser.add_argument(
        "--encoder-lr", type=positive_float, default=2e-5,
        help="Learning rate of the trainable encoders, with --unfreeze-right or --finetune",
    )
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
        from .train import train_exp5

        model_label = args.model_path.name
        summary = train_exp5(
            inference_dir=run_dir(
                args.results_root, TABLES_EXPERIMENT, args.dataset, model_label
            ) / INFERENCE_DIR,
            store_dir=args.store_dir,
            model_path=args.model_path,
            output_dir=run_dir(args.results_root, EXPERIMENT, args.dataset, model_label)
            / (
                f"{TRAIN_DIR}-unfreeze-right" if args.unfreeze_right
                else f"{TRAIN_DIR}-finetune" if args.finetune
                else TRAIN_DIR
            ),
            device=args.device,
            neigs=args.neigs,
            mlp_width=args.mlp_width,
            batch_size=args.batch_size,
            steps=args.steps,
            eval_every=args.eval_every,
            lr=args.lr,
            seed=args.seed,
            overwrite=args.overwrite,
            unfreeze_right=args.unfreeze_right,
            finetune=args.finetune,
            encoder_lr=args.encoder_lr,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"output_dir: {summary['output_dir']}\n"
        f"step: {summary['step']}\n"
        f"evaluations: {summary['evaluations']}"
    )
