"""Command line of exp7: `train`, which trains the modes one stage at a time.

Imports stay light; the training code is imported after the arguments are parsed.
"""

import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from data_pipline.sample_store.cli import nonnegative_int
from experiments.exp4_table_neural_svd.cli import EXPERIMENT as TABLES_EXPERIMENT
from experiments.exp4_table_neural_svd.cli import positive_float
from experiments.exp5_gpt2_mlp_table.cli import DEFAULT_STORE_DIR
from utils.experiment_paths import INFERENCE_DIR, add_run_arguments, run_dir

from . import EXPERIMENT, SUBNETS, WARMUP_STEPS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp7",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser(
        "train",
        help="Train one pair of sub-networks per singular function, one mode at a time, "
             "with sample-store pairs and compare with the exact SVDs of the exp4 tables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--subnet", choices=SUBNETS, default="gpt2",
        help="gpt2: each side of a mode is a trainable copy of the model's transformer; "
             "scratch: each side is a randomly initialised GPT-2 transformer",
    )
    train_parser.add_argument(
        "--modes", type=positive_int, default=8,
        help="Modes to have when the run ends; a finished run continues with a larger value",
    )
    train_parser.add_argument(
        "--stage-steps", type=positive_int, default=30000,
        help="Most steps of one stage",
    )
    train_parser.add_argument(
        "--min-stage-steps", type=nonnegative_int, default=5000,
        help="Steps of a stage before --patience may end it",
    )
    train_parser.add_argument(
        "--patience", type=nonnegative_int, default=5,
        help="Evaluations without a new best on the selection table (exp4's train table) "
             "that end a stage; 0 trains --stage-steps steps and freezes the last weights",
    )
    train_parser.add_argument(
        "--align-pairs", type=positive_int, default=16384,
        help="Fresh store pairs for the alignment at the end of each stage",
    )
    train_parser.add_argument("--scratch-layers", type=positive_int, default=6)
    train_parser.add_argument("--scratch-width", type=positive_int, default=384)
    train_parser.add_argument("--scratch-heads", type=positive_int, default=6)
    train_parser.add_argument(
        "--batch-size", type=positive_int, default=128,
        help="(history, future) pairs per step; each gives every header token of its future",
    )
    train_parser.add_argument("--eval-every", type=positive_int, default=1000)
    train_parser.add_argument(
        "--lr", type=positive_float, default=3e-4,
        help="Learning rate of the heads of a gpt2 mode, and of all of a scratch mode "
             f"after a linear warmup over the first {WARMUP_STEPS} steps of its stage",
    )
    train_parser.add_argument(
        "--encoder-lr", type=positive_float, default=2e-5,
        help="Learning rate of the two transformer copies of a gpt2 mode",
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
    if args.min_stage_steps > args.stage_steps:
        parser.error(
            f"--min-stage-steps ({args.min_stage_steps}) must not exceed "
            f"--stage-steps ({args.stage_steps})")
    try:
        from .train import train_exp7

        model_label = args.model_path.name
        summary = train_exp7(
            inference_dir=run_dir(
                args.results_root, TABLES_EXPERIMENT, args.dataset, model_label
            ) / INFERENCE_DIR,
            store_dir=args.store_dir,
            model_path=args.model_path,
            output_dir=run_dir(args.results_root, EXPERIMENT, args.dataset, model_label)
            / f"train-{args.subnet}",
            device=args.device,
            subnet=args.subnet,
            modes=args.modes,
            stage_steps=args.stage_steps,
            min_stage_steps=args.min_stage_steps,
            patience=args.patience,
            align_pairs=args.align_pairs,
            scratch_layers=args.scratch_layers,
            scratch_width=args.scratch_width,
            scratch_heads=args.scratch_heads,
            batch_size=args.batch_size,
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
        f"modes: {summary['modes']}\n"
        f"evaluations: {summary['evaluations']}"
    )
