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

from . import DEFAULT_LR, EXPERIMENT, SUBNETS, WARMUP_STEPS

DEFAULT_POOL_DIR = Path("data/center_pools/wikitext-103-train__gpt2-small__h64-f32__m64")
DEFAULT_STREAM = Path("data/token_streams/wikitext-103-train.npy")


def build_pool(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    try:
        from .center import build_center_pool

        summary = build_center_pool(
            pool_dir=args.pool_dir, model_path=args.model_path, stream_path=args.stream_path,
            device=args.device, histories=args.histories, futures=args.futures,
            history_tokens=args.history_tokens, future_tokens=args.future_tokens,
            seed=args.seed, overwrite=args.overwrite, chunk=args.chunk,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print("\n".join(f"{key}: {value}" for key, value in summary.items()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp7",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pool_parser = subparsers.add_parser(
        "center-pool",
        help="Compute, for a fixed set of futures, the mean of the model's final state over "
             "fixed histories: the column center that `train --center-pool` subtracts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pool_parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    pool_parser.add_argument(
        "--histories", type=positive_int, default=64,
        help="Fixed histories a column is averaged over, drawn from the stream")
    pool_parser.add_argument(
        "--futures", type=positive_int, default=200000,
        help="Futures of the pool, drawn from the stream; each costs --histories passes")
    pool_parser.add_argument("--history-tokens", type=positive_int, default=64)
    pool_parser.add_argument("--future-tokens", type=positive_int, default=32)
    pool_parser.add_argument("--stream-path", type=Path, default=DEFAULT_STREAM)
    pool_parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
    pool_parser.add_argument("--seed", type=nonnegative_int, default=0)
    pool_parser.add_argument(
        "--chunk", type=positive_int, default=1024, help="Sequences per pass of the model")
    pool_parser.add_argument("--device", default="cuda")
    pool_parser.add_argument("--overwrite", action="store_true")
    train_parser = subparsers.add_parser(
        "train",
        help="Train one pair of sub-networks per singular function, one mode at a time, "
             "with sample-store pairs and compare with the exact SVDs of the exp4 tables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--subnet", choices=SUBNETS, default="scratch",
        help="Encoders of each side of a mode. scratch: a randomly initialised bidirectional "
             "BERT transformer reading the model's token ids, mean-pooled over all positions; "
             "roberta: a trainable copy of the pretrained RoBERTa in --roberta-path, reading "
             "the model's tokens mapped to RoBERTa's ids by token string between <s> and "
             "</s>, mean-pooled over all positions; gpt2 (kept, not in the planned runs): a "
             "trainable copy of the model's causal transformer, read at the last token",
    )
    train_parser.add_argument(
        "--roberta-path", type=Path, default=Path("models/roberta-base"),
        help="Local pretrained model of --subnet roberta: RoBERTa-base, or another model "
             "with RoBERTa's structure and vocabulary such as models/minilmv2-l6-h384 "
             "(scripts/download_models.py --models roberta minilm). Its directory name is "
             "the name of the run's directory; a continued run must repeat it",
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
        # Absent unless given: the default depends on --subnet.
        "--lr", type=positive_float, default=argparse.SUPPRESS,
        help="Learning rate of the heads of a roberta or gpt2 mode, and of all of a scratch "
             f"mode after a linear warmup over the first {WARMUP_STEPS} steps of its stage "
             "(default: " + ", ".join(f"{rate:g} for {kind}" for kind, rate in DEFAULT_LR.items())
             + ")",
    )
    train_parser.add_argument(
        "--encoder-lr", type=positive_float, default=2e-5,
        help="Learning rate of the two encoders of a roberta or gpt2 mode, without warmup",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--store-dir", type=Path, default=DEFAULT_STORE_DIR,
        help="Sample store to draw training pairs from; it must come from --model-path",
    )
    train_parser.add_argument(
        "--center-pool", type=Path, default=None,
        help="A finished center pool (`exp7 center-pool`). The target is then the table "
             "entry minus its column's mean over the pool's fixed histories instead of "
             "minus g, the pairs are pool futures behind fresh stream histories, and "
             "--store-dir is not read. The run directory gets the suffix -center<histories>",
    )
    add_run_arguments(train_parser)
    # The evaluation tables are exp4's, which are built from this sample set.
    train_parser.set_defaults(dataset="wikitext-large")
    train_parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "center-pool":
        build_pool(args, parser)
        return
    if args.min_stage_steps > args.stage_steps:
        parser.error(
            f"--min-stage-steps ({args.min_stage_steps}) must not exceed "
            f"--stage-steps ({args.stage_steps})")
    # A roberta run is named after its pretrained model, so that two of them do not share
    # a directory; the other kinds do not read --roberta-path.
    run_name = args.roberta_path.name if args.subnet == "roberta" else args.subnet
    other_kinds = [kind for kind in SUBNETS if kind != args.subnet]
    if args.subnet == "roberta" and (not run_name or run_name in other_kinds):
        parser.error(
            f"--roberta-path {args.roberta_path} would give the run directory "
            f"train-{run_name}, which belongs to another kind; rename the model directory")
    try:
        from .train import train_exp7

        if args.center_pool is not None:
            from .center import read_pool_config

            run_name += f"-center{read_pool_config(args.center_pool)['histories']}"
        model_label = args.model_path.name
        summary = train_exp7(
            inference_dir=run_dir(
                args.results_root, TABLES_EXPERIMENT, args.dataset, model_label
            ) / INFERENCE_DIR,
            store_dir=args.store_dir,
            model_path=args.model_path,
            output_dir=run_dir(args.results_root, EXPERIMENT, args.dataset, model_label)
            / f"train-{run_name}",
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
            lr=getattr(args, "lr", DEFAULT_LR[args.subnet]),
            encoder_lr=args.encoder_lr,
            seed=args.seed,
            overwrite=args.overwrite,
            roberta_path=args.roberta_path,
            center_pool=args.center_pool,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"output_dir: {summary['output_dir']}\n"
        f"step: {summary['step']}\n"
        f"modes: {summary['modes']}\n"
        f"evaluations: {summary['evaluations']}"
    )
