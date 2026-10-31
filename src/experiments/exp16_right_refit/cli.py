import argparse
from pathlib import Path

from data_pipline.cli import positive_float, positive_int
from experiments.exp15_lingen_diagnostics.cli import add_direct_arguments
from experiments.lingen.cli import EXPERIMENTS
from utils.experiment_paths import DEFAULT_RESULTS_ROOT, DEFAULT_STREAM, TRAIN_DIR, run_dir


EXPERIMENT = "exp16_right_refit"
SOURCES = ("exp6", "exp10", "exp11")
VARIANTS = ("continue", "wide")
WIDE_RIGHT = {"width": 2048, "blocks": 3, "block_hidden": 4096, "bottleneck": 1024}
WIDE_STUDENT = {"head_width": 1024}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp16", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("pin", "Pin source training snapshots"),
        ("train", "Train a right singular function"),
        ("evaluate", "Evaluate source and refitted right functions"),
    ):
        subparser = subparsers.add_parser(
            command, help=help_text, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        subparser.add_argument("--stream", default=DEFAULT_STREAM)
        subparser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
        subparser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
        if command == "pin":
            subparser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
        elif command == "train":
            subparser.add_argument("--source", choices=SOURCES, required=True)
            subparser.add_argument("--variant", choices=VARIANTS, required=True)
            subparser.add_argument("--device", default="cuda")
            subparser.add_argument("--max-steps", type=positive_int, default=30000)
            subparser.add_argument("--lr", type=positive_float, default=1e-3)
            subparser.add_argument("--min-lr", type=positive_float, default=1e-5)
            subparser.add_argument("--lr-decay-steps", type=positive_int, default=30000)
            subparser.add_argument("--eval-every", type=positive_int, default=1000)
            subparser.add_argument("--tol", type=positive_float, default=1e-3)
            subparser.add_argument("--patience", type=positive_int, default=3)
            subparser.add_argument("--convergence-rank", type=positive_int, default=32)
            subparser.add_argument("--checkpoint-every", type=positive_int, default=100)
            subparser.add_argument("--log-every", type=positive_int, default=100)
            subparser.add_argument("--seed", type=int, default=0)
            subparser.add_argument("--overwrite", action="store_true")
        else:
            add_direct_arguments(subparser)
            subparser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
            subparser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
            subparser.add_argument("--device", default="cuda")
            subparser.add_argument("--pair-batch", type=positive_int, default=4096)
            subparser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    try:
        result_dir = run_dir(args.results_root, EXPERIMENT, args.stream, args.model_path.name)
        if args.command == "pin":
            from .pin import pin_source

            for source in args.sources:
                source_train = run_dir(
                    args.results_root, EXPERIMENTS[source][0], args.stream, args.model_path.name
                ) / TRAIN_DIR
                pin_source(source_train, result_dir / "sources" / source / TRAIN_DIR)
        elif args.command == "train":
            from .train import run_train

            summary = run_train(
                result_dir=result_dir, source=args.source, variant=args.variant,
                model_path=args.model_path, device=args.device, max_steps=args.max_steps,
                lr=args.lr, min_lr=args.min_lr, lr_decay_steps=args.lr_decay_steps,
                eval_every=args.eval_every, tol=args.tol, patience=args.patience,
                convergence_rank=args.convergence_rank, checkpoint_every=args.checkpoint_every,
                log_every=args.log_every, seed=args.seed, overwrite=args.overwrite,
            )
            print(
                f"output_dir: {summary['output_dir']}\n"
                f"stop_reason: {summary['stop_reason']}\n"
                f"steps: {summary['steps']}\n"
                f"pairs_evaluated: {summary['pairs_evaluated']}\n"
                f"final_loss: {summary['final_loss']}"
            )
        else:
            from experiments.exp15_lingen_diagnostics.direct import run_direct

            runs = {}
            for source in args.sources:
                runs[f"{source}-source"] = (
                    result_dir / "sources" / source / TRAIN_DIR,
                    EXPERIMENTS[source][1],
                )
                for variant in args.variants:
                    runs[f"{source}-{variant}"] = (
                        result_dir / f"{source}-{variant}" / TRAIN_DIR,
                        "experiments.exp16_right_refit.features",
                    )
            reference_metadata = (
                run_dir(args.results_root, EXPERIMENTS["exp6"][0], args.stream, args.model_path.name)
                / "analysis" / "metadata.json"
            )
            print(run_direct(
                runs=runs,
                reference_metadata=reference_metadata,
                stream=args.stream,
                eval_stream=args.eval_stream,
                model_path=args.model_path,
                device=args.device,
                pair_batch=args.pair_batch,
                calibration_histories=args.calibration_histories,
                calibration_futures=args.calibration_futures,
                calibration_seed=args.calibration_seed,
                tau_futures=args.tau_futures,
                validation_pairs=args.validation_pairs,
                validation_seed=args.validation_seed,
                output_dir=result_dir / "analysis" / "direct",
                overwrite=args.overwrite,
            ))
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
