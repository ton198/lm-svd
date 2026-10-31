import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from experiments.lingen.cli import EXPERIMENTS
from experiments.exp6_stream_gpt2_mlp.cli import EXPERIMENT as EXP6
from experiments.exp14_paper_lingen.cli import EXPERIMENT as EXP14
from utils.experiment_paths import DEFAULT_RESULTS_ROOT, DEFAULT_STREAM, TRAIN_DIR, run_dir


EXPERIMENT = "exp15_lingen_diagnostics"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp15", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("direct", "Diagnose direct reconstruction"),
        ("exp14", "Diagnose exp14 NeuralSVD coefficients"),
    ):
        subparser = subparsers.add_parser(
            command, help=help_text, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        subparser.add_argument("--stream", default=DEFAULT_STREAM)
        subparser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
        subparser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
        subparser.add_argument("--device", default="cuda")
        subparser.add_argument("--pair-batch", type=positive_int, default=4096)
        subparser.add_argument("--overwrite", action="store_true")
        if command == "direct":
            subparser.add_argument("--eval-stream", default="wikitext-103-validation")
            subparser.add_argument(
                "--experiments", nargs="+", choices=tuple(EXPERIMENTS),
                default=["exp6", "exp7", "exp8", "exp9", "exp10", "exp11", "exp12"],
            )
            subparser.add_argument("--calibration-histories", type=positive_int, default=512)
            subparser.add_argument("--calibration-futures", type=positive_int, default=256)
            subparser.add_argument("--calibration-seed", type=int, default=11)
            subparser.add_argument("--tau-futures", type=positive_int, default=64)
            subparser.add_argument("--validation-pairs", type=positive_int, default=16384)
            subparser.add_argument("--validation-seed", type=int, default=12)
        else:
            subparser.add_argument("--fit-dataset", default="wikitext-large")
            subparser.add_argument("--gram-batch", type=positive_int, default=256)
            subparser.add_argument("--eval-futures", type=positive_int, default=64)
            subparser.add_argument("--anchor-futures", type=positive_int, default=64)

    args = parser.parse_args(argv)
    try:
        result_dir = run_dir(args.results_root, EXPERIMENT, args.stream, args.model_path.name)
        if args.command == "direct":
            from .direct import run_direct

            print(run_direct(
                results_root=args.results_root,
                stream=args.stream,
                eval_stream=args.eval_stream,
                model_path=args.model_path,
                experiments=args.experiments,
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
        else:
            from .surrogate import run_exp14_surrogate

            print(run_exp14_surrogate(
                fit_dir=run_dir(args.results_root, EXP14, args.fit_dataset, args.model_path.name) / "fit",
                nsvd_train_dir=run_dir(args.results_root, EXP6, args.stream, args.model_path.name) / TRAIN_DIR,
                device=args.device,
                pair_batch=args.pair_batch,
                gram_batch=args.gram_batch,
                eval_futures=args.eval_futures,
                anchor_futures=args.anchor_futures,
                output_dir=result_dir / "analysis" / "exp14",
                overwrite=args.overwrite,
            ))
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
