import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from utils.analysis import CENTERING_CHOICES
from utils.experiment_paths import INFERENCE_DIR, add_run_arguments, dataset_samples_path, run_dir


EXPERIMENT = "exp1_L_M_table"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inference_parser = subparsers.add_parser(
        "inference",
        help="Build the L/M matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_arguments(inference_parser)
    inference_parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
    inference_parser.add_argument("--device", default="cuda")
    inference_parser.add_argument("--history-count", type=positive_int, default=None)
    inference_parser.add_argument("--future-count", type=positive_int, default=None)
    inference_parser.add_argument("--top-k", type=positive_int, default=50)
    inference_parser.add_argument("--overwrite", action="store_true")
    analysis_parser = subparsers.add_parser(
        "analysis",
        help="Run SVD, rank, and visualization on the inference matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_arguments(analysis_parser)
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
    try:
        result_dir = run_dir(
            args.results_root, EXPERIMENT, args.dataset, args.model_path.name
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
        from .table import build_exp1_table

        summary = build_exp1_table(
            samples_path=dataset_samples_path(args.dataset),
            model_path=args.model_path,
            device=args.device,
            history_count=args.history_count,
            future_count=args.future_count,
            top_k=args.top_k,
            output_dir=result_dir / INFERENCE_DIR,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"output_dir: {summary['output_dir']}\n"
        f"matrix: {summary['rows']} x {summary['columns']}\n"
        f"top_k: {summary['top_k']}"
    )
