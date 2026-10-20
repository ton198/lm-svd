import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from utils import (
    INFERENCE_DIR,
    add_analysis_command,
    add_run_arguments,
    dataset_samples_path,
    run_analysis,
    run_dir,
)

from .table import build_exp3_diff


EXPERIMENT = "exp3_L_M_diff"
MATRIX_NAME = "matrix.npy"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inference_parser = subparsers.add_parser(
        "inference",
        help="Build the difference between two L/M matrices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_arguments(inference_parser)
    inference_parser.add_argument("--model-a-path", type=Path, default=Path("models/gpt2-small"))
    inference_parser.add_argument("--model-b-path", type=Path, default=Path("models/gpt2-medium"))
    inference_parser.add_argument("--device", default="cuda")
    inference_parser.add_argument("--history-count", type=positive_int, default=None)
    inference_parser.add_argument("--future-count", type=positive_int, default=None)
    inference_parser.add_argument("--top-k", type=positive_int, default=50)
    inference_parser.add_argument("--overwrite", action="store_true")
    analysis_parser = add_analysis_command(subparsers)
    add_run_arguments(analysis_parser)
    analysis_parser.add_argument(
        "--model-a-path", type=Path, default=Path("models/gpt2-small"),
        help="Model whose run to analyze; only its directory name is used",
    )
    analysis_parser.add_argument(
        "--model-b-path", type=Path, default=Path("models/gpt2-medium"),
        help="Model whose run to analyze; only its directory name is used",
    )
    args = parser.parse_args(argv)
    try:
        result_dir = run_dir(
            args.results_root,
            EXPERIMENT,
            args.dataset,
            f"{args.model_a_path.name}__minus__{args.model_b_path.name}",
        )
        if args.command == "analysis":
            print(
                run_analysis(
                    result_dir=result_dir,
                    matrix_name=MATRIX_NAME,
                    centering=args.centering,
                    device=args.device,
                    overwrite=args.overwrite,
                )
            )
            return
        summary = build_exp3_diff(
            samples_path=dataset_samples_path(args.dataset),
            model_a_path=args.model_a_path,
            model_b_path=args.model_b_path,
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
        f"top_k: {summary['top_k']}\n"
        f"model_a: {summary['model_a_path']}\n"
        f"model_b: {summary['model_b_path']}"
    )
