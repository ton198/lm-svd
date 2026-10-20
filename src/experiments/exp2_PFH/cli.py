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

from .feature_logprob import build_exp2_pfh


EXPERIMENT = "exp2_PFH"
MATRIX_NAME = "feature_log_prob.npy"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inference_parser = subparsers.add_parser(
        "inference",
        help="Build the PFH matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_arguments(inference_parser)
    inference_parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
    inference_parser.add_argument("--device", default="cuda")
    inference_parser.add_argument("--history-count", type=positive_int, default=None)
    inference_parser.add_argument("--future-count", type=positive_int, default=None)
    inference_parser.add_argument(
        "--batch-size", type=positive_int, default=64, help="Inference batch size"
    )
    inference_parser.add_argument("--overwrite", action="store_true")
    analysis_parser = add_analysis_command(subparsers)
    add_run_arguments(analysis_parser)
    analysis_parser.add_argument(
        "--model-path", type=Path, default=Path("models/gpt2-small"),
        help="Model whose run to analyze; only its directory name is used",
    )
    args = parser.parse_args(argv)
    try:
        result_dir = run_dir(
            args.results_root, EXPERIMENT, args.dataset, args.model_path.name
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
        summary = build_exp2_pfh(
            samples_path=dataset_samples_path(args.dataset),
            model_path=args.model_path,
            device=args.device,
            history_count=args.history_count,
            future_count=args.future_count,
            batch_size=args.batch_size,
            output_dir=result_dir / INFERENCE_DIR,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"output_dir: {summary['output_dir']}\n"
        f"matrix: {summary['rows']} x {summary['columns']}"
    )
