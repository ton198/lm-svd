import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from utils import INFERENCE_DIR, add_analysis_command, run_analysis

from .table import build_exp3_diff


DEFAULT_SAMPLES = Path(
    "data/history_future_samples/wikitext-103-raw-v1-train-gpt2-history-future.parquet"
)
DEFAULT_RESULT_DIR = Path(
    "data/experiment_results/exp3_L_M_diff/gpt2-small__minus__gpt2-medium"
)
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
    inference_parser.add_argument("--samples-path", type=Path, default=DEFAULT_SAMPLES)
    inference_parser.add_argument("--model-a-path", type=Path, default=Path("models/gpt2-small"))
    inference_parser.add_argument("--model-b-path", type=Path, default=Path("models/gpt2-medium"))
    inference_parser.add_argument("--device", default="cuda")
    inference_parser.add_argument("--history-count", type=positive_int, default=None)
    inference_parser.add_argument("--future-count", type=positive_int, default=None)
    inference_parser.add_argument("--top-k", type=positive_int, default=50)
    inference_parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Experiment result directory; outputs go to <result-dir>/inference",
    )
    inference_parser.add_argument("--overwrite", action="store_true")
    add_analysis_command(subparsers, result_dir=DEFAULT_RESULT_DIR)
    args = parser.parse_args(argv)
    if args.command == "analysis":
        try:
            print(
                run_analysis(
                    result_dir=args.result_dir,
                    matrix_name=MATRIX_NAME,
                    centering=args.centering,
                    overwrite=args.overwrite,
                )
            )
        except (ValueError, FileExistsError) as error:
            parser.error(str(error))
        return
    result_dir = args.result_dir or (
        Path("data/experiment_results/exp3_L_M_diff")
        / f"{args.model_a_path.name}__minus__{args.model_b_path.name}"
    )
    try:
        summary = build_exp3_diff(
            samples_path=args.samples_path,
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
