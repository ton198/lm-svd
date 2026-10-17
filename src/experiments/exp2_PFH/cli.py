import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from utils import INFERENCE_DIR, add_analysis_command, run_analysis

from .feature_logprob import build_exp2_pfh


DEFAULT_SAMPLES = Path(
    "data/history_future_samples/wikitext-103-raw-v1-train-gpt2-history-future.parquet"
)
DEFAULT_RESULT_DIR = Path("data/experiment_results/exp2_PFH/gpt2-small")
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
    inference_parser.add_argument("--samples-path", type=Path, default=DEFAULT_SAMPLES)
    inference_parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
    inference_parser.add_argument("--device", default="cuda")
    inference_parser.add_argument("--history-count", type=positive_int, default=None)
    inference_parser.add_argument("--future-count", type=positive_int, default=None)
    inference_parser.add_argument(
        "--batch-size", type=positive_int, default=64, help="Inference batch size"
    )
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
        Path("data/experiment_results/exp2_PFH") / args.model_path.name
    )
    try:
        summary = build_exp2_pfh(
            samples_path=args.samples_path,
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
