import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from experiments.analysis import add_analysis_commands, run_analysis_command

from .table import build_exp1_table


DEFAULT_SAMPLES = Path(
    "data/history_future_samples/wikitext-103-raw-v1-train-gpt2-history-future.parquet"
)
DEFAULT_RESULT_DIR = Path("data/experiment_results/exp1_L_M_table/gpt2-small")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix_parser = subparsers.add_parser(
        "get_matrix",
        help="Build the L/M matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    matrix_parser.add_argument("--samples-path", type=Path, default=DEFAULT_SAMPLES)
    matrix_parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
    matrix_parser.add_argument("--device", default="cuda")
    matrix_parser.add_argument("--history-count", type=positive_int, default=None)
    matrix_parser.add_argument("--future-count", type=positive_int, default=None)
    matrix_parser.add_argument("--top-k", type=positive_int, default=50)
    matrix_parser.add_argument("--output-dir", type=Path, default=None)
    matrix_parser.add_argument("--overwrite", action="store_true")
    add_analysis_commands(subparsers, result_dir=DEFAULT_RESULT_DIR, matrix_name="matrix.npy")
    args = parser.parse_args(argv)
    if args.command != "get_matrix":
        try:
            print(run_analysis_command(args))
        except (ValueError, FileExistsError) as error:
            parser.error(str(error))
        return
    output_dir = args.output_dir or (
        Path("data/experiment_results/exp1_L_M_table") / args.model_path.name
    )
    try:
        summary = build_exp1_table(
            samples_path=args.samples_path,
            model_path=args.model_path,
            device=args.device,
            history_count=args.history_count,
            future_count=args.future_count,
            top_k=args.top_k,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"output_dir: {summary['output_dir']}\n"
        f"matrix: {summary['rows']} x {summary['columns']}\n"
        f"top_k: {summary['top_k']}"
    )
