import argparse
from pathlib import Path

from data_pipline.cli import positive_int


DEFAULT_SAMPLES = Path(
    "data/history_future_samples/wikitext-103-raw-v1-train-gpt2-history-future.parquet"
)

from .feature_logprob import build_exp2_pfh


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.exp2_PFH",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples-path", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history-count", type=positive_int, default=None)
    parser.add_argument("--future-count", type=positive_int, default=None)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output_dir = args.output_dir or (
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
            output_dir=output_dir,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"output_dir: {summary['output_dir']}\n"
        f"matrix: {summary['rows']} x {summary['columns']}"
    )
