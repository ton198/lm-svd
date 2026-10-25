import argparse
from pathlib import Path

from data_pipline.cli import positive_float, positive_int
from utils.analysis import CENTERING_CHOICES
from utils.experiment_paths import add_run_arguments, run_dir


EXPERIMENT = "exp5_L_M_neural_svd"
EXP1_EXPERIMENT = "exp1_L_M_table"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analysis_parser = subparsers.add_parser(
        "analysis",
        help="Decompose the exp1 L/M matrix with NeuralSVD and compare with exp1 exact SVD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_arguments(analysis_parser)
    analysis_parser.add_argument(
        "--model-path", type=Path, default=Path("models/gpt2-small"),
        help="Model whose exp1 run to read; only its directory name is used",
    )
    analysis_parser.add_argument(
        "--centering", choices=CENTERING_CHOICES, default="none",
        help="Secondary centering applied before NeuralSVD",
    )
    analysis_parser.add_argument("--device", default="cuda", help="Torch device for NeuralSVD training")
    analysis_parser.add_argument("--rank", type=positive_int, default=64, help="Number of learned components to retain")
    analysis_parser.add_argument("--history-batch", type=positive_int, default=64, help="Histories sampled per training step")
    analysis_parser.add_argument("--future-batch", type=positive_int, default=64, help="Futures sampled per training step")
    analysis_parser.add_argument("--steps", type=positive_int, default=30000, help="Number of training steps")
    analysis_parser.add_argument("--lr", type=positive_float, default=1e-2, help="Initial learning rate")
    analysis_parser.add_argument("--min-lr", type=positive_float, default=1e-4, help="Final cosine learning rate")
    analysis_parser.add_argument("--seed", type=int, default=0, help="Training random seed")
    analysis_parser.add_argument("--log-every", type=positive_int, default=1000, help="Training steps between log lines")
    analysis_parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    args = parser.parse_args(argv)
    try:
        exp1_result_dir = run_dir(
            args.results_root, EXP1_EXPERIMENT, args.dataset, args.model_path.name
        )
        result_dir = run_dir(
            args.results_root, EXPERIMENT, args.dataset, args.model_path.name
        )
        from .analysis import run_analysis

        print(
            run_analysis(
                exp1_result_dir=exp1_result_dir,
                result_dir=result_dir,
                centering=args.centering,
                device=args.device,
                rank=args.rank,
                history_batch=args.history_batch,
                future_batch=args.future_batch,
                steps=args.steps,
                lr=args.lr,
                min_lr=args.min_lr,
                seed=args.seed,
                log_every=args.log_every,
                overwrite=args.overwrite,
            )
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
