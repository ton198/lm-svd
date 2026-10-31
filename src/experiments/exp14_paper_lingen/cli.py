import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from utils.experiment_paths import DEFAULT_RESULTS_ROOT, dataset_samples_path, run_dir


EXPERIMENT = "exp14_paper_lingen"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp14", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("fit", "Fit LinGen coefficients"),
        ("generate", "Generate and evaluate LinGen continuations"),
        ("report", "Summarize generation results"),
    ):
        subparser = subparsers.add_parser(
            command, help=help_text, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        subparser.add_argument("--dataset", default="wikitext-large")
        subparser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
        subparser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
        subparser.add_argument("--overwrite", action="store_true")
        if command == "fit":
            subparser.add_argument("--device", default="cuda")
            subparser.add_argument("--basis-histories", type=positive_int, default=2048)
            subparser.add_argument("--futures", type=positive_int, default=1024)
            subparser.add_argument("--targets", type=positive_int, default=50)
            subparser.add_argument("--top-k", type=positive_int, default=50)
            subparser.add_argument("--nonsense-seed", type=int, default=0)
            subparser.add_argument("--pair-batch", type=positive_int, default=4096)
            subparser.add_argument("--gram-batch", type=positive_int, default=256)
            subparser.add_argument(
                "--nsvd-train-dir", type=Path,
                default=Path("data/experiment_results/exp6_stream_gpt2_mlp/wikitext-103-train__gpt2-small/train"),
            )
        elif command == "generate":
            subparser.add_argument("--device", default="cuda")
            subparser.add_argument("--eval-model-path", type=Path, default=Path("models/gpt2-small"))
            subparser.add_argument("--samples-per-target", type=positive_int, default=5)
            subparser.add_argument("--steps", type=positive_int, default=15)
            subparser.add_argument("--short-context", type=positive_int, default=5)
            subparser.add_argument("--basis-chunk", type=positive_int, default=2)
            subparser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    try:
        result_dir = run_dir(args.results_root, EXPERIMENT, args.dataset, args.model_path.name)
        if args.command == "fit":
            from .lingen import fit_coefficients

            fit_coefficients(
                samples_path=dataset_samples_path(args.dataset),
                model_path=args.model_path,
                device=args.device,
                basis_histories=args.basis_histories,
                targets=args.targets,
                future_count=args.futures,
                top_k=args.top_k,
                nonsense_seed=args.nonsense_seed,
                pair_batch=args.pair_batch,
                nsvd_train_dir=args.nsvd_train_dir,
                gram_batch=args.gram_batch,
                output_dir=result_dir / "fit",
                overwrite=args.overwrite,
            )
            print(result_dir / "fit")
        elif args.command == "generate":
            from .lingen import generate

            print(generate(
                fit_dir=result_dir / "fit",
                eval_model_path=args.eval_model_path,
                device=args.device,
                samples_per_target=args.samples_per_target,
                steps=args.steps,
                short_context=args.short_context,
                basis_chunk=args.basis_chunk,
                seed=args.seed,
                output_dir=result_dir / "lingen" / args.eval_model_path.name,
                overwrite=args.overwrite,
            ))
        else:
            from .lingen import report

            print(report(
                lingen_dir=result_dir / "lingen",
                output_dir=result_dir / "analysis",
                overwrite=args.overwrite,
            ))
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
