import argparse
import importlib
from pathlib import Path

from data_pipline.cli import positive_int
from experiments.exp6_stream_gpt2_mlp.cli import EXPERIMENT as EXP6
from experiments.exp7_stream_transformer.cli import EXPERIMENT as EXP7
from experiments.exp8_stream_token_mlp.cli import EXPERIMENT as EXP8
from experiments.exp9_stream_sentence_mlp.cli import EXPERIMENT as EXP9
from experiments.exp10_stream_sentence_mlp_joint.cli import EXPERIMENT as EXP10
from experiments.exp11_stream_student.cli import EXPERIMENT as EXP11
from experiments.exp12_stream_student_nodistill.cli import EXPERIMENT as EXP12
from experiments.exp13_stream_student_noskip.cli import EXPERIMENT as EXP13
from utils.experiment_paths import (
    DEFAULT_RESULTS_ROOT,
    DEFAULT_STREAM,
    LINGEN_DIR,
    TRAIN_DIR,
    run_dir,
    token_stream_path,
)


EXPERIMENTS = {
    "exp6": (EXP6, "experiments.exp6_stream_gpt2_mlp.features"),
    "exp7": (EXP7, "experiments.exp7_stream_transformer.features"),
    "exp8": (EXP8, "experiments.exp8_stream_token_mlp.features"),
    "exp9": (EXP9, "experiments.exp9_stream_sentence_mlp.features"),
    "exp10": (EXP10, "experiments.exp9_stream_sentence_mlp.features"),
    "exp11": (EXP11, "experiments.exp11_stream_student.features"),
    "exp12": (EXP12, "experiments.exp11_stream_student.features"),
    "exp13": (EXP13, "experiments.exp11_stream_student.features"),
}


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--stream", default=DEFAULT_STREAM, help="Training stream name used to locate the run")
    parser.add_argument("--eval-stream", default=None, help="Evaluation stream name; defaults to the training stream")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="Results root")
    parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"), help="Names the run directory only; the teacher path comes from training metadata")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lingen", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    onestep = subparsers.add_parser(
        "onestep", help="Compare distributions on independent history/future pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_arguments(onestep)
    onestep.add_argument("--pairs", type=positive_int, default=4096, help="Number of pairs")
    onestep.add_argument("--batch-size", type=positive_int, default=128, help="Pairs per batch")

    autoregressive = subparsers.add_parser(
        "autoregressive", help="Compare generated trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_arguments(autoregressive)
    autoregressive.add_argument("--sequences", type=positive_int, default=64, help="Number of sequences")
    autoregressive.add_argument("--steps", type=positive_int, default=96, help="Generated tokens per sequence")

    args = parser.parse_args(argv)
    try:
        dir_name, module = EXPERIMENTS[args.experiment]
        result_dir = run_dir(args.results_root, dir_name, args.stream, args.model_path.name)
        eval_name = args.eval_stream or args.stream
        eval_path = token_stream_path(eval_name)
        output_dir = result_dir / LINGEN_DIR / eval_name / args.command
        build_features = importlib.import_module(module).build_features
        from utils.lingen import run_lingen_autoregressive, run_lingen_onestep

        common = {
            "train_dir": result_dir / TRAIN_DIR,
            "build_features": build_features,
            "device": args.device,
            "eval_stream_path": eval_path,
            "output_dir": output_dir,
            "seed": args.seed,
            "overwrite": args.overwrite,
        }
        if args.command == "onestep":
            result = run_lingen_onestep(**common, pairs=args.pairs, batch_size=args.batch_size)
        else:
            result = run_lingen_autoregressive(**common, sequences=args.sequences, steps=args.steps)
        print(result)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
