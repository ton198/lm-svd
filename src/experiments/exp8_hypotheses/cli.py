"""Command line of exp8: `spectrum` (study A), `center` (study B) and `mean` (study C).

Imports stay light; the study code is imported after the arguments are parsed.
"""

import argparse
import json
from pathlib import Path

from data_pipline.cli import positive_int
from data_pipline.sample_store.cli import nonnegative_int
from experiments.exp4_table_neural_svd.cli import EXPERIMENT as TABLES_EXPERIMENT
from experiments.exp4_table_neural_svd.cli import positive_float
from utils.experiment_paths import INFERENCE_DIR, add_run_arguments, run_dir

from . import EXPERIMENT

DEFAULT_STREAM = Path("data/token_streams/wikitext-103-train.npy")
# --lr of study B for each encoder kind, as in exp7.
CENTER_LR = {"scratch": 1e-4, "roberta": 3e-4, "gpt2": 3e-4, "gpt2-mean": 3e-4}


def _common(parser: argparse.ArgumentParser) -> None:
    add_run_arguments(parser)
    parser.set_defaults(dataset="wikitext-large")
    parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
    parser.add_argument("--device", default="cuda")


def _training(parser: argparse.ArgumentParser, steps: int) -> None:
    parser.add_argument("--stream-path", type=Path, default=DEFAULT_STREAM,
                        help="Token stream the histories and futures are drawn from")
    parser.add_argument("--steps", type=positive_int, default=steps)
    parser.add_argument("--eval-every", type=positive_int, default=1000)
    parser.add_argument("--encoder-lr", type=positive_float, default=2e-5,
                        help="Learning rate of pretrained transformer weights")
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument("--tag", default="",
                        help="Appended to the run directory's name, to keep variants apart")
    parser.add_argument("--overwrite", action="store_true")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp8", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    spectrum = subparsers.add_parser(
        "spectrum", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Study A: how much of each exp4 table's spectrum carries over to other "
             "histories and futures (no training)")
    _common(spectrum)

    center = subparsers.add_parser(
        "center", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Study B: NeuralSVD on fresh grids minus their column means, compared with "
             "the exp4 tables minus theirs")
    _common(center)
    _training(center, steps=30000)
    center.add_argument("--layout", choices=("shared", "independent", "landmark"),
                        default="shared",
                        help="shared: one encoder per side, a mode is an output; "
                             "independent: every mode has its own two encoders (exp7); "
                             "landmark (study D): only the left side is a network, a right "
                             "function is a trained mix of the frozen model's entries behind "
                             "--landmarks fixed histories")
    center.add_argument("--encoder", choices=("gpt2", "gpt2-mean", "roberta", "scratch"),
                        default="gpt2",
                        help="exp7's encoder kinds; gpt2-mean averages the copy's positions "
                             "instead of reading the last one")
    center.add_argument("--landmarks", type=positive_int, default=64,
                        help="Landmark histories of --layout landmark, drawn from the stream")
    center.add_argument("--roberta-path", type=Path, default=Path("models/minilmv2-l6-h384"),
                        help="Pretrained model of --encoder roberta")
    center.add_argument("--nesting", choices=("joint", "sequential"), default="joint",
                        help="NestedLoRA nesting of the modes")
    center.add_argument("--modes", type=positive_int, default=32)
    center.add_argument("--history-batch", type=positive_int, default=32)
    center.add_argument("--future-batch", type=positive_int, default=16)
    center.add_argument("--lr", type=positive_float, default=argparse.SUPPRESS,
                        help="Learning rate of the heads, and of everything for --encoder "
                             "scratch (default: 1e-4 for scratch, 3e-4 otherwise)")

    mean = subparsers.add_parser(
        "mean", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Study C: a copy of the model predicts each column's mean over the histories "
             "from the future alone")
    _common(mean)
    _training(mean, steps=20000)
    mean.add_argument("--prefix-tokens", type=nonnegative_int, default=64,
                      help="Trainable vectors read in front of the future, started at one "
                           "real window; 0 reads the future alone")
    mean.add_argument("--batch-size", type=positive_int, default=128)
    mean.add_argument("--lr", type=positive_float, default=1e-3,
                      help="Learning rate of the prefix vectors")

    args = parser.parse_args(argv)
    model_label = args.model_path.name
    inference_dir = run_dir(
        args.results_root, TABLES_EXPERIMENT, args.dataset, model_label) / INFERENCE_DIR
    root = run_dir(args.results_root, EXPERIMENT, args.dataset, model_label)
    try:
        if args.command == "spectrum":
            from .spectrum import run_spectrum

            summary = run_spectrum(
                inference_dir=inference_dir, model_path=args.model_path, device=args.device,
                output_dir=root / "spectrum")
        elif args.command == "center":
            from .center import train_centered

            encoder_name = args.roberta_path.name if args.encoder == "roberta" else args.encoder
            name = f"center-{args.layout}-{encoder_name}-{args.nesting}-L{args.modes}"
            if args.layout == "landmark":
                name += f"-M{args.landmarks}"
            summary = train_centered(
                inference_dir=inference_dir, stream_path=args.stream_path,
                model_path=args.model_path, output_dir=root / (name + args.tag),
                device=args.device, layout=args.layout, encoder=args.encoder,
                nesting=args.nesting, modes=args.modes, history_batch=args.history_batch,
                future_batch=args.future_batch, steps=args.steps, eval_every=args.eval_every,
                lr=getattr(args, "lr", CENTER_LR[args.encoder]), encoder_lr=args.encoder_lr,
                seed=args.seed, overwrite=args.overwrite, roberta_path=args.roberta_path,
                landmarks=args.landmarks)
        else:
            from .mean import train_mean

            summary = train_mean(
                inference_dir=inference_dir, stream_path=args.stream_path,
                model_path=args.model_path,
                output_dir=root / (f"mean-prefix{args.prefix_tokens}" + args.tag),
                device=args.device, prefix_tokens=args.prefix_tokens,
                batch_size=args.batch_size, steps=args.steps, eval_every=args.eval_every,
                lr=args.lr, encoder_lr=args.encoder_lr, seed=args.seed,
                overwrite=args.overwrite)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
