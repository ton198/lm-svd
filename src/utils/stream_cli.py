import argparse
from pathlib import Path

from data_pipline.cli import positive_float, positive_int
from utils.experiment_paths import DEFAULT_RESULTS_ROOT, DEFAULT_STREAM


def add_stream_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stream", default=DEFAULT_STREAM, help="Token stream name; loads data/token_streams/<stream>.npy")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="Results root; run directory is <results-root>/<experiment>/<stream>__<model>")
    parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"), help="Local causal language model directory")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--rank", type=positive_int, default=256, help="Number of learned singular components")
    parser.add_argument("--nesting", choices=("sequential", "joint"), default="sequential", help="NestedLoRA loss nesting")
    parser.add_argument("--history-tokens", type=positive_int, default=64, help="Tokens per history window")
    parser.add_argument("--future-tokens", type=positive_int, default=32, help="Tokens per future window")
    parser.add_argument("--history-batch", type=positive_int, default=64, help="History windows per training step")
    parser.add_argument("--future-batch", type=positive_int, default=64, help="Future windows per training step")
    parser.add_argument("--probe-histories", type=positive_int, default=1024, help="Fixed probe history windows")
    parser.add_argument("--probe-futures", type=positive_int, default=1024, help="Fixed probe future windows")
    parser.add_argument("--probe-seed", type=int, default=1, help="Fixed probe sampling seed")
    parser.add_argument("--max-steps", type=positive_int, default=None, help="Optional training step cap; default: train until convergence")
    parser.add_argument("--eval-every", type=positive_int, default=1000, help="Steps between probe evaluations")
    parser.add_argument("--tol", type=positive_float, default=1e-3, help="Singular-value relative-change convergence tolerance")
    parser.add_argument("--patience", type=positive_int, default=3, help="Consecutive stable evaluations required for convergence")
    parser.add_argument("--convergence-rank", type=positive_int, default=32, help="Leading learned-order singular values checked for convergence (capped at --rank)")
    parser.add_argument("--checkpoint-every", type=positive_int, default=100, help="Steps between checkpoints")
    parser.add_argument("--lr", type=positive_float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine", help="Learning-rate schedule")
    parser.add_argument("--min-lr", type=positive_float, default=1e-5, help="Final cosine learning rate")
    parser.add_argument("--lr-decay-steps", type=positive_int, default=30000, help="Cosine decay length in steps; afterwards lr stays at --min-lr until convergence")
    parser.add_argument("--seed", type=int, default=0, help="Training random seed")
    parser.add_argument("--log-every", type=positive_int, default=100, help="Steps between training log lines")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs and start over")


def add_stream_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stream", default=DEFAULT_STREAM, help="Token stream name used to locate the run")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="Results root; run directory is <results-root>/<experiment>/<stream>__<model>")
    parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"), help="Names the run directory only; analysis loads the model path saved in training metadata")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--reference-future-batch", type=positive_int, default=4, help="Futures per exact-reference forward pass (probe histories times this many sequences)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing analysis outputs")
