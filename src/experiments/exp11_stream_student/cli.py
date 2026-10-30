import argparse
import json
from pathlib import Path

from data_pipline.cli import positive_float, positive_int
from utils.experiment_paths import (
    ANALYSIS_DIR, DEFAULT_RESULTS_ROOT, DEFAULT_STREAM, TRAIN_DIR, run_dir, token_stream_path,
)
from utils.stream_cli import add_stream_analysis_arguments, add_stream_train_arguments


EXPERIMENT = "exp11_stream_student"


def load_student_checkpoint(path: Path, student: dict) -> Path:
    metadata_path = path.with_name("metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError(f"student distillation does not exist: {path}")
    with metadata_path.open() as file:
        metadata = json.load(file)
    if metadata["complete"] is not True:
        raise ValueError(
            f"student distillation is incomplete: {path} "
            f"(step {metadata['steps']} of {metadata['steps_requested']})"
        )
    if metadata["student"] != student:
        raise ValueError(
            f"student checkpoint config {metadata['student']} does not match requested {student}"
        )
    return path


def run_cli(argv: list[str] | None, *, prog: str, experiment: str, distilled: bool, skip: bool) -> None:
    parser = argparse.ArgumentParser(prog=prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    if prog == "exp11":
        distill_parser = subparsers.add_parser(
            "distill", help="Distill a causal student from the teacher",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        distill_parser.add_argument("--stream", default=DEFAULT_STREAM, help="Token stream name")
        distill_parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="Results root")
        distill_parser.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"), help="Local causal language model directory")
        distill_parser.add_argument("--device", default="cuda", help="Torch device")
        distill_parser.add_argument("--model-dim", type=positive_int, default=256, help="Student model dimension")
        distill_parser.add_argument("--layers", type=positive_int, default=4, help="Student Transformer layers")
        distill_parser.add_argument("--heads", type=positive_int, default=4, help="Student attention heads")
        distill_parser.add_argument("--max-positions", type=positive_int, default=96, help="Student window length")
        distill_parser.add_argument("--batch", type=positive_int, default=64, help="Windows per training step")
        distill_parser.add_argument("--steps", type=positive_int, default=20000, help="Distillation steps")
        distill_parser.add_argument("--lr", type=positive_float, default=1e-3, help="Adam learning rate")
        distill_parser.add_argument("--min-lr", type=positive_float, default=1e-5, help="Final cosine learning rate")
        distill_parser.add_argument("--probe-windows", type=positive_int, default=256, help="Fixed probe windows")
        distill_parser.add_argument("--probe-seed", type=int, default=1, help="Fixed probe sampling seed")
        distill_parser.add_argument("--eval-every", type=positive_int, default=1000, help="Steps between probe evaluations")
        distill_parser.add_argument("--checkpoint-every", type=positive_int, default=500, help="Steps between checkpoints")
        distill_parser.add_argument("--log-every", type=positive_int, default=100, help="Steps between training log lines")
        distill_parser.add_argument("--seed", type=int, default=0, help="Training random seed")
        distill_parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs and start over")

    train_parser = subparsers.add_parser(
        "train", help="Learn streaming student NeuralSVD features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_stream_train_arguments(train_parser)
    train_parser.set_defaults(rank=64)
    train_parser.add_argument("--model-dim", type=positive_int, default=256, help="Student model dimension")
    train_parser.add_argument("--layers", type=positive_int, default=4, help="Student Transformer layers")
    train_parser.add_argument("--heads", type=positive_int, default=4, help="Student attention heads")
    train_parser.add_argument("--max-positions", type=positive_int, default=96, help="Maximum student window length")
    train_parser.add_argument("--column-dim", type=positive_int, default=256, help="Column head bottleneck dimension")
    if distilled:
        train_parser.add_argument("--student-checkpoint", type=Path, default=None, help="Distilled student checkpoint (defaults to exp11 distillation output)")
    analysis_parser = subparsers.add_parser(
        "analysis", help=f"Analyze the learned spectrum and write {ANALYSIS_DIR} outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_stream_analysis_arguments(analysis_parser)
    args = parser.parse_args(argv)
    try:
        result_dir = run_dir(args.results_root, experiment, args.stream, args.model_path.name)
        if args.command == "analysis":
            from .features import build_features
            from utils.stream_neural_svd import run_stream_analysis

            print(run_stream_analysis(
                result_dir=result_dir, build_features=build_features, device=args.device,
                reference_future_batch=args.reference_future_batch, overwrite=args.overwrite,
            ))
            return

        student = {
            "model_dim": args.model_dim, "layers": args.layers,
            "heads": args.heads, "max_positions": args.max_positions,
        }
        ckpt = None
        if args.command == "train":
            if max(args.history_tokens, args.future_tokens) > args.max_positions:
                raise ValueError(f"history/future windows exceed --max-positions {args.max_positions}")
            if distilled:
                path = args.student_checkpoint or (
                    run_dir(args.results_root, EXPERIMENT, args.stream, args.model_path.name)
                    / "distill" / "student.pt"
                )
                ckpt = load_student_checkpoint(path, student)

        from utils.inference import CausalLMInference
        from utils.stream_neural_svd import TokenStream

        stream_path = token_stream_path(args.stream)
        stream = TokenStream(stream_path)
        inference = CausalLMInference(args.model_path, args.device)
        if args.command == "distill":
            from .distill import distill_student

            summary = distill_student(
                inference=inference, stream=stream, stream_path=stream_path,
                model_path=args.model_path, student_config=student, device=args.device,
                batch=args.batch, steps=args.steps, lr=args.lr, min_lr=args.min_lr,
                probe_windows=args.probe_windows, probe_seed=args.probe_seed,
                eval_every=args.eval_every, checkpoint_every=args.checkpoint_every,
                log_every=args.log_every, seed=args.seed, output_dir=result_dir / "distill",
                overwrite=args.overwrite,
            )
            print(
                f"output_dir: {summary['output_dir']}\nsteps: {summary['steps']}\n"
                f"final_loss: {summary['final_loss']}\nprobe_kl_last: {summary['probe_kl_last']}"
            )
            return

        import torch

        from .features import build_features
        from utils.stream_neural_svd import train_stream_neural_svd

        torch.manual_seed(args.seed)
        features = build_features(
            {
                "kind": "distilled_student_svd", "student": student, "rank": args.rank,
                "column_dim": args.column_dim, "skip": skip,
                "init": "distilled" if distilled else "random",
                "student_checkpoint": None if ckpt is None else str(ckpt),
                "hidden_size": inference.model.get_output_embeddings().weight.shape[1],
            },
            inference,
        ).to(args.device)
        if ckpt is not None:
            features.trunk.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        summary = train_stream_neural_svd(
            features=features, inference=inference, stream=stream, stream_path=stream_path,
            model_path=args.model_path, device=args.device, rank=args.rank, nesting=args.nesting,
            history_tokens=args.history_tokens, future_tokens=args.future_tokens,
            history_batch=args.history_batch, future_batch=args.future_batch,
            probe_histories=args.probe_histories, probe_futures=args.probe_futures,
            probe_seed=args.probe_seed, max_steps=args.max_steps, eval_every=args.eval_every,
            tol=args.tol, patience=args.patience, convergence_rank=args.convergence_rank,
            checkpoint_every=args.checkpoint_every, lr=args.lr, lr_schedule=args.lr_schedule,
            min_lr=args.min_lr, lr_decay_steps=args.lr_decay_steps, seed=args.seed,
            log_every=args.log_every, output_dir=result_dir / TRAIN_DIR, overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"output_dir: {summary['output_dir']}\n"
        f"stop_reason: {summary['stop_reason']}\n"
        f"steps: {summary['steps']}\n"
        f"pairs_evaluated: {summary['pairs_evaluated']}\n"
        f"final_loss: {summary['final_loss']}"
    )


def main(argv: list[str] | None = None) -> None:
    run_cli(argv, prog="exp11", experiment=EXPERIMENT, distilled=True, skip=True)
