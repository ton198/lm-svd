import argparse

from data_pipline.cli import positive_int
from utils.experiment_paths import ANALYSIS_DIR, TRAIN_DIR, run_dir, token_stream_path
from utils.stream_cli import add_stream_analysis_arguments, add_stream_train_arguments


EXPERIMENT = "exp8_stream_token_mlp"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exp8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser(
        "train",
        help="Learn streaming NeuralSVD features with token-embedding MLPs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_stream_train_arguments(train_parser)
    train_parser.add_argument("--embedding-dim", type=positive_int, default=256)
    train_parser.add_argument("--mlp-width", type=positive_int, default=1024)
    analysis_parser = subparsers.add_parser(
        "analysis",
        help=f"Compare the learned spectrum with the probe SVD in {ANALYSIS_DIR}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_stream_analysis_arguments(analysis_parser)
    args = parser.parse_args(argv)
    try:
        result_dir = run_dir(
            args.results_root, EXPERIMENT, args.stream, args.model_path.name
        )
        from .features import build_features

        if args.command == "analysis":
            from utils.stream_neural_svd import run_stream_analysis

            print(run_stream_analysis(
                result_dir=result_dir,
                build_features=build_features,
                device=args.device,
                reference_future_batch=args.reference_future_batch,
                overwrite=args.overwrite,
            ))
            return

        import torch

        from utils.inference import CausalLMInference
        from utils.stream_neural_svd import (
            TokenStream,
            centered_unembedding,
            train_stream_neural_svd,
        )

        stream_path = token_stream_path(args.stream)
        stream = TokenStream(stream_path)
        inference = CausalLMInference(args.model_path, args.device)
        token_matrix = centered_unembedding(inference)
        config = {
            "kind": "token_mlp",
            "embedding_dim": args.embedding_dim,
            "mlp_width": args.mlp_width,
            "history_tokens": args.history_tokens,
            "future_tokens": args.future_tokens,
            "vocab_size": inference.vocab_size,
            "hidden_size": token_matrix.shape[1],
            "rank": args.rank,
        }
        del token_matrix
        torch.manual_seed(args.seed)
        features = build_features(config, inference).to(args.device)
        summary = train_stream_neural_svd(
            features=features,
            inference=inference,
            stream=stream,
            stream_path=stream_path,
            model_path=args.model_path,
            device=args.device,
            rank=args.rank,
            nesting=args.nesting,
            history_tokens=args.history_tokens,
            future_tokens=args.future_tokens,
            history_batch=args.history_batch,
            future_batch=args.future_batch,
            probe_histories=args.probe_histories,
            probe_futures=args.probe_futures,
            probe_seed=args.probe_seed,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            tol=args.tol,
            patience=args.patience,
            checkpoint_every=args.checkpoint_every,
            lr=args.lr,
            lr_schedule=args.lr_schedule,
            min_lr=args.min_lr,
            lr_decay_steps=args.lr_decay_steps,
            seed=args.seed,
            log_every=args.log_every,
            output_dir=result_dir / TRAIN_DIR,
            overwrite=args.overwrite,
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
