import argparse
from pathlib import Path

from data_pipline.cli import positive_float, positive_int
from utils.experiment_paths import DEFAULT_STREAM, sample_store_dir, token_stream_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sample-store")
    subparsers = parser.add_subparsers(dest="command", required=True)
    produce = subparsers.add_parser("produce", help="Produce teacher sample blocks")
    status = subparsers.add_parser("status", help="Show sample store progress")
    for command in (produce, status):
        command.add_argument("--stream", default=DEFAULT_STREAM)
        command.add_argument("--model-path", type=Path, default=Path("models/gpt2-small"))
        command.add_argument("--history-tokens", type=positive_int, default=64)
        command.add_argument("--future-tokens", type=positive_int, default=32)
        command.add_argument("--store-dir", type=Path, default=None)

    produce.add_argument("--device", default="cuda")
    produce.add_argument("--block-histories", type=positive_int, default=64)
    produce.add_argument("--block-futures", type=positive_int, default=64)
    produce.add_argument("--max-gib", type=positive_float, default=64.0)
    produce.add_argument("--shards", type=positive_int, default=1)
    produce.add_argument("--shard", type=int, default=0)
    produce.add_argument("--seed", type=int, default=0)
    produce.add_argument("--pair-batch", type=positive_int, default=4096)
    produce.add_argument("--max-blocks", type=positive_int, default=None)
    produce.add_argument("--log-every", type=positive_int, default=100)
    produce.add_argument(
        "--overwrite", action="store_true",
        help="Recreate the store; all producers must be stopped before using this option",
    )
    args = parser.parse_args(argv)

    try:
        store_dir = args.store_dir
        if store_dir is None:
            store_dir = sample_store_dir(
                args.stream, args.model_path.name, args.history_tokens, args.future_tokens
            )
        if args.command == "produce":
            from utils.sample_store import produce_blocks

            result = produce_blocks(
                store_dir=store_dir, model_path=args.model_path,
                stream_path=token_stream_path(args.stream), device=args.device,
                history_tokens=args.history_tokens, future_tokens=args.future_tokens,
                block_histories=args.block_histories, block_futures=args.block_futures,
                max_bytes=int(args.max_gib * 1024**3), shards=args.shards,
                shard=args.shard, seed=args.seed, pair_batch=args.pair_batch,
                max_blocks=args.max_blocks, log_every=args.log_every, overwrite=args.overwrite,
            )
            for key, value in result.items():
                print(f"{key}: {value}")
        else:
            from utils.sample_store import read_shard_states, read_store_config

            config = read_store_config(store_dir)
            print(f"store_dir: {store_dir}")
            for key in ("model_name", "stream_name", "block_histories", "block_futures", "num_slots", "shards"):
                print(f"{key}: {config[key]}")
            for shard, state in enumerate(read_shard_states(store_dir, config["shards"])):
                print(
                    f"shard {shard}: next_sequence {state['next_sequence']} "
                    f"filled {state['filled']}/{config['slots_per_shard']}"
                )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
