"""CLI registration for data-pipline sample-store; imports stay lightweight."""

from pathlib import Path

from .cli import positive_float, positive_int

DEFAULT_STREAM = 'wikitext-103-train'


def register(subparsers):
    parser = subparsers.add_parser('sample-store', help='Generate and inspect next-token training samples')
    commands = parser.add_subparsers(dest='store_command', required=True)
    produce = commands.add_parser('produce', help='Continuously generate paired windows and raw logits')
    status = commands.add_parser('status', help='Show published sample counts')
    for command in (produce, status):
        command.add_argument('--stream', default=DEFAULT_STREAM)
        command.add_argument('--model-path', type=Path, default=Path('models/gpt2-small'))
        command.add_argument('--history-tokens', type=positive_int, default=64)
        command.add_argument('--future-tokens', type=positive_int, default=32)
        command.add_argument('--store-dir', type=Path, default=None)
    produce.add_argument('--device', default='cuda')
    produce.add_argument('--samples-per-block', type=positive_int, default=64)
    produce.add_argument('--max-gib', type=positive_float, default=64.0)
    produce.add_argument('--shards', type=positive_int, default=1)
    produce.add_argument('--shard', type=int, default=0)
    produce.add_argument('--seed', type=int, default=0)
    produce.add_argument('--batch-size', type=positive_int, default=64)
    produce.add_argument('--max-blocks', type=positive_int, default=None)
    produce.add_argument('--log-every', type=positive_int, default=100)
    produce.add_argument('--overwrite', action='store_true', help='Rebuild explicitly; stop all producers and samplers first')


def run(args):
    from .sample_store import read_shard_states, read_store_config
    from utils.experiment_paths import validate_dataset_name

    validate_dataset_name(args.stream)
    store_dir = args.store_dir or Path('data/sample_stores') / (
        f'{args.stream}__{args.model_path.name}__h{args.history_tokens}-f{args.future_tokens}'
    )
    if args.store_command == 'produce':
        from .sample_store_producer import produce_blocks
        stream_path = Path('data/token_streams') / f'{args.stream}.npy'
        result = produce_blocks(
            store_dir=store_dir, model_path=args.model_path, stream_path=stream_path, device=args.device,
            history_tokens=args.history_tokens, future_tokens=args.future_tokens,
            samples_per_block=args.samples_per_block, max_bytes=int(args.max_gib * 1024**3),
            shards=args.shards, shard=args.shard, seed=args.seed, batch_size=args.batch_size,
            max_blocks=args.max_blocks, log_every=args.log_every, overwrite=args.overwrite,
        )
        for key, value in result.items():
            print(f'{key}: {value}')
    else:
        config = read_store_config(store_dir)
        print(f'store_dir: {store_dir}')
        for key in ('format', 'model_name', 'stream_name', 'samples_per_block', 'num_slots', 'shards'):
            print(f'{key}: {config[key]}')
        total = 0
        for shard, state in enumerate(read_shard_states(store_dir, config['shards'])):
            count = state['filled'] * config['samples_per_block']
            total += count
            print(f"shard {shard}: next_sequence {state['next_sequence']} filled {state['filled']}/{config['slots_per_shard']} samples {count}")
        print(f'published_samples: {total}')
