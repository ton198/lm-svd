"""CLI registration for data-pipline sample-store; imports stay lightweight."""

from pathlib import Path
import argparse

from ..cli import positive_float, positive_int

DEFAULT_STREAM = 'wikitext-103-train'


def nonnegative_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('must be a non-negative integer') from error
    if number < 0:
        raise argparse.ArgumentTypeError('must be a non-negative integer')
    return number


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
    produce.add_argument('--worker-id', type=nonnegative_int, required=True,
                         help='Permanent worker ID; reuse it to resume unfinished blocks')
    produce.add_argument('--seed', type=int, default=0)
    produce.add_argument('--batch-size', type=positive_int, default=64)
    produce.add_argument('--max-blocks', type=positive_int, default=None)
    produce.add_argument('--log-every', type=positive_int, default=100)
    produce.add_argument('--overwrite', action='store_true', help='Rebuild explicitly; stop all producers and samplers first')


def run(args):
    from .store import read_store_config
    from .state import read_store_state
    from utils.experiment_paths import validate_dataset_name

    validate_dataset_name(args.stream)
    store_dir = args.store_dir or Path('data/sample_stores') / (
        f'{args.stream}__{args.model_path.name}__h{args.history_tokens}-f{args.future_tokens}'
    )
    if args.store_command == 'produce':
        from .producer import produce_blocks
        stream_path = Path('data/token_streams') / f'{args.stream}.npy'
        result = produce_blocks(
            store_dir=store_dir, model_path=args.model_path, stream_path=stream_path, device=args.device,
            history_tokens=args.history_tokens, future_tokens=args.future_tokens,
            samples_per_block=args.samples_per_block, max_bytes=int(args.max_gib * 1024**3),
            worker_id=args.worker_id, seed=args.seed, batch_size=args.batch_size,
            max_blocks=args.max_blocks, log_every=args.log_every, overwrite=args.overwrite,
        )
        for key, value in result.items():
            print(f'{key}: {value}')
    else:
        config = read_store_config(store_dir)
        print(f'store_dir: {store_dir}')
        for key in ('format', 'model_name', 'stream_name', 'samples_per_block', 'num_slots'):
            print(f'{key}: {config[key]}')
        state = read_store_state(store_dir)
        print(f"published_blocks: {len(state['published'])}/{config['num_slots']}")
        print(f"next_commit: {state['next_commit']}")
        for worker_id, worker in sorted(state['workers'].items(), key=lambda item: int(item[0])):
            print(f"worker {worker_id}: slot {worker['slot']} sequence {worker['sequence']}")
        total = len(state['published']) * config['samples_per_block']
        print(f'published_samples: {total}')
