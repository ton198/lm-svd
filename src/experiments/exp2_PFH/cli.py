import argparse
from pathlib import Path

from data_pipline.cli import positive_int
from utils.analysis import CENTERING_CHOICES
from utils.experiment_paths import DEFAULT_RESULTS_ROOT, INFERENCE_DIR, run_dir

EXPERIMENT = 'exp2_PFH'


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='exp2', allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    inference = commands.add_parser('inference', help='Sample local WikiText and compute PMI', allow_abbrev=False)
    analysis = commands.add_parser('analysis', help='Analyze an existing PMI matrix', allow_abbrev=False)
    for command in (inference, analysis):
        command.add_argument('--run-name', default='wikitext-direct', help='Name identifying this inference/analysis run')
        command.add_argument('--results-root', type=Path, default=DEFAULT_RESULTS_ROOT)
        command.add_argument('--model-path', type=Path, default=Path('models/gpt2-small'))
        command.add_argument('--device', default='cuda')
        command.add_argument('--overwrite', action='store_true')
    inference.add_argument('--dataset-dir', type=Path, default=Path('data/original/wikitext'))
    inference.add_argument('--config', default='wikitext-103-raw-v1')
    inference.add_argument('--split', default='train')
    inference.add_argument('--history-count', type=positive_int, default=512)
    inference.add_argument('--future-count', type=positive_int, default=512)
    inference.add_argument('--history-tokens', type=positive_int, default=64)
    inference.add_argument('--future-tokens', type=positive_int, default=32)
    inference.add_argument('--seed', type=int, default=0)
    inference.add_argument('--pairing', choices=('independent', 'adjacent'), default='independent')
    inference.add_argument('--batch-size', type=positive_int, default=64)
    analysis.add_argument('--centering', choices=CENTERING_CHOICES, default='none')
    args = parser.parse_args(argv)
    try:
        result_dir = run_dir(args.results_root, EXPERIMENT, args.run_name, args.model_path.name)
        if args.command == 'analysis':
            from .analysis import run_analysis
            print(run_analysis(result_dir=result_dir, centering=args.centering, device=args.device, overwrite=args.overwrite))
            return
        from .feature_logprob import build_exp2_pfh, wikitext_inputs
        output_dir = result_dir / INFERENCE_DIR
        if not args.overwrite and any((output_dir / name).exists() for name in ('pmi.npy', 'metadata.json')):
            raise FileExistsError(f'output already exists: {output_dir}; pass --overwrite to replace it')
        inputs = wikitext_inputs(
            dataset_dir=args.dataset_dir, config=args.config, split=args.split, model_path=args.model_path,
            history_count=args.history_count, future_count=args.future_count,
            history_tokens=args.history_tokens, future_tokens=args.future_tokens, seed=args.seed, pairing=args.pairing,
        )
        summary = build_exp2_pfh(inputs=inputs, model_path=args.model_path, device=args.device,
                                 batch_size=args.batch_size, output_dir=output_dir, overwrite=args.overwrite)
    except (ValueError, FileExistsError, OSError) as error:
        parser.error(str(error))
    print(f"output_dir: {summary['output_dir']}\nmatrix: {summary['rows']} x {summary['columns']}")
