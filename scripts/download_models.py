"""Download the files needed for local GPT-2 inference, not every weight format."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "small": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "medium": ("openai-community/gpt2-medium", "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"),
}
INFERENCE_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download pinned GPT-2 weights, configuration, and tokenizer files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODELS), default=list(MODELS),
        help="Model sizes to download",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
        help="Parent directory for gpt2-small and gpt2-medium",
    )
    args = parser.parse_args(argv)
    for name in dict.fromkeys(args.models):
        repo_id, revision = MODELS[name]
        target = args.output_dir / f"gpt2-{name}"
        print(f"Downloading {repo_id}@{revision} to {target}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target,
            allow_patterns=list(INFERENCE_FILES),
        )
        missing = [name for name in INFERENCE_FILES if not (target / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing inference files in {target}: {', '.join(missing)}")
        print(f"Ready for offline inference: {target}", flush=True)


if __name__ == "__main__":
    main()
