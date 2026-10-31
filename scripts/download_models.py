"""Download the files needed for local model inference, not every weight format."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


GPT2_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
MINILM_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)
MODELS = {
    "small": (
        "openai-community/gpt2",
        "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "gpt2-small",
        GPT2_FILES,
    ),
    "medium": (
        "openai-community/gpt2-medium",
        "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        "gpt2-medium",
        GPT2_FILES,
    ),
    "large": (
        "openai-community/gpt2-large",
        "32b71b12589c2f8d625668d2335a01cac3249519",
        "gpt2-large",
        GPT2_FILES,
    ),
    "minilm": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "all-MiniLM-L6-v2",
        MINILM_FILES,
    ),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download pinned model weights, configuration, and tokenizer files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODELS), default=list(MODELS),
        help="Models to download",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
        help="Parent directory for the downloaded model directories",
    )
    args = parser.parse_args(argv)
    for name in dict.fromkeys(args.models):
        repo_id, revision, local_dir_name, files = MODELS[name]
        target = args.output_dir / local_dir_name
        print(f"Downloading {repo_id}@{revision} to {target}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target,
            allow_patterns=list(files),
        )
        missing = [name for name in files if not (target / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing inference files in {target}: {', '.join(missing)}")
        print(f"Ready for offline inference: {target}", flush=True)


if __name__ == "__main__":
    main()
