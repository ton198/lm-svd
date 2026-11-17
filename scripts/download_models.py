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
ROBERTA_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
# A 6-layer, 384-wide model distilled from RoBERTa-Large; its repository has no safetensors.
MINILM_FILES = (
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
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
    # exp7's --subnet roberta; downloaded only when named.
    "roberta": (
        "FacebookAI/roberta-base",
        "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        "roberta-base",
        ROBERTA_FILES,
    ),
    # A smaller pretrained model for exp7's --subnet roberta; downloaded only when named.
    "minilm": (
        "nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large",
        "a61886e95cedc4cd2440f71cf9a55320ee1d8e06",
        "minilmv2-l6-h384",
        MINILM_FILES,
    ),
}
DEFAULT_MODELS = ["small", "medium", "large"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download pinned model weights, configuration, and tokenizer files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODELS), default=DEFAULT_MODELS,
        help="Models to download; roberta (RoBERTa-base) and minilm (MiniLMv2 L6-H384), "
             "both for exp7, only when named",
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
