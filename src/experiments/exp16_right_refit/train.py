import json
from pathlib import Path

import torch

from utils.experiment_paths import TRAIN_DIR
from utils.inference import CausalLMInference
from utils.stream_neural_svd import TokenStream, train_stream_neural_svd

from .features import build_features


def run_train(
    *, result_dir: Path, source: str, variant: str, model_path: Path,
    device: str, max_steps: int, lr: float, min_lr: float,
    lr_decay_steps: int, eval_every: int, tol: float, patience: int,
    convergence_rank: int, checkpoint_every: int, log_every: int,
    seed: int, overwrite: bool,
) -> dict:
    from .cli import WIDE_RIGHT, WIDE_STUDENT

    pinned = result_dir / "sources" / source / TRAIN_DIR
    metadata_path = pinned / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"pinned source does not exist: {pinned}; run exp16 pin first")
    meta = json.loads(metadata_path.read_text())
    if Path(meta["model_path"]).name != model_path.name:
        raise ValueError(
            f"pinned {source} model {meta['model_path']} does not match --model-path {model_path}"
        )

    source_model_path = Path(meta["model_path"])
    inference = CausalLMInference(source_model_path, device)
    stream_path = Path(meta["stream_path"])
    stream = TokenStream(stream_path)
    torch.manual_seed(seed)
    right = {} if variant == "continue" else (WIDE_STUDENT if source == "exp11" else WIDE_RIGHT)
    config = {
        "kind": "right_refit",
        "source": source,
        "variant": variant,
        "source_features": meta["features"],
        "source_train_dir": str(pinned),
        "source_steps": meta["steps"],
        "right": dict(right),
    }
    features = build_features(config, inference).to(device)
    features.load_source_state(torch.load(pinned / "features.pt", map_location="cpu", weights_only=True))
    return train_stream_neural_svd(
        features=features,
        inference=inference,
        stream=stream,
        stream_path=stream_path,
        model_path=source_model_path,
        device=device,
        rank=meta["rank"],
        nesting=meta["nesting"],
        history_tokens=meta["history_tokens"],
        future_tokens=meta["future_tokens"],
        history_batch=meta["history_batch"],
        future_batch=meta["future_batch"],
        probe_histories=meta["probe_histories"],
        probe_futures=meta["probe_futures"],
        probe_seed=meta["probe_seed"],
        max_steps=max_steps,
        eval_every=eval_every,
        tol=tol,
        patience=patience,
        convergence_rank=convergence_rank,
        checkpoint_every=checkpoint_every,
        lr=lr,
        lr_schedule="cosine",
        min_lr=min_lr,
        lr_decay_steps=lr_decay_steps,
        seed=seed,
        log_every=log_every,
        output_dir=result_dir / f"{source}-{variant}" / TRAIN_DIR,
        overwrite=overwrite,
    )
