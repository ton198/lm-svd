"""Train exp6 on pairs made in the step and compare it with the exact SVDs.

The pairs, the target (the table entry minus g), the loss, the evaluation
tables and the comparison are exp5's; see experiments.exp5_gpt2_mlp_table.train.
Only the network differs: each side is a trainable copy of the model's
transformer, and the right side is bilinear in the header token's unembedding
row (see model.py). The frozen model still gives g and the header tokens.
"""

from __future__ import annotations

from pathlib import Path

import torch

from experiments.exp5_gpt2_mlp_table.tables import token_features
from experiments.exp5_gpt2_mlp_table.train import (
    fresh_metadata,
    prepare_fresh_run,
    run_training,
)

from .model import build_model


def train_exp6(
    *,
    inference_dir: Path,
    stream_path: Path,
    model_path: Path,
    output_dir: Path,
    device: str,
    neigs: int,
    head_rank: int,
    batch_size: int,
    steps: int,
    eval_every: int,
    lr: float,
    encoder_lr: float,
    seed: int,
    overwrite: bool,
    encode_batch: int = 1024,
    eval_chunk: int = 256,
) -> dict:
    """Train on fresh stream pairs; every eval_every steps compare both exp4 tables.

    Writes the same files as exp5 to output_dir and continues an existing
    checkpoint with the same settings up to `steps`; pass overwrite to start
    again. The heads and the projection of the unembedding rows use lr, the two
    encoders use encoder_lr. head_rank is d, the width of that projection.
    """
    inference, tables, top_k, source = prepare_fresh_run(
        inference_dir=inference_dir, stream_path=stream_path, model_path=model_path,
        device=device, seed=seed, encode_batch=encode_batch,
    )
    config = {
        "inference_dir": str(inference_dir), "model_path": str(model_path),
        "neigs": neigs, "head_rank": head_rank, "batch_size": batch_size, "lr": lr,
        "encoder_lr": encoder_lr, "seed": seed, "optimizer": "Adam", **source.config,
    }

    torch.manual_seed(seed)
    model = build_model(inference, neigs, head_rank).to(device)
    network = model.model
    optimizer = torch.optim.Adam([
        {"params": [*network.left_head.parameters(), *network.right_head.parameters(),
                    *network.token_projection.parameters()], "lr": lr},
        {"params": [*network.left_encoder.parameters(),
                    *network.right_encoder.parameters()], "lr": encoder_lr},
    ])
    return run_training(
        model=model,
        optimizer=optimizer,
        forward=lambda histories, futures, header_ids: model(histories, futures, header_ids),
        features=lambda table: token_features(model, table, eval_chunk),
        config=config,
        metadata={
            "hidden_size": inference.model.config.hidden_size,
            "left_encoder": "trainable copy of the model's transformer, dropout off, "
                            "reading the history; LayerNorm + linear head on the last "
                            "hidden state",
            "right_encoder": "trainable copy of the model's transformer, dropout off, "
                             "reading the future; LayerNorm + linear head on the last "
                             "hidden state gives B(future), neigs x head_rank",
            "right_features": "<B_l(future), P e_z> / sqrt(head_rank); e_z is the frozen "
                              "model's unembedding row of the header token minus the mean "
                              "row, over the root mean square of all entries; P is a "
                              "trained head_rank x hidden projection",
            **fresh_metadata(
                device=device, top_k=top_k, tables=tables, model=model, source=source),
        },
        description=f"rank {neigs}, bilinear right side with head rank {head_rank}",
        tables=tables, top_k=top_k, source=source, output_dir=output_dir, device=device,
        batch_size=batch_size, steps=steps, eval_every=eval_every, seed=seed,
        overwrite=overwrite,
    )
