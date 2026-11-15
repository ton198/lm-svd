"""Probe which inputs can predict the right singular functions of the exp5 target.

The exp4 training table minus g (exp5's target) is split by future. The left
basis U is the exact one, taken from the fit futures' columns, so a column's
exact coordinates are t = A'^T U and only the right side is being learned. Each
probe regresses t on some input, and is scored on held-out futures by the
captured energy 1 - ||A' - U t_hat^T||^2 / ||A'||^2, whose upper bound is the
energy of the exact coordinates.

Run from the repository root: uv run python scripts/probe_right_function.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.exp5_gpt2_mlp_table.features import future_only_logits
from utils.inference import CausalLMInference

TABLE_DIR = Path(
    "data/experiment_results/exp4_table_neural_svd/wikitext-large__gpt2-small/inference/train"
)
OUTPUT = Path("data/experiment_results/exp5_gpt2_mlp_table/right_function_probe.json")
MODEL_PATH = "models/gpt2-small"
DEVICE = "cuda"
RANKS = (1, 8, 32, 128)
TEST_FUTURES, VAL_FUTURES = 200, 124
MID_LAYERS = (4, 8)


def main() -> None:
    torch.manual_seed(0)
    tokens = np.load(TABLE_DIR / "tokens.npz")
    futures = torch.from_numpy(tokens["futures"])
    header = torch.from_numpy(tokens["header_token_ids"]).long().to(DEVICE)
    count, top_k = header.shape
    inference = CausalLMInference(MODEL_PATH, DEVICE)

    # ---- inputs from the frozen model, one future at a time (no history) ----
    finals, mids = [], []
    with torch.no_grad():
        for start in range(0, count, 128):
            out = inference.model.base_model(
                input_ids=futures[start:start + 128].to(DEVICE), output_hidden_states=True
            )
            finals.append(out.last_hidden_state.float())
            mids.append(torch.cat([out.hidden_states[layer][:, -1] for layer in MID_LAYERS], -1))
    final = torch.cat(finals)  # (F, tokens, hidden), after ln_f
    x_last, x_mean, x_mid = final[:, -1], final.mean(1), torch.cat(mids).float()
    x_all = final.flatten(1)
    g = future_only_logits(inference, x_last).gather(1, header)  # (F, K)
    unembedding = inference.model.get_output_embeddings().weight.detach().float()
    unembedding = unembedding - unembedding.mean(0)
    e_z = unembedding[header]  # (F, K, hidden)

    # ---- the target: exact coordinates of every column in the exact left basis ----
    residual = torch.from_numpy(np.load(TABLE_DIR / "matrix.npy")).to(DEVICE)
    residual -= g.reshape(-1)
    order = torch.randperm(count, generator=torch.Generator().manual_seed(0)).to(DEVICE)
    test, val, train = (
        order[:TEST_FUTURES],
        order[TEST_FUTURES:TEST_FUTURES + VAL_FUTURES],
        order[TEST_FUTURES + VAL_FUTURES:],
    )
    by_future = residual.reshape(residual.shape[0], count, top_k)
    fit = torch.cat([train, val])
    gram = torch.zeros(residual.shape[0], residual.shape[0], dtype=torch.float64, device=DEVICE)
    for start in range(0, len(fit), 64):
        block = by_future[:, fit[start:start + 64]].flatten(1).double()
        gram += block @ block.T
    left = torch.linalg.eigh(gram).eigenvectors.flip(1)[:, : max(RANKS)].float()  # (N, R)
    target = torch.cat([
        torch.einsum("nfk,nr->fkr", by_future[:, start:start + 64], left)
        for start in range(0, count, 64)
    ])  # (F, K, R)
    norm_sq = by_future.double().square().sum((0, 2))  # (F,)
    del residual, by_future, gram
    torch.cuda.empty_cache()

    def energy(prediction: torch.Tensor, subset: torch.Tensor) -> list[float]:
        """Captured energy on the columns of the futures in `subset`, per rank."""
        result = []
        for rank in RANKS:
            t = target[subset][..., :rank].double()
            p = prediction[subset][..., :rank].double()
            result.append(((2 * (t * p).sum() - p.square().sum()) / norm_sq[subset].sum()).item())
        return result

    def standardize(features: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        mean, std = features[rows].mean(0), features[rows].std(0)
        return torch.cat(
            [(features - mean) / (std + 1e-6), torch.ones_like(features[:, :1])], dim=1
        ).double()

    def ridge(features, outputs, fit_rows, val_rows, score):
        """Ridge on fit_rows with the penalty chosen by score() of the full prediction."""
        x, y = features[fit_rows], outputs[fit_rows].double()
        dual = x.shape[1] > x.shape[0]
        system = x @ x.T if dual else x.T @ x
        scale = system.diagonal().mean()
        eye = torch.eye(system.shape[0], dtype=torch.float64, device=DEVICE)
        best = None
        for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
            solved = torch.linalg.solve(system + alpha * scale * eye, y if dual else x.T @ y)
            weights = x.T @ solved if dual else solved
            prediction = (features @ weights).float()
            value = score(prediction)
            if best is None or value > best[0]:
                best = (value, prediction, alpha)
        return best[1], best[2]

    def train_mlp(features, outputs, fit_rows, score, width=2048, depth=2, epochs=300):
        """MSE regression with early stopping on score() of the full prediction."""
        torch.manual_seed(0)
        layers, size = [], features.shape[1]
        for _ in range(depth):
            layers += [nn.Linear(size, width), nn.GELU()]
            size = width
        model = nn.Sequential(*layers, nn.Linear(size, outputs.shape[1])).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        x, y = features[fit_rows].float(), outputs[fit_rows].float()
        scale = y.square().mean().sqrt()
        batch = min(2048, max(64, len(x) // 8))
        best = None
        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(len(x), device=DEVICE)
            for start in range(0, len(x), batch):
                rows = permutation[start:start + batch]
                loss = (model(x[rows]) - y[rows] / scale).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 5 == 4:
                model.eval()
                with torch.no_grad():
                    prediction = torch.cat([
                        model(features[start:start + 8192].float()) * scale
                        for start in range(0, len(features), 8192)
                    ])
                value = score(prediction)
                if best is None or value > best[0]:
                    best = (value, prediction, epoch + 1)
        return best[1], best[2]

    slots = target.shape[1]
    rank_count = target.shape[2]

    # Per-future probes predict all K slots at once (exp5's form): rows are futures.
    def per_future(name, features, kind, rows=train):
        inputs = standardize(features, rows)
        outputs = target.flatten(1)
        reshape = lambda p: p.reshape(count, slots, rank_count)
        score = lambda p: energy(reshape(p), val)[-1]
        if kind == "ridge":
            prediction, chosen = ridge(inputs, outputs, rows, val, score)
        else:
            prediction, chosen = train_mlp(inputs, outputs, rows, score, depth=1)
        return name, reshape(prediction), chosen

    # Per-column probes predict one column from future and token inputs: rows are columns.
    def columns_of(subset):
        return (subset[:, None] * slots + torch.arange(slots, device=DEVICE)).reshape(-1)

    def per_column(name, features, kind, rows=train):
        fit_rows = columns_of(rows)
        inputs = standardize(features, fit_rows)
        outputs = target.flatten(0, 1)
        reshape = lambda p: p.reshape(count, slots, rank_count)
        score = lambda p: energy(reshape(p), val)[-1]
        if kind == "ridge":
            prediction, chosen = ridge(inputs, outputs, fit_rows, None, score)
        else:
            prediction, chosen = train_mlp(inputs, outputs, fit_rows, score)
        return name, reshape(prediction), chosen

    def spread(features):  # (F, d) -> one row per column
        return features[:, None, :].expand(-1, slots, -1).flatten(0, 1)

    def pca(features, rows, dims):
        centered = features - features[rows].mean(0)
        basis = torch.pca_lowrank(centered[rows], q=dims, center=False)[2]
        return centered @ basis

    token = e_z.flatten(0, 1)
    g_column = g.reshape(-1, 1)
    slot_onehot = torch.eye(slots, device=DEVICE).repeat(count, 1)
    train_columns = columns_of(train)
    interaction = torch.einsum(
        "ci,cj->cij", spread(pca(x_last, train, 48)), pca(token, train_columns, 48)
    ).flatten(1)

    experiments = [
        lambda: per_column("slot index + g only (no future, no token)",
                           torch.cat([slot_onehot, g_column], 1), "ridge"),
        lambda: per_column("token embedding + g only (no future)",
                           torch.cat([token, g_column], 1), "ridge"),
        lambda: per_future("exp5 form: last token -> all slots, linear", x_last, "ridge"),
        lambda: per_future("exp5 form: last token -> all slots, MLP", x_last, "mlp"),
        lambda: per_future("all slots from last + mean, linear",
                           torch.cat([x_last, x_mean], 1), "ridge"),
        lambda: per_future("all slots from last + layers 4, 8, linear",
                           torch.cat([x_last, x_mid], 1), "ridge"),
        lambda: per_future("all slots from every position, linear", x_all, "ridge"),
        lambda: per_column("last token + token embedding + g, linear (additive)",
                           torch.cat([spread(x_last), token, g_column], 1), "ridge"),
        lambda: per_column("last token x token embedding (bilinear, 48 x 48 PCA)",
                           torch.cat([interaction, spread(x_last), token, g_column], 1), "ridge"),
        lambda: per_column("last token + token embedding + g, MLP",
                           torch.cat([spread(x_last), token, g_column], 1), "mlp"),
        lambda: per_column("last + mean + layers 4, 8 + token embedding + g, MLP",
                           torch.cat([spread(torch.cat([x_last, x_mean, x_mid], 1)), token,
                                      g_column], 1), "mlp"),
    ]

    results = {
        "table": str(TABLE_DIR), "ranks": list(RANKS),
        "futures": {"train": len(train), "val": len(val), "test": len(test)},
        "exact": {"train": energy(target, train), "test": energy(target, test)},
        "probes": [],
    }
    header_line = f"{'probe':<58}" + "".join(f"  test r{rank:<4}" for rank in RANKS) + "  train r128"
    print(header_line)
    print(f"{'exact coordinates (upper bound)':<58}"
          + "".join(f"  {value:9.3f}" for value in results["exact"]["test"])
          + f"  {results['exact']['train'][-1]:10.3f}", flush=True)
    for experiment in experiments:
        started = time.perf_counter()
        name, prediction, chosen = experiment()
        row = {"name": name, "test": energy(prediction, test),
               "train": energy(prediction, train), "chosen": chosen}
        results["probes"].append(row)
        print(f"{name:<58}" + "".join(f"  {value:9.3f}" for value in row["test"])
              + f"  {row['train'][-1]:10.3f}   [{chosen}, {time.perf_counter() - started:.0f}s]",
              flush=True)
        del prediction
        torch.cuda.empty_cache()

    # How the two main forms change with the number of training futures.
    print("\nlearning curve (test energy at rank 1 / 8 / 128)")
    results["learning_curve"] = []
    for size in (175, 350, len(train)):
        subset = train[:size]
        for label, run in (
            ("exp5 form, linear", lambda: per_future("", x_last, "ridge", subset)),
            ("token-aware MLP", lambda: per_column(
                "", torch.cat([spread(x_last), token, g_column], 1), "mlp", subset)),
        ):
            _, prediction, _ = run()
            values = energy(prediction, test)
            results["learning_curve"].append({"futures": size, "probe": label, "test": values})
            print(f"  {size:4d} futures  {label:<20}  {values[0]:.3f} / {values[1]:.3f} / {values[3]:.3f}",
                  flush=True)
            del prediction
            torch.cuda.empty_cache()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved {OUTPUT}")


if __name__ == "__main__":
    main()
