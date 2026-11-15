# exp5: GPT-2 hidden state + MLP on the L_M table — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add experiment `exp5`, which trains two MLPs on frozen GPT-2 last-token hidden states with the NestedLoRA loss on sample-store pairs, with the first mode fixed to the future-only logit, and compares the learned modes with exact SVDs of the exp4 tables.

**Architecture:** A new package `src/experiments/exp5_gpt2_mlp_table/` with five small modules: GPT-2 side quantities (`features.py`), the trainable network (`model.py`), the paired loss (`loss.py`), evaluation tables (`tables.py`) and the training loop (`train.py`), plus `cli.py`. It reads exp4's inference output and the sample store, and reuses exp4's `Table`, `table_columns`, `compare_with_exact_svd` and `format_comparison` unchanged.

**Tech Stack:** Python 3.14, PyTorch, transformers (local GPT-2), NumPy, `unittest`, uv.

**Spec:** `docs/superpowers/specs/2026-11-15-exp5-gpt2-mlp-table-design.md` (Chinese). Read it before starting; this plan argues from it.

## Global Constraints

- Run everything from the repository root. Tests: `uv run python -m unittest discover -s tests -p "test_exp5*.py" -v`. Full suite: `uv run python -m unittest discover -s tests`.
- Do not change the behavior, outputs or checkpoint format of exp4, `svd_inference` or `data_pipline.sample_store`. Their existing tests must pass unchanged.
- GPT-2 is frozen: no GPT-2 parameter receives a gradient and no GPT-2 weight is written to `checkpoint.pt`.
- The right network receives only the last-token hidden state of `f`. It never receives header token ids, `h`, or `h + f`.
- Hidden state means `CausalLMInference.final_hidden_states(ids)[:, -1]` (last layer, after `ln_f`).
- Column order everywhere is future-major, then header slot `j`: column index `future * top_k + j`.
- `top_k` is read from the exp4 validation table (`val/tokens.npz`, `header_token_ids.shape[1]`); never hard-code 50.
- Defaults: `--neigs 128`, `--mlp-width 4096`, `--batch-size 256`, `--steps 100000`, `--eval-every 1000`, `--lr 3e-4`, `--seed 42`, Adam, joint nesting, `--dataset wikitext-large`, `--model-path models/gpt2-small`, `--store-dir data/sample_stores/wikitext-103-train__gpt2-small__h64-f32`.
- Header-token tolerance for evaluation tables: `1e-3`.
- CLI modules must not import `torch` or `transformers` at import time (see `tests/test_cli_startup.py`).
- Code style follows exp4: module docstrings, type hints, `write_atomic` for every output file, `ValueError`/`FileExistsError` for user errors.
- Commit on a branch named `exp5-gpt2-mlp-table`, not on `main`. Commit messages follow the repository style, e.g. `exp5: add the paired NestedLoRA loss`.

## Review Focus

Conditions the spec implies that are most likely to bite, each pinned by a test in the task that owns the code:

1. **The store was produced by a different model that happens to share vocabulary and window sizes.** Training would silently mix that model's `K` with this model's `g`. Expected: `ValueError` before training (Task 5, `test_rejects_a_store_from_other_model_files`).
2. **Stored header tokens differ from a recomputed top-k only in the order of near-tied tokens.** Expected: the table loads and `g` follows the stored order (Task 4, `test_reordered_headers_load_and_g_follows_the_stored_order`); a stored token outside the top-k still fails (Task 4, `test_a_header_outside_the_top_k_is_rejected`).
3. **A batch of one pair.** Expected: finite loss and gradients (Task 3, `test_single_pair_batch_is_finite`).
4. **The store exists but no block is published yet** (producer just started). Expected: the trainer waits instead of failing (Task 5, `test_waits_for_the_first_block`).
5. **Resuming with `--steps` not above the checkpoint's step, or `--eval-every` above `--steps`.** Expected: the first returns at once with the history intact; the second evaluates once at the final step (Task 5, `test_resume_and_step_edge_cases`).

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/experiments/exp5_gpt2_mlp_table/__init__.py` | Constants only: `EXPERIMENT`, `TRAIN_DIR`, `HEADER_TOLERANCE` |
| `src/experiments/exp5_gpt2_mlp_table/__main__.py` | `python -m` entry: calls `cli.main()` |
| `.../features.py` | Everything computed from frozen GPT-2: hidden states, future-only logits, header tokens and `g`, pair targets |
| `.../model.py` | `Exp5Network` (two MLPs) and `build_model` |
| `.../loss.py` | `paired_loss` |
| `.../tables.py` | `EvalTable`, `load_eval_table`, `eval_features` |
| `.../train.py` | `check_store`, `draw_pairs`, `sampler_seed`, `train_exp5` |
| `.../cli.py` | `exp5 train` argument parsing |
| `tests/test_exp5_gpt2_mlp_table.py` | All exp5 tests and the shared fixture |
| `bluehive_run/exp5-train.sbatch` | Slurm launcher |
| Modify `pyproject.toml`, `tests/test_cli_startup.py`, `README.md` | Script entry, startup test, documentation |

---

### Task 1: GPT-2 side quantities (`features.py`) and the shared test fixture

**Files:**
- Create: `src/experiments/exp5_gpt2_mlp_table/__init__.py`, `src/experiments/exp5_gpt2_mlp_table/features.py`
- Test: `tests/test_exp5_gpt2_mlp_table.py`

**Interfaces:**
- Consumes: `utils.inference.CausalLMInference` (`final_hidden_states`, `model.get_output_embeddings()`, `device`); `experiments.exp1_L_M_table.table.select_header_tokens`, `centered_logits` (tests only).
- Produces:
  - `__init__.py`: `EXPERIMENT = "exp5_gpt2_mlp_table"`, `TRAIN_DIR = "train"`, `HEADER_TOLERANCE = 1e-3`.
  - `last_hidden(inference, ids: Tensor, batch_size: int = 1024) -> Tensor` — `ids` `(N, T)` integer; returns `(N, d)` float32 on `inference.device`, no grad, computed in chunks of `batch_size`.
  - `future_only_logits(inference, future_hidden: Tensor) -> Tensor` — `(N, d)` → `(N, V)`: the output embedding applied to the hidden state, minus its mean over the vocabulary.
  - `header_tokens(centered: Tensor, top_k: int) -> tuple[Tensor, Tensor]` — `(N, V)` → ids `(N, top_k)` int64 and values `(N, top_k)` float32, descending by value, ties broken by lower token id (stable sort; do not use `torch.topk`). Raises `ValueError` unless `1 <= top_k <= V`.
  - `pair_targets(pair_logits: Tensor, header_ids: Tensor, g: Tensor) -> Tensor` — raw `(B, V)` logits of `h + f`; returns `(B, top_k)`: centered over the vocabulary, gathered at `header_ids`, minus `g`.

- [ ] **Step 1: Write the fixture and the failing tests**

The fixture is shared by every later task. Tokenizer files are needed because `produce_blocks` loads a tokenizer from the model directory.

```python
VOCAB, HISTORY_TOKENS, FUTURE_TOKENS, TOP_K = 17, 3, 2, 3
HISTORIES = [[row, (row + 1) % VOCAB, (row + 4) % VOCAB] for row in range(6)]
FUTURES = [[(row + 2) % VOCAB, (row + 5) % VOCAB] for row in range(5)]


class Exp5Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        cls.model_path = cls.root / "model"
        GPT2LMHeadModel(GPT2Config(
            vocab_size=VOCAB, n_positions=8, n_embd=8, n_layer=1, n_head=2,
            bos_token_id=None, eos_token_id=None,
        )).save_pretrained(cls.model_path)
        tokenizer = Tokenizer(WordLevel(
            {"[UNK]": 0, **{str(i): i for i in range(1, VOCAB)}}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="[UNK]", model_max_length=8,
        ).save_pretrained(cls.model_path)
        cls.inference = CausalLMInference(cls.model_path, "cpu")
        # exp4 tables: train 4 x 3 futures, val 2 x 2 futures.
        samples_path = cls.root / "samples.parquet"
        windows = HISTORIES + FUTURES
        pq.write_table(pa.table({
            "sample_id": pa.array(range(len(windows)), type=pa.int32()),
            "group": ["history"] * len(HISTORIES) + ["future"] * len(FUTURES),
            "token_ids": pa.array(windows, type=pa.list_(pa.int32())),
        }), samples_path)
        cls.inference_dir = cls.root / "inference"
        build_exp4_tables(
            samples_path=samples_path, model_path=cls.model_path, device="cpu",
            train_histories=4, train_futures=3, val_histories=2, val_futures=2,
            top_k=TOP_K, batch_size=2, output_dir=cls.inference_dir, overwrite=False,
        )
        # Sample store: 6 blocks of 4 pairs.
        stream_path = cls.root / "stream.npy"
        np.save(stream_path, (np.arange(200, dtype=np.uint16) * 7) % VOCAB)
        stream_path.with_suffix(".json").write_text(json.dumps({
            "vocab_size": VOCAB, "config": "tiny", "split": "train",
            "tokenizer_path": str(cls.model_path),
        }))
        cls.store_dir = cls.root / "store"
        produce_blocks(
            store_dir=cls.store_dir, model_path=cls.model_path, stream_path=stream_path,
            device="cpu", history_tokens=HISTORY_TOKENS, future_tokens=FUTURE_TOKENS,
            samples_per_block=4, max_bytes=8 * slot_bytes(4, HISTORY_TOKENS, FUTURE_TOKENS, VOCAB),
            worker_id=0, seed=0, batch_size=2, max_blocks=6, log_every=100, overwrite=False,
        )
```

Tests in `class FeatureTests(Exp5Fixture)`:

```python
def test_last_hidden_matches_the_model_and_ignores_chunking(self):
    ids = torch.tensor(HISTORIES)
    expected = self.inference.final_hidden_states(ids)[:, -1]
    for batch_size in (1, 4, 100):
        hidden = last_hidden(self.inference, ids, batch_size)
        self.assertFalse(hidden.requires_grad)
        torch.testing.assert_close(hidden, expected, rtol=0, atol=1e-6)

def test_headers_match_exp1_selection(self):
    futures = torch.tensor(FUTURES)
    centered = future_only_logits(self.inference, last_hidden(self.inference, futures))
    ids, g = header_tokens(centered, TOP_K)
    expected_ids, expected_g = select_header_tokens(
        self.inference, [Sample(i, row) for i, row in enumerate(FUTURES)], TOP_K)
    self.assertEqual(ids.dtype, torch.int64)
    self.assertEqual(ids.tolist(), expected_ids.tolist())
    np.testing.assert_allclose(g.numpy(), expected_g, rtol=0, atol=1e-5)

def test_ties_go_to_the_lower_token_id(self):
    ids, values = header_tokens(torch.tensor([[1.0, 3.0, 3.0, 0.0, 3.0]]), 2)
    self.assertEqual(ids.tolist(), [[1, 2]])
    self.assertEqual(values.tolist(), [[3.0, 3.0]])

def test_top_k_must_fit_the_vocabulary(self):
    for top_k in (0, 6):
        with self.assertRaises(ValueError):
            header_tokens(torch.zeros(1, 5), top_k)

def test_pair_targets_are_the_table_entries_minus_g(self):
    h, f = torch.tensor(HISTORIES[:2]), torch.tensor(FUTURES[:2])
    ids, g = header_tokens(
        future_only_logits(self.inference, last_hidden(self.inference, f)), TOP_K)
    raw = torch.stack([
        self.inference.next_token_logits(h[i].tolist() + f[i].tolist()) for i in range(2)])
    target = pair_targets(raw + 5.0, ids, g)  # a per-pair shift must not matter
    for i in range(2):
        entries = centered_logits(raw[i])[ids[i]]
        torch.testing.assert_close(target[i], entries - g[i], rtol=0, atol=1e-5)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run python -m unittest discover -s tests -p "test_exp5*.py" -v`
Expected: import error for `experiments.exp5_gpt2_mlp_table`.

- [ ] **Step 3: Implement `__init__.py` and `features.py` with the signatures above**

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run python -m unittest discover -s tests -p "test_exp5*.py" -v`
Expected: 5 tests, OK.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/exp5_gpt2_mlp_table tests/test_exp5_gpt2_mlp_table.py
git commit -m "exp5: frozen GPT-2 features, header tokens and pair targets"
```

---

### Task 2: The network (`model.py`)

**Files:**
- Create: `src/experiments/exp5_gpt2_mlp_table/model.py`
- Test: `tests/test_exp5_gpt2_mlp_table.py`

**Interfaces:**
- Consumes: `utils.nestedlora.NestedLoRA`.
- Produces:
  - `class Exp5Network(nn.Module)`, `__init__(self, hidden_size: int, rank: int, top_k: int, mlp_width: int)`.
    - `left_features(history_hidden: Tensor) -> Tensor` — `(B, d)` → `(B, rank)`.
    - `right_features(future_hidden: Tensor) -> Tensor` — `(F, d)` → `(F * top_k, rank)`; row `f * top_k + j` is slot `j` of future `f`.
    - `forward(history_hidden, future_hidden) -> tuple[Tensor, Tensor]` — both of the above.
  - `build_model(hidden_size: int, rank: int, top_k: int, mlp_width: int) -> NestedLoRA` — `NestedLoRA(model=Exp5Network(...), neigs=rank)` (joint nesting, the default).
- Each side is exactly `LayerNorm(hidden_size, elementwise_affine=False) → Linear(hidden_size, mlp_width) → GELU → Linear(mlp_width, out)`, with `out = rank` on the left and `out = top_k * rank` on the right.

- [ ] **Step 1: Write the failing tests** (`class ModelTests(unittest.TestCase)`)

```python
def test_shapes_and_column_order(self):
    torch.manual_seed(0)
    network = Exp5Network(hidden_size=8, rank=4, top_k=3, mlp_width=16)
    left, right = network(torch.randn(5, 8), torch.randn(2, 8))
    self.assertEqual(left.shape, (5, 4))
    self.assertEqual(right.shape, (2 * 3, 4))
    # Row f * top_k + j is slot j of future f: each future's block depends only on that future.
    hidden = torch.randn(2, 8)
    both = network.right_features(hidden)
    torch.testing.assert_close(both[3:], network.right_features(hidden[1:]))
    # Slot j of one future is the j-th row of the final layer's (top_k, rank) output.
    raw = network.right[-1](network.right[:-1](hidden[:1])).reshape(3, 4)
    torch.testing.assert_close(both[:3], raw)

def test_layers_and_parameter_count(self):
    network = Exp5Network(hidden_size=8, rank=4, top_k=3, mlp_width=16)
    for side, out in ((network.left, 4), (network.right, 12)):
        self.assertEqual(
            [type(layer).__name__ for layer in side], ["LayerNorm", "Linear", "GELU", "Linear"])
        self.assertEqual((side[1].in_features, side[1].out_features), (8, 16))
        self.assertEqual((side[3].in_features, side[3].out_features), (16, out))
        self.assertFalse(side[0].elementwise_affine)

def test_build_model_wraps_the_network_with_joint_nesting(self):
    model = build_model(hidden_size=8, rank=4, top_k=3, mlp_width=16)
    self.assertIsInstance(model, NestedLoRA)
    self.assertFalse(model.sequential)
    self.assertEqual(model.neigs, 4)
```

`left` and `right` are therefore `nn.Sequential` attributes named exactly `left` and `right`.

- [ ] **Step 2: Run and confirm the tests fail** — same command; expected: import error for `model`.
- [ ] **Step 3: Implement `model.py`**
- [ ] **Step 4: Run and confirm they pass** — expected: 8 tests, OK.
- [ ] **Step 5: Commit**

```bash
git add src/experiments/exp5_gpt2_mlp_table/model.py tests/test_exp5_gpt2_mlp_table.py
git commit -m "exp5: MLP heads that output every header slot of a future"
```

---

### Task 3: The paired loss (`loss.py`)

**Files:**
- Create: `src/experiments/exp5_gpt2_mlp_table/loss.py`
- Test: `tests/test_exp5_gpt2_mlp_table.py`

**Interfaces:**
- Consumes: `NestedLoRA.compute_loss_kernel(f, g, K)`, `NestedLoRA.vector_mask`, `NestedLoRA.matrix_mask`.
- Produces: `paired_loss(model: NestedLoRA, left: Tensor, right: Tensor, target: Tensor) -> Tensor` — `left` `(B, L)`, `right` `(B * top_k, L)`, `target` `(B, top_k)`; scalar. Raises `ValueError` when `right.shape[0] != target.numel()` or `left.shape[0] != target.shape[0]`.

The loss is

```
−2 · mean_{i,j} [ target[i,j] · Σ_l vector_mask[l] · left[i,l] · right[i·top_k + j, l] ]
+ Σ ( matrix_mask ⊙ (leftᵀleft / B) ⊙ (rightᵀright / (B·top_k)) )
```

- [ ] **Step 1: Write the failing tests** (`class PairedLossTests(unittest.TestCase)`)

```python
def reference(self, model, left, right, target):
    batch, top_k = target.shape
    slots = right.reshape(batch, top_k, -1)
    operator = -2 * (
        target[:, :, None] * left[:, None, :] * slots * model.vector_mask
    ).sum(-1).mean()
    metric = (
        model.matrix_mask * (left.T @ left / batch) * (right.T @ right / (batch * top_k))
    ).sum()
    return operator + metric

def check(self, batch, top_k, rank):
    torch.manual_seed(0)
    model = build_model(hidden_size=8, rank=rank, top_k=top_k, mlp_width=16).double()
    left = torch.randn(batch, rank, dtype=torch.float64, requires_grad=True)
    right = torch.randn(batch * top_k, rank, dtype=torch.float64, requires_grad=True)
    target = torch.randn(batch, top_k, dtype=torch.float64)
    loss = paired_loss(model, left, right, target)
    expected = self.reference(model, left, right, target)
    torch.testing.assert_close(loss, expected)
    for got, want in zip(
        torch.autograd.grad(loss, (left, right)), torch.autograd.grad(expected, (left, right))
    ):
        torch.testing.assert_close(got, want)
        self.assertTrue(torch.isfinite(got).all())

def test_value_and_gradients_match_the_formula(self):
    self.check(batch=5, top_k=3, rank=4)

def test_single_pair_batch_is_finite(self):
    self.check(batch=1, top_k=3, rank=4)

def test_mismatched_shapes_are_rejected(self):
    model = build_model(hidden_size=8, rank=4, top_k=3, mlp_width=16)
    with self.assertRaises(ValueError):
        paired_loss(model, torch.zeros(2, 4), torch.zeros(5, 4), torch.zeros(2, 3))
    with self.assertRaises(ValueError):
        paired_loss(model, torch.zeros(3, 4), torch.zeros(6, 4), torch.zeros(2, 3))
```

- [ ] **Step 2: Run and confirm the tests fail** — expected: import error for `loss`.

- [ ] **Step 3: Implement `paired_loss`**

Use the existing loss rather than re-deriving gradients: build a `(B, B * top_k)` kernel that is zero except `kernel[i, i * top_k + j] = B * target[i, j]`, and return `model.compute_loss_kernel(left, right, kernel)[0]`. With that kernel `Tg` and `Tadjf` reduce to the paired operator term, and the metric term already uses the whole batch.

- [ ] **Step 4: Run and confirm they pass** — expected: 11 tests, OK.
- [ ] **Step 5: Commit**

```bash
git add src/experiments/exp5_gpt2_mlp_table/loss.py tests/test_exp5_gpt2_mlp_table.py
git commit -m "exp5: add the paired NestedLoRA loss"
```

---

### Task 4: Evaluation tables (`tables.py`)

**Files:**
- Create: `src/experiments/exp5_gpt2_mlp_table/tables.py`
- Test: `tests/test_exp5_gpt2_mlp_table.py`

**Interfaces:**
- Consumes: Task 1 (`last_hidden`, `future_only_logits`), Task 2 (`Exp5Network.left_features`, `.right_features`), `HEADER_TOLERANCE`; from exp4: `experiments.exp4_table_neural_svd.train.Table`, `table_columns`; `experiments.exp4_table_neural_svd.SVD_RANK` (256); `utils.linalg.matrix_svd(matrix, device=...)` (returns `.S`, `.U`, `.Vt`).
- Produces:

```python
@dataclass(frozen=True)
class EvalTable:
    table: Table             # matrix - g, with the exact SVD of that matrix
    history_hidden: Tensor   # (N, d) float32
    future_hidden: Tensor    # (F, d) float32
    residual_energy: float   # ||matrix - g||^2 / ||matrix||^2
    residual_top_mode: float # S[0]^2 / ||matrix - g||^2
```

  - `load_eval_table(table_dir: Path, inference: CausalLMInference, device: str | torch.device, batch_size: int = 1024) -> EvalTable` — reads `matrix.npy` and `tokens.npz` (`histories`, `futures`, `header_token_ids`) from an exp4 table directory. `g[f, j]` is the centered future-only logit of stored header token `(f, j)`. Raises `ValueError` naming the number of offending futures when, for any future, `g[f].min() < kth_largest(centered[f], top_k) − HEADER_TOLERANCE`. Subtracts `g.reshape(-1)` from every row, takes the float64 thin SVD of the result and keeps the leading `SVD_RANK` vectors, exactly as `load_table`'s column-mean branch does. All tensors end on `device`.
  - `eval_features(model: NestedLoRA, table: EvalTable, chunk: int = 2048) -> tuple[Tensor, Tensor]` — sets `model.eval()`, no grad; returns left `(N, L)` and right `(C, L)` in float64, `C = F * top_k`, computed in chunks of `chunk` hidden states.

- [ ] **Step 1: Write the failing tests** (`class EvalTableTests(Exp5Fixture)`)

```python
def stored(self, name):
    directory = self.inference_dir / name
    return np.load(directory / "matrix.npy"), dict(np.load(directory / "tokens.npz"))

def expected_g(self, tokens):
    centered = future_only_logits(
        self.inference, last_hidden(self.inference, torch.from_numpy(tokens["futures"])))
    return centered.gather(1, torch.from_numpy(tokens["header_token_ids"]).long())

def test_matrix_loses_g_and_the_svd_belongs_to_the_result(self):
    for name in ("train", "val"):
        with self.subTest(table=name):
            matrix, tokens = self.stored(name)
            loaded = load_eval_table(self.inference_dir / name, self.inference, "cpu")
            g = self.expected_g(tokens).reshape(-1).numpy()
            np.testing.assert_allclose(loaded.table.matrix.numpy(), matrix - g, atol=1e-5)
            table = loaded.table
            rebuilt = (table.left_vectors * table.singular_values[: table.left_vectors.shape[1]]
                       ) @ table.right_vectors.T
            np.testing.assert_allclose(rebuilt.numpy(), matrix - g, atol=1e-4)
            residual = np.square(matrix - g).sum()
            self.assertAlmostEqual(
                loaded.residual_energy, residual / np.square(matrix).sum(), places=5)
            self.assertAlmostEqual(
                loaded.residual_top_mode, table.singular_values[0].item() ** 2 / residual, places=5)
            self.assertEqual(table.histories.tolist(), tokens["histories"].tolist())
            self.assertEqual(table.columns.shape, (matrix.shape[1], FUTURE_TOKENS + 1))

def test_hidden_states_are_cached_for_every_history_and_future(self):
    _, tokens = self.stored("train")
    loaded = load_eval_table(self.inference_dir / "train", self.inference, "cpu", batch_size=2)
    for cached, ids in ((loaded.history_hidden, tokens["histories"]),
                        (loaded.future_hidden, tokens["futures"])):
        expected = self.inference.final_hidden_states(torch.from_numpy(ids))[:, -1]
        torch.testing.assert_close(cached, expected, rtol=0, atol=1e-6)

def copy_table(self, name):
    target = self.root / f"copy-{self.id().rsplit('.', 1)[-1]}"
    shutil.copytree(self.inference_dir / name, target)
    return target

def test_reordered_headers_load_and_g_follows_the_stored_order(self):
    directory = self.copy_table("train")
    matrix, tokens = np.load(directory / "matrix.npy"), dict(np.load(directory / "tokens.npz"))
    # Swap slots 0 and 1 of the first future, in the header ids and in the matrix columns.
    tokens["header_token_ids"][0, [0, 1]] = tokens["header_token_ids"][0, [1, 0]]
    matrix[:, [0, 1]] = matrix[:, [1, 0]]
    np.save(directory / "matrix.npy", matrix)
    np.savez(directory / "tokens.npz", **tokens)
    loaded = load_eval_table(directory, self.inference, "cpu")
    g = self.expected_g(tokens).reshape(-1).numpy()
    np.testing.assert_allclose(loaded.table.matrix.numpy(), matrix - g, atol=1e-5)

def test_a_header_outside_the_top_k_is_rejected(self):
    directory = self.copy_table("train")
    tokens = dict(np.load(directory / "tokens.npz"))
    centered = future_only_logits(
        self.inference, last_hidden(self.inference, torch.from_numpy(tokens["futures"])))
    tokens["header_token_ids"][0, -1] = centered[0].argmin().item()
    np.savez(directory / "tokens.npz", **tokens)
    with self.assertRaisesRegex(ValueError, "1 future"):
        load_eval_table(directory, self.inference, "cpu")

def test_features_cover_every_row_and_column_in_table_order(self):
    loaded = load_eval_table(self.inference_dir / "train", self.inference, "cpu")
    torch.manual_seed(0)
    model = build_model(hidden_size=8, rank=4, top_k=TOP_K, mlp_width=16)
    left, right = eval_features(model, loaded, chunk=2)
    self.assertEqual((left.dtype, right.dtype), (torch.float64, torch.float64))
    self.assertEqual(left.shape, (4, 4))
    self.assertEqual(right.shape, (3 * TOP_K, 4))
    torch.testing.assert_close(
        right.float(), model.model.right_features(loaded.future_hidden), atol=1e-6, rtol=0)
    self.assertFalse(model.training)
    # exp4's comparison accepts the result unchanged.
    comparison = compare_with_exact_svd(left, right, loaded.table)
    self.assertEqual(comparison["ranks"], [1, 2, 4])
```

- [ ] **Step 2: Run and confirm the tests fail** — expected: import error for `tables`.
- [ ] **Step 3: Implement `tables.py`**
- [ ] **Step 4: Run and confirm they pass** — expected: 16 tests, OK.
- [ ] **Step 5: Commit**

```bash
git add src/experiments/exp5_gpt2_mlp_table/tables.py tests/test_exp5_gpt2_mlp_table.py
git commit -m "exp5: evaluation tables with the future-only logit removed"
```

---

### Task 5: The training loop (`train.py`)

**Files:**
- Create: `src/experiments/exp5_gpt2_mlp_table/train.py`
- Test: `tests/test_exp5_gpt2_mlp_table.py`

**Interfaces:**
- Consumes: Tasks 1–4; `data_pipline.sample_store.SampleSampler` (`.config`, `.wait_until_ready()`, `.sample() -> (h, f, logits)`); `data_pipline.sample_store.producer.model_file_identity`; exp4's `compare_with_exact_svd`, `format_comparison`, `RANKS`, `TABLE_NAMES`; `utils.common.save_json`, `write_atomic`.
- Produces:
  - `sampler_seed(seed: int, step: int) -> int` — `int(np.random.SeedSequence([seed, step]).generate_state(1)[0])`.
  - `check_store(config: dict, model_path: Path, *, history_tokens: int, future_tokens: int, vocab_size: int) -> None` — `ValueError` naming the differing quantity when the store's `history_tokens`, `future_tokens` or `vocab_size` differ, or when `{name: entry["size"]}` of `config["model_files"]` differs from that of `model_file_identity(model_path)` (modification times are not compared).
  - `draw_pairs(sampler: SampleSampler, batch_size: int) -> tuple[Tensor, Tensor, Tensor]` — `h (B, H)` int64, `f (B, F)` int64, `logits (B, V)` float32, stacked from `batch_size` calls of `sampler.sample()`.
  - `train_exp5(*, inference_dir: Path, store_dir: Path, model_path: Path, output_dir: Path, device: str, neigs: int, mlp_width: int, batch_size: int, steps: int, eval_every: int, lr: float, seed: int, overwrite: bool, encode_batch: int = 1024, eval_chunk: int = 2048) -> dict` — returns `{"output_dir": str, "step": int, "evaluations": int}`.

**Behavior of `train_exp5`**, in order:

1. `ValueError` if `inference_dir / "metadata.json"` is missing (`"no finished tables in ...; run exp4 inference first"`), or if `seed < 0`.
2. Load GPT-2 once (`CausalLMInference(model_path, device)`). Load both tables with `load_eval_table`. `top_k` comes from the validation table's columns.
3. Create `SampleSampler(store_dir, seed=sampler_seed(seed, step))` and run `check_store` against the tables' window lengths and `inference.vocab_size`.
4. `config` (compared on resume): `inference_dir`, `store_dir`, `model_path` (all `str`), `neigs`, `mlp_width`, `batch_size`, `lr`, `seed`, `"optimizer": "Adam"`, `"first_mode": "1 x centered future-only logit of the header token"`.
5. Overwrite, checkpoint, resume and `FileExistsError` handling exactly as `train_exp4` (`checkpoint.pt` with a different `config` → `ValueError` mentioning `--overwrite`; `eval_history.jsonl` without a checkpoint → `FileExistsError`). `torch.manual_seed(seed)` before `build_model`. Adam over `model.parameters()`.
6. Write `metadata.json` before training: `config` plus `device`, `top_k`, `hidden_size`, `ranks`, `trainable_parameters`, `table_shapes`, `tables` (`{name: {"residual_energy", "residual_top_mode"}}`), `store` (the store's `store.json` content), and these fixed strings:
   - `"hidden_state"`: `"final layer after ln_f at the last token; history and future are encoded separately"`
   - `"value"`: `"logit(token | history + future) - mean(logits over full vocabulary) - centered logit(token | future)"`
   - `"sampling"`: `"independent (history, future) pairs from the sample store; every header token of each pair"`
   - `"isolation"`: `"store windows are drawn from the whole training stream and are not excluded from the evaluation tables; a pair overlaps a validation window with probability of about 0.1%"`
7. Print `Waiting for the first sample block in <store_dir>` and call `sampler.wait_until_ready()`.
8. Each step: `draw_pairs` → `last_hidden` of `h` and of `f` → `future_only_logits` + `header_tokens` → `pair_targets` → `model(history_hidden, future_hidden)` → `paired_loss` → `RuntimeError(f"non-finite loss at step {step + 1}")` if not finite → backward, optimizer step.
9. When `step % eval_every == 0 or step == steps`: append a record `{"step", "train_loss", "kernel_energy", "seconds_per_step", "train", "val"}` (`train_loss` and `kernel_energy = mean(target²)` averaged since the last evaluation; `train`/`val` from `compare_with_exact_svd(*eval_features(model, table, eval_chunk), table.table)`), print it with `format_comparison`, then write `checkpoint.pt` (`model_state_dict`, `optimizer_state_dict`, `step`, `history`, `config`) and `eval_history.jsonl` atomically.
10. On resume the sampler is created with the resumed `step`, so a resumed run does not replay the first pairs.

- [ ] **Step 1: Write the failing tests** (`class TrainingTests(Exp5Fixture)`)

```python
def train(self, output_dir, **changes):
    options = dict(
        inference_dir=self.inference_dir, store_dir=self.store_dir, model_path=self.model_path,
        output_dir=output_dir, device="cpu", neigs=2, mlp_width=16, batch_size=8,
        steps=4, eval_every=2, lr=1e-2, seed=0, overwrite=False,
    )
    options.update(changes)
    return train_exp5(**options)

def test_writes_outputs_and_compares_both_tables(self):
    output = self.root / "run-outputs"
    summary = self.train(output)
    self.assertEqual(summary, {"output_dir": str(output), "step": 4, "evaluations": 2})
    records = [json.loads(line) for line in (output / "eval_history.jsonl").read_text().splitlines()]
    self.assertEqual([record["step"] for record in records], [2, 4])
    for record in records:
        self.assertEqual(
            set(record), {"step", "train_loss", "kernel_energy", "seconds_per_step", "train", "val"})
        self.assertEqual(record["train"]["ranks"], [1, 2])
        self.assertTrue(np.isfinite(record["train_loss"]))
    metadata = json.loads((output / "metadata.json").read_text())
    self.assertEqual(metadata["top_k"], TOP_K)
    self.assertEqual(metadata["mlp_width"], 16)
    self.assertEqual(metadata["table_shapes"], {"train": [4, 9], "val": [2, 6]})
    self.assertEqual(set(metadata["tables"]["val"]), {"residual_energy", "residual_top_mode"})
    self.assertIn("0.1%", metadata["isolation"])
    self.assertEqual(metadata["store"]["history_tokens"], HISTORY_TOKENS)

def test_only_the_mlps_are_trained_and_saved(self):
    output = self.root / "run-frozen"
    self.train(output)
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    self.assertTrue(all(key.startswith(("model.left.", "model.right."))
                        for key in checkpoint["model_state_dict"]))
    metadata = json.loads((output / "metadata.json").read_text())
    network = Exp5Network(hidden_size=8, rank=2, top_k=TOP_K, mlp_width=16)
    self.assertEqual(
        metadata["trainable_parameters"], sum(p.numel() for p in network.parameters()))

def test_training_raises_the_captured_energy(self):
    output = self.root / "run-learns"
    self.train(output, steps=300, eval_every=100, batch_size=16)
    records = [json.loads(line) for line in (output / "eval_history.jsonl").read_text().splitlines()]
    first, last = (record["train"]["captured_energy"][0] for record in (records[0], records[-1]))
    self.assertGreater(last, first)
    self.assertGreater(last, 0.0)
    self.assertLess(records[-1]["train_loss"], records[0]["train_loss"])

def test_resume_and_step_edge_cases(self):
    output = self.root / "run-resume"
    self.train(output, steps=2)
    self.assertEqual(self.train(output, steps=4)["evaluations"], 2)   # continues from step 2
    self.assertEqual(self.train(output, steps=4), {"output_dir": str(output), "step": 4, "evaluations": 2})
    self.assertEqual(self.train(output, steps=3)["step"], 4)          # not above the checkpoint
    with self.assertRaisesRegex(ValueError, "--overwrite"):
        self.train(output, steps=6, lr=1e-3)
    self.assertEqual(self.train(output, steps=2, lr=1e-3, overwrite=True)["step"], 2)
    late = self.root / "run-late-eval"
    self.assertEqual(self.train(late, steps=3, eval_every=10)["evaluations"], 1)

def test_rejects_missing_tables_and_a_negative_seed(self):
    with self.assertRaisesRegex(ValueError, "exp4 inference"):
        self.train(self.root / "run-none", inference_dir=self.root / "absent")
    with self.assertRaises(ValueError):
        self.train(self.root / "run-seed", seed=-1)

def tampered_store(self, **changes):
    store = self.root / f"store-{self.id().rsplit('.', 1)[-1]}"
    shutil.copytree(self.store_dir, store)
    config = json.loads((store / "store.json").read_text())
    config.update(changes)
    (store / "store.json").write_text(json.dumps(config))
    return store, config

def test_rejects_a_store_with_other_window_lengths_or_vocabulary(self):
    for key in ("history_tokens", "future_tokens", "vocab_size"):
        with self.subTest(key=key):
            _, config = self.tampered_store()
            config[key] += 1
            with self.assertRaisesRegex(ValueError, key):
                check_store(config, self.model_path, history_tokens=HISTORY_TOKENS,
                            future_tokens=FUTURE_TOKENS, vocab_size=VOCAB)

def test_rejects_a_store_from_other_model_files(self):
    _, config = self.tampered_store()
    name = next(iter(config["model_files"]))
    config["model_files"][name]["size"] += 1
    with self.assertRaisesRegex(ValueError, "model"):
        check_store(config, self.model_path, history_tokens=HISTORY_TOKENS,
                    future_tokens=FUTURE_TOKENS, vocab_size=VOCAB)
    # A different modification time alone is accepted.
    _, config = self.tampered_store()
    config["model_files"][name]["mtime_ns"] += 1
    check_store(config, self.model_path, history_tokens=HISTORY_TOKENS,
                future_tokens=FUTURE_TOKENS, vocab_size=VOCAB)

def test_waits_for_the_first_block(self):
    with patch.object(SampleSampler, "wait_until_ready", autospec=True) as wait:
        self.train(self.root / "run-wait")
    wait.assert_called_once()

def test_draw_pairs_stacks_store_samples(self):
    h, f, logits = draw_pairs(SampleSampler(self.store_dir, seed=3), 5)
    self.assertEqual((h.shape, f.shape, logits.shape),
                     ((5, HISTORY_TOKENS), (5, FUTURE_TOKENS), (5, VOCAB)))
    expected = self.inference.next_token_logits(h[0].tolist() + f[0].tolist())
    torch.testing.assert_close(logits[0], expected, rtol=0, atol=1e-6)
    self.assertNotEqual(sampler_seed(0, 0), sampler_seed(0, 2))
```

If `test_training_raises_the_captured_energy` does not pass with these values because the tiny random model leaves too little structure, raise `steps` or `lr` in the test; do not weaken the three assertions.

- [ ] **Step 2: Run and confirm the tests fail** — expected: import error for `train`.
- [ ] **Step 3: Implement `train.py`**
- [ ] **Step 4: Run and confirm they pass** — expected: 25 tests, OK.
- [ ] **Step 5: Run the full suite**

Run: `uv run python -m unittest discover -s tests`
Expected: OK, with no change to any exp4 or sample-store test.

- [ ] **Step 6: Commit**

```bash
git add src/experiments/exp5_gpt2_mlp_table/train.py tests/test_exp5_gpt2_mlp_table.py
git commit -m "exp5: train on sample-store pairs and compare with exact SVDs"
```

---

### Task 6: CLI, launcher and documentation

**Files:**
- Create: `src/experiments/exp5_gpt2_mlp_table/cli.py`, `src/experiments/exp5_gpt2_mlp_table/__main__.py`, `bluehive_run/exp5-train.sbatch`
- Modify: `pyproject.toml` (`[project.scripts]`), `tests/test_cli_startup.py` (module list), `README.md`
- Test: `tests/test_exp5_gpt2_mlp_table.py`

**Interfaces:**
- Consumes: `train_exp5`; `utils.experiment_paths.add_run_arguments`, `run_dir`, `INFERENCE_DIR`; `data_pipline.cli.positive_int`; `EXPERIMENT`, `TRAIN_DIR`; exp4's `cli.EXPERIMENT` and `cli.positive_float`.
- Produces: `main(argv: list[str] | None = None) -> None`, script `exp5 = "experiments.exp5_gpt2_mlp_table.cli:main"`.

Command: `exp5 train` with `--neigs`, `--mlp-width`, `--batch-size`, `--steps`, `--eval-every`, `--lr`, `--seed`, `--store-dir`, `--dataset`, `--results-root`, `--model-path`, `--device` (default `cuda`), `--overwrite`; defaults from Global Constraints. Paths:

- tables: `run_dir(results_root, <exp4 EXPERIMENT>, dataset, model_path.name) / INFERENCE_DIR`
- output: `run_dir(results_root, EXPERIMENT, dataset, model_path.name) / TRAIN_DIR`

`ValueError` and `FileExistsError` from `train_exp5` go to `parser.error`. On success print `output_dir`, `step` and `evaluations`, one per line, as exp4 does. `torch` is imported only inside the command branch.

- [ ] **Step 1: Write the failing tests**

In `tests/test_cli_startup.py`, add `"experiments.exp5_gpt2_mlp_table.cli"` to the module tuple. In the exp5 test file (`class CLITests(Exp5Fixture)`):

```python
def run_cli(self, *extra):
    main([
        "train", "--results-root", str(self.results_root), "--dataset", "tiny",
        "--model-path", str(self.model_path), "--store-dir", str(self.store_dir),
        "--device", "cpu", "--neigs", "2", "--mlp-width", "16", "--batch-size", "8",
        "--steps", "2", "--eval-every", "2", *extra,
    ])

def setUp(self):
    # Place the fixture's tables where the CLI looks for exp4's output.
    self.results_root = self.root / f"results-{self.id().rsplit('.', 1)[-1]}"
    target = self.results_root / "exp4_table_neural_svd" / "tiny__model" / "inference"
    shutil.copytree(self.inference_dir, target)

def test_train_writes_under_the_exp5_run_directory(self):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        self.run_cli()
    run = self.results_root / "exp5_gpt2_mlp_table" / "tiny__model" / "train"
    self.assertTrue((run / "eval_history.jsonl").is_file())
    self.assertIn(f"output_dir: {run}", output.getvalue())
    self.assertIn("step: 2", output.getvalue())

def test_errors_are_reported_as_usage_errors(self):
    for extra in (["--store-dir", str(self.root / "no-store")], ["--dataset", "missing"]):
        with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.run_cli(*extra)
            self.assertEqual(raised.exception.code, 2)

def test_defaults_match_the_design(self):
    with patch("experiments.exp5_gpt2_mlp_table.train.train_exp5") as train:
        train.return_value = {"output_dir": "x", "step": 0, "evaluations": 0}
        with contextlib.redirect_stdout(io.StringIO()):
            main(["train"])
    options = train.call_args.kwargs
    self.assertEqual(
        {key: options[key] for key in
         ("neigs", "mlp_width", "batch_size", "steps", "eval_every", "lr", "seed", "device")},
        {"neigs": 128, "mlp_width": 4096, "batch_size": 256, "steps": 100000,
         "eval_every": 1000, "lr": 3e-4, "seed": 42, "device": "cuda"})
    self.assertEqual(
        options["store_dir"], Path("data/sample_stores/wikitext-103-train__gpt2-small__h64-f32"))
    self.assertEqual(options["inference_dir"], Path(
        "data/experiment_results/exp4_table_neural_svd/wikitext-large__gpt2-small/inference"))
    self.assertEqual(options["output_dir"], Path(
        "data/experiment_results/exp5_gpt2_mlp_table/wikitext-large__gpt2-small/train"))
```

- [ ] **Step 2: Run and confirm the tests fail** — expected: import error for `cli`; `test_cli_startup` fails for the new module.
- [ ] **Step 3: Implement `cli.py` and `__main__.py`; add the script to `pyproject.toml`; run `uv sync`**
- [ ] **Step 4: Run and confirm they pass**

Run: `uv run python -m unittest discover -s tests -p "test_exp5*.py" -v && uv run python -m unittest tests.test_cli_startup -v && uv run exp5 train -h`
Expected: 28 exp5 tests OK, startup tests OK, help text listing the options above.

- [ ] **Step 5: Add `bluehive_run/exp5-train.sbatch`**

Copy `bluehive_run/exp4-train.sbatch`; change the job name to `exp5-train` and the last line to `exec .venv/bin/python -u -m experiments.exp5_gpt2_mlp_table train "$@"`. Keep partition, QOS, GPU, CPUs, memory and time. Add a comment that the sample store must already have published blocks or a producer must be running (`bluehive_run/sample-store-produce.sbatch`).

- [ ] **Step 6: Document exp5 in `README.md`**

Add a row to the experiment table at the top, a link in Navigation, and a section "Experiment 5: GPT-2 hidden states and MLPs on the L_M table" after the exp4 section. The section must state, in the README's existing style:

- what the two networks take and output, and that GPT-2 is frozen;
- the fixed first mode and the trained residual, with the three-row table of measured energies from the spec;
- the prerequisites: `exp4 inference` and a sample store with published blocks;
- the commands (`uv run exp5 train`, the sbatch, a small CPU smoke run using `--steps`, `--neigs`, `--mlp-width`, `--batch-size`, `--device cpu`);
- the outputs and their fields;
- that exp5's numbers are measured on the tables minus `g` and are not comparable with exp4's `train/` or `train-column-mean/` numbers except through each run's own `exact_energy`;
- that store pairs are not excluded from the evaluation tables (about 0.1% chance of partial overlap), so both tables are held out to the same degree;
- that a resumed run does not reproduce the same pairs.

- [ ] **Step 7: Run the full suite and commit**

Run: `uv run python -m unittest discover -s tests`
Expected: OK.

```bash
git add src/experiments/exp5_gpt2_mlp_table pyproject.toml uv.lock tests bluehive_run/exp5-train.sbatch README.md
git commit -m "add exp5: GPT-2 hidden states and MLPs on the L_M table"
```
