# exp7: an independent sub-network per singular function, trained and frozen one at a time — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add experiment `exp7`, which gives every singular function its own sub-network, trains the modes in stages on what the frozen earlier modes leave, aligns the learned functions at the end of each stage and compares them with the exact SVDs of the exp4 tables.

**Architecture:** A new package `src/experiments/exp7_independent_subnets/` with four small modules: the sub-networks of one mode (`model.py`), the stage's residual, loss and stopping rule (`stage.py`), the alignment (`align.py`) and the staged training loop (`train.py`), plus `cli.py`. Pairs, targets, `g`, the evaluation tables and the comparison come unchanged from exp5 and exp4.

**Tech Stack:** Python 3.14, PyTorch, transformers (local GPT-2), NumPy, `unittest`, uv.

**Spec:** `docs/superpowers/specs/2026-11-16-exp7-independent-subnets-design.md` (Chinese). Read it before starting; this plan argues from it.

## Global Constraints

- Run everything from the repository root. exp7 tests: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -v` (add `-k <test name>` for one test). Full suite: `uv run python -m unittest discover -s tests`.
- Do not change the behavior, outputs or checkpoint format of exp4, exp5, exp6 or `data_pipline.sample_store`; import from them only. Their tests must pass unchanged.
- The training target is exp5's: `logit − vocabulary mean − g`, on each future's header tokens. No other centering.
- No trainable parameter is shared between two modes, or between the two sides of a mode. The only shared tensor is the frozen teacher's `token_vectors` constant, built once.
- A frozen mode never changes and never receives a gradient: `requires_grad_(False)`, `eval()`, forward under `torch.no_grad()`.
- The left sub-network reads the history only; the right sub-network reads the future only. Nothing trainable reads `history + future`.
- Table names: exp4's `train` table is the **selection table** (record key `selection`), its `val` table is the **report table** (record key `report`). Only the selection table decides when a stage ends and which weights are frozen. exp4's files and directories keep their names.
- Every call of `compare_with_exact_svd` passes `ranks=tuple(range(1, l + 1))` for `l` modes; the selection metric is the last entry of `captured_energy`.
- Column order everywhere is future-major, then header slot `j`: column `future * top_k + j`. `top_k` comes from `prepare_run`; never hard-code 50.
- Defaults: `--subnet gpt2`, `--modes 8`, `--stage-steps 30000`, `--min-stage-steps 5000`, `--patience 5`, `--align-pairs 16384`, `--scratch-layers 6`, `--scratch-width 384`, `--scratch-heads 6`, `--batch-size 128`, `--eval-every 1000`, `--lr 3e-4`, `--encoder-lr 2e-5`, `--seed 42`, Adam, `--dataset wikitext-large`, `--model-path models/gpt2-small`, `--store-dir data/sample_stores/wikitext-103-train__gpt2-small__h64-f32`. Warmup of a `scratch` stage: 1000 steps, linear.
- Output directory: `<results-root>/exp7_independent_subnets/<dataset>__<model>/train-<subnet>/`.
- CLI modules must not import `torch` or `transformers` at import time (`tests/test_cli_startup.py`).
- Code style follows exp6: module docstrings, type hints, `write_atomic` for every output file, `ValueError`/`FileExistsError` for user errors.
- Work on a branch named `exp7-independent-subnets`, not on `main`. Commit messages follow the repository style, e.g. `exp7: add the stage loss`. Do not add `reports/` or `research_notes/`.

## Review Focus

Conditions the spec implies that are most likely to bite, each pinned by a test in the task that owns the code:

1. **Three or more modes.** exp4's default ranks are 1, 2, 4, 8, so with 3 modes a reader of `captured_energy[-1]` gets the rank-2 energy, which does not depend on the mode in training; every later stage would then freeze its first evaluation. Expected: the comparison lists ranks 1..l and the metric moves with the current mode (Task 4, `test_third_stage_selects_on_rank_three`).
2. **`--eval-every` does not divide `--stage-steps`, or exceeds it.** Expected: the last step of a stage is still evaluated, so a stage never freezes without a fresh metric (Task 4, `test_last_step_of_a_stage_is_evaluated`).
3. **A job is preempted in the middle of stage 2 and resubmitted.** Expected: mode 1's file is untouched, the stage continues at its last checkpoint with its best weights and stale count, and no step is recorded twice (Task 5, `test_resume_in_the_middle_of_a_stage`).
4. **Two learned modes are nearly collinear, or one is nearly zero** (a later mode that only repairs an earlier one). The Gram matrices are then singular. Expected: finite transforms and the least-squares fit of the span that is left, no NaN (Task 3, `test_collinear_and_zero_modes_give_finite_transforms`).
5. **A finished directory is run again with fewer `--modes` than it holds, or with the same number.** Expected: fewer is a `ValueError` that names both numbers; the same number returns at once and rewrites nothing (Task 5, `test_modes_below_and_equal_to_the_finished_count`).

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/experiments/exp7_independent_subnets/__init__.py` | Names: `EXPERIMENT`, `SUBNETS`, `TABLE_KEYS`, `WARMUP_STEPS` |
| `src/experiments/exp7_independent_subnets/__main__.py` | `python -m` entry, as exp6's |
| `src/experiments/exp7_independent_subnets/model.py` | `token_vectors`, `scratch_encoder`, `Mode`, `build_mode` |
| `src/experiments/exp7_independent_subnets/stage.py` | `residual_target`, `stage_loss`, `Selection` |
| `src/experiments/exp7_independent_subnets/align.py` | `Moments`, `alignment`, `max_correlation` |
| `src/experiments/exp7_independent_subnets/train.py` | `train_step`, `mode_table_features`, `selection_energy`, `train_exp7` |
| `src/experiments/exp7_independent_subnets/cli.py` | `exp7 train` |
| `tests/test_exp7_independent_subnets.py` | All exp7 tests; reuses `Exp5Fixture` from `tests/test_exp5_gpt2_mlp_table.py` |
| `bluehive_run/exp7-train.sbatch`, `pyproject.toml`, `README.md`, `tests/test_cli_startup.py` | Entry points and documentation |

`__init__.py` holds exactly:

```python
EXPERIMENT = "exp7_independent_subnets"
SUBNETS = ("gpt2", "scratch")
# exp4's table directory -> the name exp7 records it under.
TABLE_KEYS = {"train": "selection", "val": "report"}
WARMUP_STEPS = 1000
```

---

### Task 1: The sub-networks of one mode

**Files:**
- Create: `src/experiments/exp7_independent_subnets/__init__.py`, `__main__.py`, `model.py`
- Test: `tests/test_exp7_independent_subnets.py`

**Interfaces:**
- Consumes: `experiments.exp5_gpt2_mlp_table.model.trainable_encoder(inference) -> nn.Module`, `linear_head(hidden_size, rank) -> nn.Sequential`; `utils.inference.CausalLMInference` (`.model`, `.vocab_size`, `.max_length`).
- Produces:
  - `token_vectors(unembedding: Tensor (V, E)) -> Tensor (V, E)`: float32, detached, rows minus the mean row, divided by the root mean square of all entries.
  - `scratch_encoder(vocab_size: int, positions: int, layers: int, width: int, heads: int) -> nn.Module`: a randomly initialised `transformers.GPT2Model` with every dropout at 0.
  - `class Mode(nn.Module)` with attributes `left_encoder`, `right_encoder`, `left_head`, `right_head` and methods
    - `left_features(history_ids: Tensor (B, H)) -> Tensor (B,)`
    - `right_vectors(future_ids: Tensor (F, T)) -> Tensor (F, E)`
    - `right_features(future_ids: Tensor (F, T), header_ids: Tensor (F, K), vectors: Tensor (V, E)) -> Tensor (F, K)`, equal to `<right_vectors(f), vectors[z]> / sqrt(E)`
  - `build_mode(inference, subnet: str, *, scratch_layers: int, scratch_width: int, scratch_heads: int) -> Mode`. `gpt2`: both encoders are `trainable_encoder(inference)`. `scratch`: both are `scratch_encoder(inference.vocab_size, inference.max_length, ...)`. The heads are `linear_head(encoder width, 1)` on the left and `linear_head(encoder width, E)` on the right, `E` being the teacher's unembedding width. An unknown `subnet` raises `ValueError`.

- [ ] **Step 1: Create the branch and commit the spec and this plan**

```bash
git checkout -b exp7-independent-subnets
git add docs/superpowers/specs/2026-11-16-exp7-independent-subnets-design.md docs/superpowers/plans/2026-11-16-exp7-independent-subnets.md
git commit -m "exp7: design and implementation plan"
```

- [ ] **Step 2: Write the failing tests**

In `tests/test_exp7_independent_subnets.py`, class `ModeTests(Exp5Fixture)` (import `FUTURES, HISTORIES, TOP_K, VOCAB, Exp5Fixture` from `test_exp5_gpt2_mlp_table`; the fixture's GPT-2 has width 8, 1 layer, 8 positions). `HEADER = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 16]])`; `SCRATCH = dict(scratch_layers=1, scratch_width=4, scratch_heads=2)`.

```python
def test_token_vectors_are_the_centered_unembedding_at_unit_scale(self):
    weight = self.inference.model.get_output_embeddings().weight
    vectors = token_vectors(weight)
    centered = weight.detach() - weight.detach().mean(dim=0)
    self.assertFalse(vectors.requires_grad)
    self.assertAlmostEqual(vectors.square().mean().item(), 1.0, places=5)
    torch.testing.assert_close(vectors * centered.square().mean().sqrt(), centered, atol=1e-6, rtol=0)

def test_shapes_and_the_bilinear_right_side(self):
    for subnet in ("gpt2", "scratch"):
        mode = build_mode(self.inference, subnet, **SCRATCH)
        vectors = token_vectors(self.inference.model.get_output_embeddings().weight)
        futures = torch.tensor(FUTURES)
        with torch.no_grad():
            left = mode.left_features(torch.tensor(HISTORIES))
            w = mode.right_vectors(futures)
            right = mode.right_features(futures, self.HEADER, vectors)
        self.assertEqual((left.shape, w.shape, right.shape), ((6,), (5, 8), (5, TOP_K)))
        expected = torch.einsum("fe,fke->fk", w, vectors[self.HEADER]) / 8 ** 0.5
        torch.testing.assert_close(right, expected, atol=1e-6, rtol=0)

def test_gpt2_mode_starts_at_the_teacher_and_scratch_does_not(self):
    # gpt2: before training each encoder gives the teacher's hidden state.
    # scratch: width 4, its own embeddings; no parameter of the teacher is reused.

def test_the_four_parts_of_a_mode_and_two_modes_share_no_parameter(self):
    # ids of all parameters of mode_a.left_encoder, .right_encoder, .left_head, .right_head,
    # of mode_b, and of self.inference.model are pairwise disjoint, for both subnets.

def test_each_side_reads_its_own_input_only(self):
    # left_features has zero gradient w.r.t. the right encoder and head and vice versa;
    # right_features depends on a header token only through vectors[z]:
    # with vectors[2] = 2 * vectors[0] - 3 * vectors[1], column z=2 equals 2*col0 - 3*col1.

def test_unknown_subnet_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "subnet"):
        build_mode(self.inference, "mlp", **SCRATCH)
```

- [ ] **Step 3: Run them and see them fail**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -v`
Expected: `ModuleNotFoundError: No module named 'experiments.exp7_independent_subnets'`

- [ ] **Step 4: Implement `__init__.py`, `__main__.py` (copy exp6's) and `model.py` to the interfaces above**

The last hidden state is `encoder(input_ids=ids, use_cache=False).last_hidden_state[:, -1].float()`, as in exp6. Heads keep exp6's `LayerNorm(no affine) -> Linear`.

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -v`
Expected: all `ModeTests` pass.

- [ ] **Step 6: Commit**

```bash
git add src/experiments/exp7_independent_subnets tests/test_exp7_independent_subnets.py
git commit -m "exp7: one sub-network per singular function"
```

---

### Task 2: The stage's residual, loss and stopping rule

**Files:**
- Create: `src/experiments/exp7_independent_subnets/stage.py`
- Test: `tests/test_exp7_independent_subnets.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `residual_target(target: Tensor (B, K), frozen_left: Tensor (B, k), frozen_right: Tensor (B, K, k)) -> Tensor (B, K)`: `target − Σ_i frozen_left[:, i, None] * frozen_right[:, :, i]`; with `k = 0` it is `target`.
  - `stage_loss(left: Tensor (B,), right: Tensor (B, K), residual: Tensor (B, K)) -> Tensor` (scalar): `−2 · mean_b[left_b · mean_j(residual_bj · right_bj)] + mean_b(left_b²) · mean_{b,j}(right_bj²)`, written with plain autograd.
  - `@dataclass class Selection` with fields `max_steps: int`, `min_steps: int`, `patience: int`, `best: float | None = None`, `best_step: int = 0`, `stale: int = 0` and methods
    - `record(metric: float, stage_step: int) -> bool`: True when `metric` is a new best (the first metric is one); otherwise adds 1 to `stale`. A new best resets `stale` to 0. With `patience == 0` it records nothing and returns False.
    - `reason(stage_step: int) -> str | None`: `"steps"` when `stage_step >= max_steps`; else `"patience"` when `patience > 0`, `stage_step >= min_steps` and `stale >= patience`; else `None`.
    - `keeps_best -> bool` (property): `patience > 0`.

- [ ] **Step 1: Write the failing tests**

Class `StageTests(unittest.TestCase)`:

```python
def test_residual_removes_the_frozen_modes(self):
    # k = 0 returns target itself; k = 2 equals the explicit sum; shapes (B, K).

def test_stage_loss_is_the_rank_one_paired_loss(self):
    # Value and both gradients equal exp5's paired_loss with
    # NestedLoRA(torch.nn.Identity(), neigs=1), left[:, None], right.reshape(-1, 1), B = 4, K = 3.

def test_stage_gradient_is_the_sequentially_nested_one(self):
    # One pair is a full 1 x K grid. For L = 3 modes with left (1, L), right (K, L), target (1, K):
    # the gradient of paired_loss(NestedLoRA(Identity(), neigs=3, sequential=True), ...) w.r.t.
    # column l of left and of right equals the gradient of
    # stage_loss(left[:, l], right[:, l][None], residual_target(target, left[:, :l], right[None, :, :l]))
    # for l = 0, 1, 2 (atol 1e-6).

def test_single_pair_and_single_header_are_finite(self):
    # B = 1, K = 1: finite loss and gradients.

def test_selection_keeps_the_best_and_counts_stale_evaluations(self):
    s = Selection(max_steps=100, min_steps=30, patience=2)
    self.assertEqual([s.record(m, t) for m, t in ((0.5, 10), (0.9, 20), (0.1, 30), (0.1, 40))],
                     [True, True, False, False])
    self.assertEqual((s.best, s.best_step, s.stale), (0.9, 20, 2))
    self.assertEqual(s.reason(40), "patience")

def test_selection_waits_for_min_steps_and_stops_at_max_steps(self):
    # stale >= patience before min_steps gives None; stage_step == max_steps gives "steps"
    # even with a fresh best.

def test_patience_zero_is_a_fixed_number_of_steps(self):
    s = Selection(max_steps=4, min_steps=0, patience=0)
    self.assertFalse(s.keeps_best)
    self.assertFalse(s.record(0.9, 2))
    self.assertEqual((s.reason(2), s.reason(4)), (None, "steps"))
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -k StageTests -v`
Expected: `ImportError` for `experiments.exp7_independent_subnets.stage`.

- [ ] **Step 3: Implement `stage.py` to the interfaces above**

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -k StageTests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/exp7_independent_subnets/stage.py tests/test_exp7_independent_subnets.py
git commit -m "exp7: add the stage loss and the stopping rule"
```

---

### Task 3: The alignment

**Files:**
- Create: `src/experiments/exp7_independent_subnets/align.py`
- Test: `tests/test_exp7_independent_subnets.py`

**Interfaces:**
- Consumes: nothing.
- Produces (all float64):
  - `class Moments` with `__init__(modes: int)`, `add(left: Tensor (P, L), right: Tensor (P, K, L), target: Tensor (P, K)) -> None` and properties `lam_f (L, L) = E[f fᵀ]` over pairs, `lam_g (L, L) = E[g gᵀ]` over all pair-and-header columns, `cross (L, L)` with `cross[i, j] = E_pairs[ f_i · mean_k(target_k · g_kj) ]`. `add` may be called many times; the properties are the means over everything added.
  - `alignment(lam_f, lam_g, cross, floor: float = 1e-10) -> tuple[Tensor, Tensor, Tensor]` returning `(T_f, T_g, S)`:

    ```
    M = lam_f^{-1/2} · cross · lam_g^{-1/2}        (symmetric inverse square roots)
    M = P · diag(S) · Qᵀ                           (SVD, S descending)
    T_f = lam_f^{-1/2} · P · diag(S)^{1/2}
    T_g = lam_g^{-1/2} · Q · diag(S)^{1/2}
    ```

    In an inverse square root, an eigenvalue below `floor` times the largest one contributes 0 (a pseudo-inverse), so a dropped direction ends with `S = 0`.
  - `max_correlation(gram: Tensor (L, L)) -> float`: the largest absolute off-diagonal entry of `gram / sqrt(diag ⊗ diag)`; `0.0` for `L = 1`.

- [ ] **Step 1: Write the failing tests**

Class `AlignmentTests(unittest.TestCase)`. Shared setup: `torch.manual_seed(0)`; `A = U diag(s) Vᵀ` with `N = 12`, `C = 20`, `s = (5, 3, 2, 1, 0.5)`, float64. For "the product measure" use `lam_f = FᵀF / N`, `lam_g = GᵀG / C`, `cross = Fᵀ A G / (N · C)`.

```python
def test_alignment_recovers_the_truncated_svd_from_a_mixed_basis(self):
    # F = U[:, :3] @ M1, G = V[:, :3] @ M2 with random invertible, non-orthogonal M1, M2.
    T_f, T_g, S = alignment(lam_f, lam_g, cross)
    torch.testing.assert_close((F @ T_f) @ (G @ T_g).T, U[:, :3] * s[:3] @ V[:, :3].T, atol=1e-8, rtol=0)
    torch.testing.assert_close(S, s[:3] / (N * C) ** 0.5, atol=1e-10, rtol=0)
    torch.testing.assert_close((F @ T_f).T @ (F @ T_f) / N, torch.diag(S), atol=1e-10, rtol=0)
    torch.testing.assert_close((G @ T_g).T @ (G @ T_g) / C, torch.diag(S), atol=1e-10, rtol=0)

def test_alignment_is_the_least_squares_fit_in_the_span(self):
    # F, G random (not singular vectors): (F T_f)(G T_g)ᵀ equals P_F A P_G,
    # P the orthogonal projectors onto the column spans.

def test_one_mode_is_rescaled(self):
    # L = 1: T_f * T_g == cross / (lam_f * lam_g).

def test_collinear_and_zero_modes_give_finite_transforms(self):
    # F[:, 1] = 2 * F[:, 0] and G[:, 2] = 0: every output is finite, S[-1] == 0 (atol 1e-12),
    # and (F T_f)(G T_g)ᵀ still equals P_F A P_G.

def test_moments_match_their_definitions_in_any_chunking(self):
    # P = 6 pairs, K = 3, L = 2: adding all at once, or in chunks of 4 and 2, gives the same
    # lam_f, lam_g, cross as the explicit means.

def test_max_correlation(self):
    self.assertEqual(max_correlation(torch.tensor([[4.0]])), 0.0)
    self.assertAlmostEqual(max_correlation(torch.tensor([[4.0, -1.0], [-1.0, 1.0]])), 0.5)
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -k AlignmentTests -v`
Expected: `ImportError` for `experiments.exp7_independent_subnets.align`.

- [ ] **Step 3: Implement `align.py` to the interfaces above** (`torch.linalg.eigh` for the inverse square roots, `torch.linalg.svd` for `M`).

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -k AlignmentTests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/exp7_independent_subnets/align.py tests/test_exp7_independent_subnets.py
git commit -m "exp7: add the alignment of the learned functions"
```

---

### Task 4: Staged training from an empty directory

**Files:**
- Create: `src/experiments/exp7_independent_subnets/train.py`
- Test: `tests/test_exp7_independent_subnets.py`

**Interfaces:**
- Consumes: Task 1 (`Mode`, `build_mode`, `token_vectors`), Task 2 (`residual_target`, `stage_loss`, `Selection`), Task 3 (`Moments`, `alignment`, `max_correlation`); from exp5 `prepare_run`, `run_metadata`, `draw_pairs`, `sampler_seed`, `FIRST_MODE` (`train.py`), `last_hidden`, `future_only_logits`, `header_tokens`, `pair_targets` (`features.py`), `EvalTable` (`tables.py`); from exp4 `compare_with_exact_svd`, `format_comparison`; `SampleSampler`; `write_atomic`, `save_json`.
- Produces:
  - `train_step(current: Mode, frozen: list[Mode], vectors: Tensor, optimizer, histories: Tensor (B, H), futures: Tensor (B, T), header_ids: Tensor (B, K), target: Tensor (B, K)) -> tuple[float, float]`: one optimizer step on `current`; returns the loss and the mean square of the residual. A non-finite loss raises `RuntimeError`.
  - `mode_table_features(mode: Mode, table: EvalTable, vectors: Tensor, chunk: int) -> tuple[Tensor (N,), Tensor (C,)]`: float64 features of every row and column, the mode in `eval()` under `no_grad`. Rows are `table.table.histories`; columns are `table.future_ids` with header ids `table.table.columns[:, -1].reshape(len(table.future_ids), -1)`.
  - `selection_energy(record: dict) -> float`: `record["selection"]["captured_energy"][-1]`.
  - `train_exp7(*, inference_dir: Path, store_dir: Path, model_path: Path, output_dir: Path, device: str, subnet: str, modes: int, stage_steps: int, min_stage_steps: int, patience: int, align_pairs: int, scratch_layers: int, scratch_width: int, scratch_heads: int, batch_size: int, eval_every: int, lr: float, encoder_lr: float, seed: int, overwrite: bool, encode_batch: int = 1024, eval_chunk: int = 256) -> dict` returning `{"output_dir": str, "step": int, "modes": int, "evaluations": int}` (`modes` is the number of frozen modes, `step` the global step).

**Behavior to implement (decisions the tests below pin):**

- **A stage.** Before building mode `l`, seed torch with `sampler_seed(seed, 10**9 + l)`. Optimizer: Adam; `gpt2` uses `encoder_lr` for the two encoders and `lr` for the two heads; `scratch` uses `lr` for everything with a linear warmup over the stage's first `WARMUP_STEPS` steps. Each step: `draw_pairs`, the frozen model's `last_hidden` of the futures, `header_tokens(future_only_logits(...), top_k)`, `pair_targets`, then `train_step`.
- **Evaluation** at every `eval_every`-th step of the stage and at the stage's last step (`stage_step == stage_steps`). The features of frozen modes on the two tables are computed once, when the mode is frozen. One record:

  ```
  step (global), stage (1-based), stage_step, train_loss, kernel_energy, residual_energy,
  seconds_per_step, selection, report
  ```

  `train_loss`, `kernel_energy` (mean of `target²`) and `residual_energy` are means since the previous evaluation. `selection` and `report` are `compare_with_exact_svd(left (N, l), right (C, l), table, ranks=tuple(range(1, l + 1)))` of exp4's `train` and `val` tables.
- **End of a stage.** After each evaluation call `Selection.record(selection_energy(record), stage_step)` and keep a copy of the current mode's weights when it returns True; then ask `Selection.reason(stage_step)`. When a reason comes back: load the kept weights if `keeps_best`, freeze the mode, write `modes/mode-XX.pt` (two digits, 1-based; a dict with the mode's `state_dict`, `stage`, `best_step`), compute its table features, run the alignment and append one line to `stages.jsonl`.
- **Alignment at the end of a stage.** Draw `align_pairs` fresh pairs (in batches of `batch_size`), forward all `l` modes under `no_grad`, fill `Moments` with the features and the **target** (not the residual), call `alignment`. Write `alignment.pt` (`T_f`, `T_g`, `S`, `modes`). The stage line:

  ```
  stage, steps, best_step, reason, selection_energy, selection, report,
  aligned: {selection, report}, singular_values, left_correlation, right_correlation
  ```

  `selection`/`report` are the unaligned comparisons of the frozen modes; `aligned` uses the table features times `T_f`/`T_g`; `singular_values` is `S`; the correlations are `max_correlation(lam_f)` and `max_correlation(lam_g)`. `best_step` is the stage step whose weights were frozen (`steps` when `patience == 0`).
- **Files.** `eval_history.jsonl`, `stages.jsonl`, `modes/`, `alignment.pt`, `metadata.json` (the config, `run_metadata(...)` with `model` = mode 1 and its `ranks` and `table_shapes` keys replaced by `"ranks": "1..l at stage l"` and shapes keyed `selection`/`report`, the scratch sizes, and a `tables` note saying which exp4 table each key is). `checkpoint.pt` belongs to Task 5; this task still writes it at every evaluation with whatever Task 5 needs, but only Task 5 reads it. An existing `eval_history.jsonl` without `--overwrite` is a `FileExistsError` until Task 5 replaces that rule.
- Printed lines follow exp5: one per evaluation with stage, loss and `format_comparison` of both tables, one per finished stage with the reason and the aligned energies.

- [ ] **Step 1: Write the failing tests**

Class `TrainingTests(Exp5Fixture)` with a helper `train(output_dir, **changes)` whose defaults are `subnet="gpt2", modes=2, stage_steps=4, min_stage_steps=0, patience=0, align_pairs=32, scratch_layers=1, scratch_width=4, scratch_heads=2, batch_size=8, eval_every=2, lr=1e-2, encoder_lr=1e-3, seed=0, overwrite=False, device="cpu"`, stdout captured into `self.log`, and helpers `records(dir)` / `stages(dir)` that read the two jsonl files.

```python
def test_train_step_updates_only_the_current_mode(self):
    # Build three modes, freeze the first two (requires_grad_(False), eval()), run train_step
    # on the third with a hand-made batch from the fixture. Every parameter of the frozen
    # modes is bit-identical afterwards and has grad None; at least one parameter of each of
    # the current mode's four parts changed; the returned residual energy equals
    # residual_target(...).square().mean() computed by hand.

def test_writes_outputs_for_two_stages(self):
    output = self.root / "exp7-outputs"
    summary = self.train(output)
    self.assertEqual(summary, {"output_dir": str(output), "step": 8, "modes": 2, "evaluations": 4})
    records = self.records(output)
    self.assertEqual([(r["stage"], r["stage_step"], r["step"]) for r in records],
                     [(1, 2, 2), (1, 4, 4), (2, 2, 6), (2, 4, 8)])
    self.assertEqual(set(records[0]), {"step", "stage", "stage_step", "train_loss", "kernel_energy",
                                       "residual_energy", "seconds_per_step", "selection", "report"})
    self.assertEqual([r["selection"]["ranks"] for r in records], [[1], [1], [1, 2], [1, 2]])
    # Stage 1 trains on the target itself.
    self.assertAlmostEqual(records[0]["residual_energy"], records[0]["kernel_energy"], places=6)
    stages = self.stages(output)
    self.assertEqual([(s["stage"], s["steps"], s["best_step"], s["reason"]) for s in stages],
                     [(1, 4, 4, "steps"), (2, 4, 4, "steps")])
    self.assertEqual(set(stages[1]["aligned"]), {"selection", "report"})
    self.assertEqual(len(stages[1]["singular_values"]), 2)
    self.assertTrue((output / "modes" / "mode-01.pt").is_file() and (output / "modes" / "mode-02.pt").is_file())
    self.assertEqual(torch.load(output / "alignment.pt", weights_only=True)["T_f"].shape, (2, 2))
    metadata = json.loads((output / "metadata.json").read_text())
    self.assertEqual(metadata["table_shapes"], {"selection": [4, 9], "report": [2, 6]})
    # No file holds the teacher's token vectors or the frozen model.

def test_selection_table_is_exp4_train_and_report_table_is_exp4_val(self):
    # records[-1]["selection"] equals compare_with_exact_svd of the features on
    # load_eval_table(self.inference_dir / "train", ...), rebuilt from modes/mode-*.pt.

def test_third_stage_selects_on_rank_three(self):
    # modes=3: the stage-3 records list ranks [1, 2, 3] and selection_energy(record) is
    # record["selection"]["captured_energy"][2].

def test_last_step_of_a_stage_is_evaluated(self):
    # stage_steps=3, eval_every=2, modes=1 -> stage_step values [2, 3];
    # stage_steps=3, eval_every=5 -> [3].

def test_patience_freezes_the_best_weights_and_patience_zero_the_last(self):
    # With the store unchanged and the same seed, the pairs repeat, so:
    #   a = train(modes=1, stage_steps=2)   -> mode-01 holds the weights after 2 steps
    #   b = train(modes=1, stage_steps=4)   -> the weights after 4 steps (they differ from a's)
    #   c = train(modes=1, stage_steps=6, patience=1), selection_energy patched to return
    #       0.9 then 0.1 -> the stage ends at stage step 4 with reason "patience" and
    #       best_step 2, and mode-01 equals a's, not b's.
    # (a, b and c use three output directories; all three draw the same pairs in stage 1.)

def test_scratch_subnet_trains_with_its_own_small_transformers(self):
    # subnet="scratch": outputs as above; the mode file's encoders have width 4;
    # metadata records scratch_layers/width/heads; the log names the warmup.

def test_two_stages_learn_and_the_second_starts_from_a_smaller_residual(self):
    # stage_steps=300, eval_every=100, batch_size=16, align_pairs=256:
    # stage-1 rank-1 captured energy > 0.5 on both tables (an untrained mode is at or below 0);
    # every stage-2 record has residual_energy < kernel_energy;
    # the aligned selection energy at rank 2 is finite and at most the exact one + 1e-6.

def test_rejects_missing_tables_a_negative_seed_and_another_store(self):
    # As exp6's test of the same name, through train_exp7.

def test_existing_outputs_need_overwrite(self):
    # Removed in Task 5 when resuming replaces this rule: a second run on the same directory
    # raises FileExistsError; with overwrite=True it starts again.
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -k TrainingTests -v`
Expected: `ImportError` for `experiments.exp7_independent_subnets.train`.

- [ ] **Step 3: Implement `train.py` to the interfaces and behavior above**

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -v`
Expected: PASS (the learning test takes the longest; exp6's equivalent runs in well under a minute).

- [ ] **Step 5: Commit**

```bash
git add src/experiments/exp7_independent_subnets/train.py tests/test_exp7_independent_subnets.py
git commit -m "exp7: train the modes in stages and freeze each one"
```

---

### Task 5: Resuming and adding modes

**Files:**
- Modify: `src/experiments/exp7_independent_subnets/train.py`
- Test: `tests/test_exp7_independent_subnets.py`

**Interfaces:**
- Consumes: Task 4's `train_exp7` and files.
- Produces: no new names. `checkpoint.pt` becomes the record of progress:

  ```
  config            every setting except modes, device, overwrite, encode_batch, eval_chunk
  completed_modes   number of frozen modes; modes/mode-01.pt .. mode-<completed>.pt are final
  step              global step
  history           the eval records so far
  stage             None between stages, else a dict:
      stage_step, mode_state_dict, best_state_dict (None unless keeps_best),
      optimizer_state_dict, scheduler_state_dict (None for gpt2), selection (dataclasses.asdict)
  ```

**Behavior to implement:**

- `checkpoint.pt` is written at every evaluation, and again at the end of a stage after the mode file, `stages.jsonl` and `alignment.pt` (then with `stage: None` and `completed_modes` raised by one).
- Starting on a directory with a checkpoint: a different `config` raises `ValueError` mentioning `--overwrite`. Frozen modes are loaded from `modes/` (a missing file for a completed mode raises `ValueError` naming it) and their table features recomputed. A `stage` dict continues that stage at `stage_step` with its weights, best weights, optimizer, scheduler and `Selection`; the loss window starts empty. Print `Resuming stage <l> at stage step <s> (step <n>)`.
- `modes` is not part of `config`. `modes < completed_modes` raises `ValueError` naming both numbers. `modes == completed_modes` with no stage in progress returns the summary at once and writes nothing. `modes > completed_modes` trains the missing stages; `stages.jsonl` and `eval_history.jsonl` grow.
- The sampler is created with `sampler_seed(seed, step)` at the global step the run starts from.
- `overwrite` deletes `checkpoint.pt`, both jsonl files, `alignment.pt` and `modes/` before starting. Without a checkpoint, an existing `eval_history.jsonl` is still a `FileExistsError`.

- [ ] **Step 1: Write the failing tests**

Class `ResumeTests(Exp5Fixture)`, reusing the `train` helper (move it to a mixin). To interrupt a run, patch `experiments.exp7_independent_subnets.train.draw_pairs` with a wrapper that raises `KeyboardInterrupt` on its first call after `eval_history.jsonl` holds `n` records. (Counting calls would be fragile: the alignment draws pairs too.)

```python
def test_resume_in_the_middle_of_a_stage(self):
    # modes=2, stage_steps=6, eval_every=2, patience=3. Interrupt once 5 records exist (stage 2,
    # after its evaluation at stage step 4). mode-01.pt bytes are recorded. Run again:
    # the log says "Resuming stage 2 at stage step 4 (step 10)"; mode-01.pt is byte-identical;
    # the records' steps are [2, 4, 6, 8, 10, 12] with no repeat; stages.jsonl has two lines;
    # the summary is {"step": 12, "modes": 2, "evaluations": 6, ...}.

def test_resume_restores_the_best_weights_and_the_stale_count(self):
    # modes=1, stage_steps=8, eval_every=2, patience=2, selection_energy patched to
    # 0.9, 0.1 before the interrupt (once 2 records exist) and 0.1 after it:
    # the stage ends at stage step 6 with reason "patience" and best_step 2, and mode-01
    # equals the weights of a plain stage_steps=2, patience=0 run.

def test_adding_modes_keeps_the_frozen_ones(self):
    # train(modes=1) then train(modes=3) on the same directory: mode-01.pt is byte-identical,
    # the second run's log has no stage-1 line, records continue at step 5, stages.jsonl has
    # three lines with stages [1, 2, 3].

def test_modes_below_and_equal_to_the_finished_count(self):
    # After train(modes=2): train(modes=1) raises ValueError matching "2" and "1";
    # train(modes=2) returns the same summary and leaves every file's mtime unchanged.

def test_resume_needs_the_same_settings(self):
    # After train(modes=1): each of {"subnet": "scratch"}, {"stage_steps": 6}, {"patience": 1},
    # {"lr": 1e-3}, {"align_pairs": 16}, {"scratch_width": 8} with modes=2 raises ValueError
    # matching "--overwrite"; overwrite=True with a change starts again at stage 1 and
    # removes the old mode files.

def test_a_missing_mode_file_is_reported(self):
    # After train(modes=2), delete modes/mode-01.pt: train(modes=3) raises ValueError naming it.
```

Delete `test_existing_outputs_need_overwrite` from Task 4 and replace it with a check inside `test_resume_needs_the_same_settings` that a directory holding only `eval_history.jsonl` (no checkpoint) raises `FileExistsError`.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -k ResumeTests -v`
Expected: FAIL (`FileExistsError` on the second run of each test).

- [ ] **Step 3: Implement resuming in `train.py` to the behavior above**

- [ ] **Step 4: Run all exp7 tests and see them pass**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/exp7_independent_subnets/train.py tests/test_exp7_independent_subnets.py
git commit -m "exp7: resume a stage and add modes to a finished run"
```

---

### Task 6: Command line, Slurm script and README

**Files:**
- Create: `src/experiments/exp7_independent_subnets/cli.py`, `bluehive_run/exp7-train.sbatch`
- Modify: `pyproject.toml` (`[project.scripts]`: `exp7 = "experiments.exp7_independent_subnets.cli:main"`), `tests/test_cli_startup.py` (add `"experiments.exp7_independent_subnets.cli"` to the module list), `README.md`
- Test: `tests/test_exp7_independent_subnets.py`

**Interfaces:**
- Consumes: `train_exp7` (Task 5); `positive_int` (`data_pipline.cli`), `positive_float` and `EXPERIMENT as TABLES_EXPERIMENT` (`experiments.exp4_table_neural_svd.cli`), `DEFAULT_STORE_DIR` (`experiments.exp5_gpt2_mlp_table.cli`), `INFERENCE_DIR`, `add_run_arguments`, `run_dir` (`utils.experiment_paths`).
- Produces: `main(argv: list[str] | None = None) -> None` with one subcommand, `train`, and the arguments of Global Constraints. `--patience` and `--min-stage-steps` accept 0 (non-negative integers); the others that count something are positive. `--subnet` has `choices=SUBNETS`. `--min-stage-steps` above `--stage-steps` is a usage error. Output directory `run_dir(results_root, EXPERIMENT, dataset, model_path.name) / f"train-{subnet}"`. `ValueError` and `FileExistsError` from `train_exp7` go to `parser.error`. Prints `output_dir`, `step`, `modes`, `evaluations`.

- [ ] **Step 1: Write the failing tests**

Class `CLITests(Exp5Fixture)`:

```python
def test_train_writes_under_the_subnet_directory(self):
    # As exp6's CLI test: copy the tables to <results>/exp4_table_neural_svd/tiny__model/inference,
    # run main(["train", ..., "--subnet", "scratch", "--modes", "1", "--stage-steps", "2",
    #           "--min-stage-steps", "0", "--patience", "0", "--eval-every", "2",
    #           "--align-pairs", "16", "--scratch-layers", "1", "--scratch-width", "4",
    #           "--scratch-heads", "2", "--batch-size", "8", "--device", "cpu"]).
    # The run directory is <results>/exp7_independent_subnets/tiny__model/train-scratch;
    # stdout has "modes: 1"; the same command with --lr 0.5 exits with code 2.

def test_defaults_match_the_design(self):
    # patch train_exp7; main(["train"]) passes
    # subnet "gpt2", modes 8, stage_steps 30000, min_stage_steps 5000, patience 5,
    # align_pairs 16384, scratch_layers 6, scratch_width 384, scratch_heads 6, batch_size 128,
    # eval_every 1000, lr 3e-4, encoder_lr 2e-5, seed 42, device "cuda", overwrite False,
    # model_path Path("models/gpt2-small"),
    # store_dir Path("data/sample_stores/wikitext-103-train__gpt2-small__h64-f32"),
    # inference_dir Path("data/experiment_results/exp4_table_neural_svd/wikitext-large__gpt2-small/inference"),
    # output_dir Path("data/experiment_results/exp7_independent_subnets/wikitext-large__gpt2-small/train-gpt2").

def test_usage_errors(self):
    # "--subnet mlp", "--modes 0", "--patience -1", and "--min-stage-steps 10 --stage-steps 5"
    # each exit with code 2.
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -k CLITests -v`
Expected: `ImportError` for `experiments.exp7_independent_subnets.cli`.

- [ ] **Step 3: Implement `cli.py`, add the script to `pyproject.toml`, add the module to `tests/test_cli_startup.py`**

`train_exp7` is imported inside `main`, after parsing, as in exp6.

- [ ] **Step 4: Write `bluehive_run/exp7-train.sbatch`**

Copy `bluehive_run/exp6-train.sbatch`; job name `exp7-train`; `--mem=48G`; the last line runs `experiments.exp7_independent_subnets train "$@"`. Its comment says: resubmitting continues the current stage; `--modes 1` first, then the same directory with a larger `--modes`; needs the exp4 tables and a store with published blocks.

- [ ] **Step 5: Add the README section "Experiment 7: an independent sub-network per singular function"**

Place it after Experiment 6 and add it to the navigation at the top. It states, in the README's existing style (plain sentences, a `text` diagram, tables for files and columns):

- what differs from exp6: one pair of sub-networks per mode with no shared trainable parameter; stages that train one mode on the pairs' target minus the frozen modes' output; the two sub-network kinds;
- the diagram of one mode from the spec, and why the right side is `<w_l(future), e_z>`;
- when a stage ends (`--stage-steps`, `--min-stage-steps`, `--patience`, and `--patience 0` as a fixed number of steps that freezes the last weights);
- **the two tables' names**: the selection table is exp4's `train` table and the report table is exp4's `val` table; exp7 trains on neither; only the selection table decides when a mode is frozen;
- the alignment, what `aligned` means and that it changes no weights;
- the commands of the first step and how to continue with more modes:

  ```bash
  uv run exp7 train --subnet gpt2 --modes 1
  uv run exp7 train --subnet scratch --modes 1
  uv run exp7 train --subnet gpt2 --modes 8     # continues the same directory
  ```

- every output file and every field of `eval_history.jsonl` and `stages.jsonl`;
- that ranks are listed 1..l (not exp5/exp6's 1, 2, 4, 8), that a finished stage cannot be lengthened, and that only `--modes` may change between runs of one directory;
- the reference values on the report table from the spec's last section (0.45–0.47, 0.51, 0.654) and its table of how to read the first stage;
- the resource estimates from the spec, marked as not measured;
- a CPU smoke run in the form of exp5's.

- [ ] **Step 6: Run everything**

Run: `uv run python -m unittest discover -s tests`
Expected: every test passes, including `test_cli_startup` and the exp4, exp5 and exp6 tests.

Run: `uv run exp7 train -h`
Expected: the help text with the defaults above.

- [ ] **Step 7: Commit**

```bash
git add src/experiments/exp7_independent_subnets/cli.py bluehive_run/exp7-train.sbatch pyproject.toml uv.lock tests/test_cli_startup.py tests/test_exp7_independent_subnets.py README.md
git commit -m "add exp7: an independent sub-network per singular function"
```

(`uv.lock` only if adding the script changed it.)
