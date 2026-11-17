# exp7: BERT-style sub-networks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change exp7's sub-networks from "causal transformer, last token" to "bidirectional transformer, mean pooling": `scratch` becomes a from-scratch BERT-structure encoder, a new kind `roberta` is a pretrained RoBERTa-base, and `gpt2` stays in the code as an unused option.

**Architecture:** Every encoder of a mode becomes a module that maps token ids `(B, T)` to one pooled vector `(B, width)`; the pooling lives inside the encoder wrapper, so `Mode`, the heads, the stages, the alignment and resuming do not change. `roberta` adds an id mapping from GPT-2's token ids to RoBERTa's (same sub-word strings, other numbers) and a path setting.

**Tech Stack:** Python 3.14, PyTorch, transformers (`BertModel`, `RobertaModel`, local files only), `unittest`, uv.

**Spec:** `docs/superpowers/specs/2026-11-16-exp7-independent-subnets-design.md` (Chinese), section "修订（2026-11-16）：子网络改为 BERT 形式" and the rewritten "编码器类型". The first plan, `docs/superpowers/plans/2026-11-16-exp7-independent-subnets.md`, built what this plan changes; its Global Constraints still hold.

## Global Constraints

- Run everything from the repository root. exp7 tests: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -v`. Full suite: `uv run python -m unittest discover -s tests`.
- Do not change the behavior, outputs or checkpoint format of exp4, exp5, exp6 or `data_pipline.sample_store`; import from them only.
- Only the encoders change. The target, the loss, the stages, the stopping rule, the alignment, the comparison with the exact SVDs, the files a run writes and the resume rules stay as they are; their existing tests keep passing with the same assertions (a test may change only where it names a sub-network kind, a default or a state-dict key).
- No trainable parameter is shared between two modes, or between the two sides of a mode. The left sub-network reads the history only; the right sub-network reads the future only.
- Sub-network kinds: `SUBNETS = ("scratch", "roberta", "gpt2")`; the default of `--subnet` is `scratch`.
  - `scratch`: random `transformers.BertModel` without its pooler, bidirectional, dropout 0, fed GPT-2 token ids directly; pooled vector = mean over all positions of the last hidden state.
  - `roberta`: a trainable `RobertaModel` loaded from `--roberta-path` (default `models/roberta-base`) without its pooler, dropout 0; GPT-2 ids are mapped to RoBERTa ids by token string, `<s>` is put in front and `</s>` behind; pooled vector = mean over all positions (the two special tokens included).
  - `gpt2`: as before — a trainable copy of the teacher's transformer; pooled vector = last position's hidden state.
- Learning rates: the pretrained kinds (`roberta`, `gpt2`) use `encoder_lr` for the two encoders and `lr` for the two heads, no warmup; `scratch` uses `lr` for everything with the 1000-step linear warmup.
- Sequences have a fixed length and no padding; no attention mask is needed.
- Models are loaded from local directories only (`local_files_only=True`); nothing in exp7 downloads.
- CLI modules must not import `torch` or `transformers` at import time (`tests/test_cli_startup.py`).
- Code style follows exp6: module docstrings, type hints, `ValueError`/`FileExistsError` for user errors, `write_atomic` for outputs.
- Work on branch `exp7-bert-subnets`. Do not add `reports/` or `research_notes/`.

## Review Focus

1. **A GPT-2 token that RoBERTa's vocabulary does not have** (another tokenizer at `--model-path`, or a wrong `--roberta-path`). Expected: `ValueError` naming how many tokens are missing and one example, before any training (Task 2, `test_a_token_missing_from_roberta_is_rejected`).
2. **`--roberta-path` does not exist** (the model was never downloaded on this machine). Expected: `ValueError` that names the directory and `scripts/download_models.py --models roberta`; with `--overwrite` nothing is deleted first (Task 2, `test_a_missing_roberta_directory_is_reported_before_anything_is_deleted`).
3. **A sequence longer than the encoder's positions** (RoBERTa has 512 usable positions; a `scratch` encoder has as many as the teacher). Expected: `ValueError` naming both lengths instead of an index error inside the embedding (Task 2, `test_too_long_sequences_are_rejected`).
4. **A directory trained with one kind resumed with another, or with another `--roberta-path`.** Expected: the existing settings error that names `--overwrite` (Task 2, `test_resume_needs_the_same_roberta_path`).
5. **Mean pooling silently replaced by one position** (first or last token) would still train. Expected: tests pin the pooled vector as the mean over all positions, and that a `scratch` encoder is bidirectional (Task 1, `test_scratch_is_bidirectional_and_mean_pooled`).

---

### Task 1: Pooled encoders; `scratch` becomes a bidirectional transformer with mean pooling

**Files:**
- Modify: `src/experiments/exp7_independent_subnets/__init__.py`, `model.py`, `train.py` (only the metadata text and the optimizer's choice of kind), `cli.py` (default and help of `--subnet`), `README.md` (exp7 section), `tests/test_exp7_independent_subnets.py`

**Interfaces:**
- Consumes: the existing `Mode`, `build_mode`, `scratch_encoder`, `token_vectors` in `model.py`; `experiments.exp5_gpt2_mlp_table.model.trainable_encoder`, `linear_head`.
- Produces (all in `model.py`):
  - An encoder is an `nn.Module` with an attribute `width: int` and `forward(ids: Tensor (B, T)) -> Tensor (B, width)` (float32), the pooled vector.
  - `class LastToken(nn.Module)`: `LastToken(transformer, width)`; pooled vector = `transformer(input_ids=ids, use_cache=False).last_hidden_state[:, -1]`.
  - `class MeanPool(nn.Module)`: `MeanPool(transformer, width)`; pooled vector = `transformer(input_ids=ids).last_hidden_state.mean(dim=1)`.
  - `scratch_encoder(vocab_size: int, positions: int, layers: int, width: int, heads: int) -> MeanPool` around `BertModel(BertConfig(vocab_size=vocab_size, hidden_size=width, num_hidden_layers=layers, num_attention_heads=heads, intermediate_size=4 * width, max_position_embeddings=positions, type_vocab_size=1, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0), add_pooling_layer=False)`.
  - `Mode.left_features` / `right_vectors` call the encoder and then the head; their shapes and `right_features` are unchanged.
  - `build_mode(...)`: `gpt2` wraps `trainable_encoder(inference)` in `LastToken`; `scratch` uses `scratch_encoder(inference.vocab_size, inference.max_length, ...)`. Signature unchanged in this task.
  - `__init__.py`: `SUBNETS = ("scratch", "roberta", "gpt2")` (`roberta` is rejected by `build_mode` with a `ValueError` until Task 2) and a new name `PRETRAINED = ("roberta", "gpt2")`, which `train.py` uses to choose the learning-rate groups instead of comparing with `"gpt2"`.

- [ ] **Step 1: Write the failing tests** in `ModeTests` (keep the existing ones; adapt them where they name a kind or reach into an encoder):

```python
def test_scratch_is_bidirectional_and_mean_pooled(self):
    # encoder = build_mode(self.inference, "scratch", **SCRATCH).left_encoder
    # - it is a MeanPool around a BertModel with no pooler, width 4, 1 layer, 2 heads,
    #   both dropouts 0, vocabulary and positions of the teacher;
    # - encoder(ids) equals encoder.transformer(input_ids=ids).last_hidden_state.mean(dim=1);
    # - bidirectional: changing the LAST token changes the transformer's hidden state at
    #   position 0 (it would not for a causal model);
    # - the pooled vector differs from the hidden state of the last position and of the first.

def test_gpt2_is_causal_and_reads_the_last_token(self):
    # encoder = build_mode(self.inference, "gpt2", **SCRATCH).left_encoder is a LastToken;
    # encoder(ids) equals the teacher's last hidden state (the existing "starts at the teacher"
    # check); changing the last token leaves the transformer's hidden state at position 0 unchanged.

def test_every_kind_gives_one_vector_per_sequence(self):
    # for "scratch" and "gpt2": encoder.width is the encoder's hidden size and
    # encoder(ids).shape == (B, encoder.width), dtype float32, for histories and futures.
```

In `TrainingTests`: change `test_scratch_subnet_trains_with_its_own_small_transformers` so that it also asserts the mode file's encoder is the BERT structure (a key of the state dict names an `encoder.layer`), and make `test_learning_rates_of_the_parameter_groups` cover `scratch` as before. In `CLITests.test_defaults_match_the_design` the default becomes `subnet="scratch"` and the default output directory ends in `train-scratch`.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -v`
Expected: the new tests fail (`ImportError` for `MeanPool` / `LastToken`), `test_defaults_match_the_design` fails on `subnet`.

- [ ] **Step 3: Implement** the interfaces above; in `cli.py` set the default of `--subnet` to `scratch` and rewrite its help for the three kinds; in `train.py` use `PRETRAINED` for the learning-rate groups and update the metadata text that describes the encoders (per kind: what the encoder is and how it is pooled).

- [ ] **Step 4: Update the README's exp7 section**: the diagram of one mode says "pooled vector" instead of "last token's hidden state"; a table of the kinds (`scratch` default: bidirectional, mean pooling, from scratch; `gpt2`: kept, not used in the planned runs; `roberta`: added by Task 2 — leave it out here); the default in every command; the scratch description in the resource estimates.

- [ ] **Step 5: Run all tests**

Run: `uv run python -m unittest discover -s tests`
Expected: every test passes.

- [ ] **Step 6: Commit**

```bash
git add src/experiments/exp7_independent_subnets tests/test_exp7_independent_subnets.py README.md
git commit -m "exp7: scratch sub-networks are bidirectional with mean pooling"
```

---

### Task 2: The pretrained kind `roberta`

**Files:**
- Modify: `src/experiments/exp7_independent_subnets/model.py`, `train.py`, `cli.py`, `scripts/download_models.py`, `README.md`, `tests/test_exp7_independent_subnets.py`

**Interfaces:**
- Consumes: Task 1's `MeanPool`, `PRETRAINED`, `build_mode`; `utils.inference.CausalLMInference` (`.model_path`, `.vocab_size`, `.max_length`, its tokenizer).
- Produces:
  - `token_id_map(model_path: Path, roberta_path: Path, vocab_size: int) -> Tensor (vocab_size,)` int64: entry `i` is RoBERTa's id of the token string that has id `i` in the tokenizer at `model_path`. Both vocabularies are read with `AutoTokenizer.from_pretrained(path, local_files_only=True).get_vocab()`. Raises `ValueError` when `roberta_path` is not a directory (the message names it and `scripts/download_models.py --models roberta`), and when tokens are missing (the message gives their number and one example).
  - `class RobertaInput(nn.Module)`: `RobertaInput(transformer, id_map, bos_id, eos_id)`; `forward(ids (B, T))` gives the transformer `[bos] + id_map[ids] + [eos]` and returns its `last_hidden_state` object unchanged in form, so that it can sit inside `MeanPool`. `id_map` is a buffer with `persistent=False` (it is derived from the tokenizer files and is saved in no mode file).
  - `roberta_encoder(roberta_path: Path, model_path: Path, vocab_size: int) -> MeanPool` around `RobertaInput(RobertaModel.from_pretrained(roberta_path, add_pooling_layer=False, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, local_files_only=True), ...)`, in `train()` mode; `bos_id` / `eos_id` are the model config's `bos_token_id` / `eos_token_id`.
  - `build_mode(inference, subnet, *, scratch_layers, scratch_width, scratch_heads, roberta_path: Path | None = None) -> Mode`: `roberta` builds two `roberta_encoder(roberta_path, inference.model_path, inference.vocab_size)`; `roberta_path=None` with `subnet="roberta"` is a `ValueError`.
  - `check_lengths(mode: Mode, history_tokens: int, future_tokens: int) -> None` (or the same check inside `build_mode` / `train_exp7`, the implementer's choice): `ValueError` naming both numbers when a side's sequence, with the tokens the encoder adds, does not fit its positions.
  - `train_exp7(..., roberta_path: Path, ...)`: a new required keyword; `str(roberta_path)` is part of the checked `config` for every kind. With `subnet="roberta"` the directory and the token mapping are checked before `--overwrite` deletes anything.
  - `cli.py`: `--roberta-path` (default `models/roberta-base`).
  - `scripts/download_models.py`: a model `roberta` → `FacebookAI/roberta-base` at revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b` into `models/roberta-base`, files `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`. The script's default (`--models` not given) stays the three GPT-2 models; `roberta` is downloaded only when named.

- [ ] **Step 1: Write the failing tests**

A helper in the test file builds a tiny RoBERTa directory for the fixture: a random `RobertaModel(RobertaConfig(vocab_size=VOCAB + 4, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, max_position_embeddings=12, pad_token_id=1, bos_token_id=0, eos_token_id=2))` saved with a word-level tokenizer whose vocabulary holds `<s>`: 0, `<pad>`: 1, `</s>`: 2, `<unk>`: 3 and the fixture's token strings in **reversed** order (so no id agrees with the teacher's). Read `Exp5Fixture.setUpClass` in `tests/test_exp5_gpt2_mlp_table.py` for the teacher's tokenizer (`"[UNK]"` and `"1"` .. `"16"`).

```python
def test_roberta_ids_follow_the_token_strings(self):
    # id_map = token_id_map(self.model_path, roberta_dir, VOCAB): for every teacher id i,
    # roberta_vocab[teacher_token_string(i)] == id_map[i]; no id equals its own index here.

def test_roberta_encoder_wraps_the_sequence_and_mean_pools(self):
    # encoder(ids) equals the mean over all T + 2 positions of the plain RobertaModel's
    # last_hidden_state on [0] + id_map[ids] + [2]; encoder.width == 8; dropouts are 0;
    # the encoder starts at the saved weights (equal to a fresh from_pretrained copy);
    # "id_map" is not a key of the mode's state_dict.

def test_a_token_missing_from_roberta_is_rejected(self):
    # a RoBERTa directory whose tokenizer lacks two of the teacher's tokens:
    # ValueError matching "2" and one of the missing strings.

def test_a_missing_roberta_directory_is_reported_before_anything_is_deleted(self):
    # train(subnet="roberta", roberta_path=missing, overwrite=True) on a finished directory:
    # ValueError matching "download_models.py" and the path; every file's mtime is unchanged.

def test_too_long_sequences_are_rejected(self):
    # a tiny RoBERTa with max_position_embeddings too small for the fixture's histories
    # plus <s> and </s>: ValueError naming both lengths, raised before training.

def test_roberta_modes_share_nothing_and_train(self):
    # two roberta modes and the teacher have pairwise disjoint parameters;
    # train(subnet="roberta", modes=2, roberta_path=roberta_dir) writes the usual outputs,
    # metadata names the roberta path, and the optimizer has the heads at lr and the
    # encoders at encoder_lr with no warmup scheduler.

def test_resume_needs_the_same_roberta_path(self):
    # after train(subnet="roberta", modes=1, roberta_path=a): the same with roberta_path=b
    # and modes=2 raises ValueError matching "--overwrite"; with roberta_path=a it continues
    # and mode-01.pt is byte-identical.
```

`CLITests`: `--roberta-path` defaults to `Path("models/roberta-base")` and is passed through; a run with `--subnet roberta` and the tiny directory writes under `train-roberta`. A test of `scripts/download_models.py` (patch `snapshot_download`): without `--models` it is called for the three GPT-2 models only; with `--models roberta` once, with the repository, revision, directory and files above.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run python -m unittest discover -s tests -p "test_exp7*.py" -v`
Expected: `ImportError` for `token_id_map` / `roberta_encoder`.

- [ ] **Step 3: Implement** the interfaces above. `train_exp7`'s existing callers in the tests get `roberta_path` through the `train` helper's defaults.

- [ ] **Step 4: Update the README**: the models table gets the `models/roberta-base` row (repository `FacebookAI/roberta-base`, the pinned revision) and the download command `uv run python scripts/download_models.py --models roberta`; in the exp7 section add `roberta` to the table of kinds (what it is, how GPT-2 ids are mapped, `<s>`/`</s>`, mean pooling, learning rates), why RoBERTa and not the original BERT (same sub-word strings as GPT-2, so no re-tokenization — one or two sentences from the spec's "修订" section), `--roberta-path` and that it takes part in the settings check, the first-step commands

  ```bash
  uv run python scripts/download_models.py --models roberta   # once per machine
  uv run exp7 train --subnet scratch --modes 1
  uv run exp7 train --subnet roberta --modes 1
  ```

  the table of how to read the first stage as the spec's "怎么读结果" now has it (rows comparing `scratch` with `roberta`, and the row "both below 0.47"), and the resource estimates (`roberta` about 125 M parameters per encoder, estimated, not measured). Mention in `bluehive_run/exp7-train.sbatch`'s comment that `--subnet roberta` needs `models/roberta-base` on the cluster.

- [ ] **Step 5: Run everything**

Run: `uv run python -m unittest discover -s tests` — every test passes.
Run: `uv run exp7 train -h` — shows `--subnet {scratch,roberta,gpt2}` (default: scratch) and `--roberta-path`.

- [ ] **Step 6: Commit**

```bash
git add src/experiments/exp7_independent_subnets scripts/download_models.py bluehive_run/exp7-train.sbatch tests README.md
git commit -m "exp7: add pretrained RoBERTa sub-networks"
```
