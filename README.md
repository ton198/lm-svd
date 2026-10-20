# LM-SVD: GPT-2 Matrix Experiments

Build matrices from local GPT-2 inference, compute their singular value decompositions (SVDs), measure numerical rank, and plot singular-value decay and cumulative energy. For matrices too large to store, learn leading singular components online with NeuralSVD.

Experiments use sampled history/future token sequences but construct different matrices:

| Experiment | Matrix entry | Matrix shape | Matrix file |
| --- | --- | --- | --- |
| `exp1` | Centered next-token logit for a future-specific header token | histories × (futures × top-k) | `matrix.npy` |
| `exp2` | Conditional log probability of an entire future given a history | histories × futures | `feature_log_prob.npy` |
| `exp3` | Difference between two centered-logit matrices using model A's future-specific headers | histories × (futures × top-k) | `matrix.npy` |
| `exp4` | Centered next-token logit for every vocabulary token | histories × (futures × vocabulary) | Not stored; online NeuralSVD |

Experiments **exp1–exp3** have two subcommands, run in order: `inference` builds the matrix, and `analysis` runs SVD → numerical rank → visualization together with a selectable secondary centering strategy. Once a matrix exists, their analysis does not load a language model. Their analysis defaults to CUDA; use `--device cpu` to run without a GPU. **exp4** instead uses `train` → `analysis`, defaults to the larger `wikitext-large` dataset, and analyzes learned features on CPU without loading a model or materializing the matrix.

## 1. Install dependencies

Requirements: Python 3.14 or newer and [uv](https://docs.astral.sh/uv/). Run all commands below from the repository root in a Bash-compatible shell.

```bash
uv sync
```

Matrix generation defaults to CUDA and uses float32 model weights. Use `--device cpu` if CUDA is unavailable. Experiment 2's batch size defaults to 64; reduce `--batch-size` if inference runs out of GPU memory, especially with GPT-2 Medium.

## 2. Download models for offline inference

```bash
# Download or reuse the inference files for both models.
uv run python scripts/download_models.py

# Optional: download just one model.
uv run python scripts/download_models.py --models medium
```

The script downloads **complete inference files**, not every file in the upstream repositories. Each local directory contains `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and `merges.txt`. These include the full model weights, not just a tokenizer. Duplicate PyTorch, TensorFlow, Flax, Rust, and ONNX weight formats are not downloaded.

| Local directory | Hugging Face repository | Pinned revision |
| --- | --- | --- |
| `models/gpt2-small` | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| `models/gpt2-medium` | [openai-community/gpt2-medium](https://huggingface.co/openai-community/gpt2-medium) | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` |

Re-running the script reuses files already downloaded at the pinned revisions. `--output-dir PATH` changes the parent directory of `gpt2-small` and `gpt2-medium`; by default it is the repository's `models/` directory. Internet access is required to fetch missing files. Download the source dataset in the next section before enabling offline mode.

After downloading both models and source data, all tokenization and model inference can run offline.

Both models have been verified with local-only tokenizer loading and CPU inference, including next-token logits and conditional continuation log probabilities.

## 3. Prepare shared history/future samples

### WikiText windows

Download the source dataset before generating history/future windows:

```bash
# Default: WikiText-103 raw, including train, validation, and test.
uv run python scripts/download_samples.py

# The training split alone is sufficient for the default experiments.
uv run python scripts/download_samples.py --splits train

# Optional: download all four WikiText configurations.
uv run python scripts/download_samples.py \
  --configs wikitext-103-raw-v1 wikitext-103-v1 wikitext-2-raw-v1 wikitext-2-v1
```

The script downloads Parquet shards and the dataset card from [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) at revision `b08601e04326c79dfdd32d625aee71d232d685c3`. It reuses files already downloaded at that revision. The default destination is `data/original/wikitext/`; `--output-dir PATH` changes it. Pass that same path to `build-windows --dataset-dir PATH` if you change it. `--configs` selects configurations and `--splits` selects `train`, `validation`, or `test`.

This script downloads **source text data**, not the generated experiment sample file. The separate `build-windows` step below creates history/future token sequences. Its default input files match:

```text
data/original/wikitext/wikitext-103-raw-v1/train-*.parquet
```

Each shard must have a non-null `text` column. `--dataset-dir` selects the dataset root; `--config` selects its subdirectory; `--split` selects the filename prefix.

Once both downloads are complete, enable offline mode and build the shared sample file:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv run data-pipline build-windows \
  --dataset-dir data/original/wikitext \
  --config wikitext-103-raw-v1 \
  --split train \
  --tokenizer-path models/gpt2-small \
  --history-count 512 --history-tokens 64 \
  --future-count 512 --future-tokens 32 \
  --seed 0
```

These exports remain active in the current shell. Before running either download script again, use `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE` to allow network access.

The output is `data/history_future_samples/wikitext.parquet`. If it already exists, reuse it or explicitly pass `--overwrite` to rebuild it. Experiments exp1–exp3 select this file by default with `--dataset wikitext`. GPT-2 Small and Medium use the same GPT-2 token vocabulary, so the same sample file can be reused for comparisons between them.

For exp4, build the larger window set from the same downloaded source:

```bash
uv run data-pipline build-wikitext-large
```

This writes `data/history_future_samples/wikitext-large.parquet` with **10000 histories × 64 tokens and 10000 futures × 32 tokens** by default. It accepts the same window flags as `build-windows`, including `--history-count`, `--future-count`, `--dataset-dir`, `--tokenizer-path`, `--seed` (default 0), `--output`, and `--overwrite`. Only the default counts and output path differ.

The pipeline processes sorted shards and rows, encodes each text row independently, concatenates token IDs without inserting separators or EOS tokens, and samples distinct non-overlapping windows. Windows may cross source-row boundaries. Output columns are `sample_id`, `group`, `token_count`, `start_token`, `token_ids`, and `text`. Experiments use **`token_ids`**, not re-encoded text. Sampling settings and source information are stored in the Parquet schema's `lm_svd` metadata.

Experiments load named sample sets only from the fixed directory `data/history_future_samples/<dataset>.parquet`, relative to the working directory. Use `--dataset NAME`, not a file path. Names must contain only lowercase letters, digits, and hyphens, starting with a letter or digit (`[a-z0-9][a-z0-9-]*`). Prepared names include `wikitext`, `wikitext-large`, and `task-single-text`. If a producer's output directory is changed, place its Parquet file in this fixed directory before using it in an experiment.

### Task × document samples

For task-conditioned prompts instead of unrelated windows, build from the frozen task catalog at `data/original/task_sources/single_text.jsonl`, C4 documents and a local tokenizer. The builder needs no other task-source files. Download the pinned sources as needed:

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
uv run python scripts/download_task_sources.py
# The frozen catalog must already exist at data/original/task_sources/single_text.jsonl.
uv run data-pipline build-task-inputs --tokenizer-path models/gpt2-small
```

The downloader reuses existing files under `data/original/`; `--output-dir` changes that source root. Its Natural Instructions (SNI), FLAN and tasksource-instruct files are used only as references by the `synthesize-single-text-tasks` project skill, not by the builder. The skill generates instructions one at a time with smol subagents, rejects near-duplicates using embedding recall and review, and appends accepted instructions to the frozen catalog. The builder itself does not generate instructions.

Each nonblank catalog line contains `id`, `source`, `category` and `instruction`; IDs must be unique. The frozen catalog contains 1309 histories: 213 SNI, 88 tasksource, 8 FLAN and 1000 synthetic instructions. The builder writes only `data/history_future_samples/task-single-text.parquet`, pairing these histories with 1309 C4 document futures by default for a 1309 × 1309 matrix. Every catalog task must pass deterministic deduplication and token-length checks; rejected catalog tasks cause an error rather than being silently dropped. The C4 revision and catalog's SHA-256 digest are recorded in the output's `lm_svd` Parquet metadata, together with selection settings and passage rejection counts.

Original-source classification is heuristic: it can miss multi-field requirements. Synthetic instructions are reviewed for use with one arbitrary English prose document, but this is **not a semantic compatibility guarantee** between every instruction and every document. Task deduplication, token-length checks, and passage filtering still use deterministic rules.

Histories render as `Instruction: {instruction}` without trailing whitespace; futures render as `\n\nDocument: {passage}\n\nResponse:`. Their concatenation places the instruction before the document and response cue. The builder checks tokenization boundaries and model context length. Histories are ordered by seeded category round-robin, so leading prefixes selected with inference's `--history-count N` are category-balanced.

Builder controls include `--task-catalog` (default `data/original/task_sources/single_text.jsonl`), `--input-count` (1309 futures), `--history-max-tokens` (256), `--passage-min-tokens` / `--passage-max-tokens` (96 / 160 passage tokens), and `--seed` (0). `--tokenizer-path` selects the tokenizer, recorded as `tokenizer_path` in `lm_svd` metadata rather than a filename suffix; rebuilding with another tokenizer targets the same filename and requires `--overwrite` to replace existing files. `--output-dir` relocates the generated Parquet file. Use `--c4-paths` to select local C4 files, including after changing the downloader's destination. Existing outputs are refused unless `--overwrite` is explicit.

Select the task dataset by name for exp1–exp3; dataset-prefixed run directories keep its results separate. exp4 requires equal-length concatenated inputs within each block and is not suitable for these variable-length task/document samples:

```bash
DATASET=task-single-text
RESULTS_ROOT=data/experiment_results
uv run exp1 inference --device cuda --dataset "$DATASET" --results-root "$RESULTS_ROOT"
uv run exp2 inference --device cuda --dataset "$DATASET" --results-root "$RESULTS_ROOT"
uv run exp3 inference --device cuda --dataset "$DATASET" --results-root "$RESULTS_ROOT"
```

Omit sample limits to use all rows, or set `--history-count N --future-count N` for leading subsets. exp2 supports these variable-length document futures, including in the same batch. Its summed log probabilities are length-sensitive, not normalized by future length.

### BlueHive Slurm run

After installing `.venv` and preparing the local models and `task-single-text` samples on the cluster, submit from the repository root:

```bash
cd /scratch/lli81/lm-svd
mkdir -p bluehive_run/logs
sbatch bluehive_run/task-single-text.sbatch
```

To submit only one experiment, use the single-experiment launcher with an explicit experiment name:

```bash
bash submit-single.sh exp3  # Required: select exp1, exp2, or exp3.
```

The launcher requires exactly one experiment name; missing or invalid arguments produce an error without submitting a job. It switches to its own repository directory, creates the log directory, and overrides the existing batch script's array range with a single task ID (`--array=3` for `exp3`). Resource requests, dataset, inference → analysis ordering, and log paths remain the same; no other experiment is submitted. Do not run it alongside another job writing the same experiment's result directory.

Create the log directory before submission, as shown above, so Slurm can open `bluehive_run/logs/slurm-<job-name>-<array-job-id>_<array-task-id>.out` and `.err` before the script starts. The script uses `SLURM_SUBMIT_DIR` as its working directory, so submit from the repository root. Per-experiment inference and analysis logs are created under `bluehive_run/logs/slurm-<job-name>-<array-job-id>_<array-task-id>/`.

The submission creates three independently scheduled array tasks (`--array=1-3`), mapping task IDs 1, 2 and 3 to `exp1`, `exp2` and `exp3`, respectively. Each task requests one H100, 4 CPUs, 16 GiB host memory and a maximum of two hours (`02:00:00`). Each runs only its assigned experiment's inference, then its matching analysis; any failed command stops that task, so failed inference skips analysis. The script uses all dataset rows and the default result directories, without `--overwrite`: existing outputs cause failure rather than being silently replaced. Do not submit overlapping arrays or other jobs against those same result directories.

The `h100` partition and `gpu:H100:1` GRES spelling must match the cluster configuration. Each task's resource request is not a measured guarantee that its experiment's full dataset will fit in memory or finish within two hours; reduce inference batch size for GPU-memory pressure or adjust the host-memory/time request based on observed usage.

## 4. Experiment 1: centered-logit matrix

For each future, select its top-k next-token candidates using the **future alone**. These tokens define that future's matrix columns. For each history/future pair, evaluate the concatenated history and future, subtract the mean logit over the entire vocabulary, and retain the selected header tokens:

```text
A[history, (future, token)] = logit(token | history + future)
                            - mean_vocabulary(logits | history + future)
```

Rows correspond to histories. Columns are grouped by future, then descending future-only logit rank. This is a centered-logit matrix, not a probability matrix.

### Full workflow

Set `MODEL` to `gpt2-small` or `gpt2-medium`; the remaining commands work unchanged. No sample limits are passed, so all histories and futures in the sample file are used. With the default samples and top-k 50, the shape is `512 × 25600`.

```bash
MODEL=gpt2-small
DATASET=wikitext
RESULTS_ROOT=data/experiment_results
RESULT_DIR="$RESULTS_ROOT/exp1_L_M_table/${DATASET}__${MODEL}"

uv run exp1 inference \
  --model-path "models/$MODEL" \
  --device cuda \
  --top-k 50 \
  --dataset "$DATASET" --results-root "$RESULTS_ROOT"

uv run exp1 analysis --model-path "models/$MODEL" \
  --dataset "$DATASET" --results-root "$RESULTS_ROOT" --centering none
```

`inference` writes `matrix.npy`, `rows.parquet`, `columns.parquet`, and `metadata.json` under `$RESULT_DIR/inference/`. The Parquet files map rows and columns to sample IDs, header ranks, token IDs, and future-only centered logits.

### Small CPU smoke run

This uses GPT-2 Small and the leading samples from the same input file. `mktemp` creates a fresh directory so existing full results are not touched. The expected shape is `1 × 2`.

```bash
RESULTS_ROOT=$(mktemp -d /tmp/lm-svd-exp1.XXXXXX)
RESULT_DIR="$RESULTS_ROOT/exp1_L_M_table/wikitext__gpt2-small"
uv run exp1 inference --device cpu \
  --history-count 1 --future-count 1 --top-k 2 \
  --dataset wikitext --results-root "$RESULTS_ROOT"
uv run exp1 analysis --device cpu --model-path models/gpt2-small \
  --dataset wikitext --results-root "$RESULTS_ROOT" --centering global
echo "Results: $RESULT_DIR"
```


## 5. Experiment 2: conditional future log-probability matrix

For each history/future pair, sum the conditional log probabilities of the future's tokens:

```text
B[history, future] = sum_t log p(future_t | history + future_<t)
```

These are log-probability **sums**, not averages, probabilities, or centered logits. Rows correspond to histories; columns correspond to futures. Futures may have different token lengths, including within a batch; the summed scores remain length-sensitive.

### Full workflow

Set `MODEL` to `gpt2-small` or `gpt2-medium`. All histories and futures in the sample file are used. The default sample file produces a `512 × 512` matrix. Lower `--batch-size` if necessary; it controls how many futures are scored together for each history.

```bash
MODEL=gpt2-small
DATASET=wikitext
RESULTS_ROOT=data/experiment_results
RESULT_DIR="$RESULTS_ROOT/exp2_PFH/${DATASET}__${MODEL}"

uv run exp2 inference \
  --model-path "models/$MODEL" \
  --device cuda \
  --batch-size 64 \
  --dataset "$DATASET" --results-root "$RESULTS_ROOT"

uv run exp2 analysis --model-path "models/$MODEL" \
  --dataset "$DATASET" --results-root "$RESULTS_ROOT" --centering none
```

`inference` writes `feature_log_prob.npy` and `metadata.json` under `$RESULT_DIR/inference/`. Keep the matrix filename unchanged: `exp2 analysis` reads `inference/feature_log_prob.npy`.

### Small CPU smoke run

The expected shape is `2 × 2`.

```bash
RESULTS_ROOT=$(mktemp -d /tmp/lm-svd-exp2.XXXXXX)
RESULT_DIR="$RESULTS_ROOT/exp2_PFH/wikitext__gpt2-small"
uv run exp2 inference --device cpu \
  --history-count 2 --future-count 2 --batch-size 2 \
  --dataset wikitext --results-root "$RESULTS_ROOT"
uv run exp2 analysis --device cpu --model-path models/gpt2-small \
  --dataset wikitext --results-root "$RESULTS_ROOT" --centering none
echo "Results: $RESULT_DIR"
```

## 6. Experiment 3: difference between centered-logit matrices

`exp3` computes the exp1 matrix for two models on the same sampled histories and futures, then stores `D = L_A - L_B`. Model A selects the top-k header tokens from each future-only context; model B is evaluated on those exact token IDs, so each difference column refers to the same future/token pair in both matrices. The models must use the same token-ID-to-token mapping. Equal vocabulary size is checked, but cannot establish tokenizer identity.

```text
D[history, (future, token)] = L_A[history, (future, token)]
                            - L_B[history, (future, token)]
L[history, (future, token)] = centered logit(token | history + future)
```

### Full workflow

By default, model A is GPT-2 Small, model B is GPT-2 Medium, and all samples are used. The matrix is `512 × 25600` at top-k 50.

```bash
MODEL_A=gpt2-small
MODEL_B=gpt2-medium
DATASET=wikitext
RESULTS_ROOT=data/experiment_results
RESULT_DIR="$RESULTS_ROOT/exp3_L_M_diff/${DATASET}__${MODEL_A}__minus__${MODEL_B}"

uv run exp3 inference --device cuda --top-k 50 \
  --model-a-path "models/$MODEL_A" --model-b-path "models/$MODEL_B" \
  --dataset "$DATASET" --results-root "$RESULTS_ROOT"
uv run exp3 analysis \
  --model-a-path "models/$MODEL_A" --model-b-path "models/$MODEL_B" \
  --dataset "$DATASET" --results-root "$RESULTS_ROOT" --centering none
```

Override the models with `--model-a-path` and `--model-b-path`. `inference` writes `matrix.npy` (the difference), `matrix_a.npy`, `matrix_b.npy`, `rows.parquet`, `columns.parquet`, and `metadata.json` under `$RESULT_DIR/inference/`. Column tokens and future-only logits in `columns.parquet` come from model A. Analysis processes only the difference `matrix.npy`, not the two component matrices. `global` and `row` centering are linear, so `center(L_A - L_B) = center(L_A) - center(L_B)`; `deflate1` is not linear because it removes the leading singular component of the difference matrix itself.

## 7. Experiment 4: online full-vocabulary NeuralSVD

exp4 learns leading singular components of the exp1 centered-logit matrix **without top-k selection**:

```text
A[h, (f, z)] = logit(z | h + f) - mean_vocabulary(logits | h + f)
```

Columns are future-major, then token ID over the full vocabulary. With the default `wikitext-large` samples and GPT-2 Small's vocabulary of 50257, the matrix is `10000 × 502570000` (about 20 TB in float32). It is never stored: training samples history and future subsets independently, generates their full-vocabulary logit block online, and feeds it to `NestedLoRASVDLoss`.

For a bias-free output layer with token embeddings `e_z`, centered logits equal `s(h, f) · (e_z - mean_e)`, where `s` is the final hidden state. exp4 uses a sparse history lookup for left features and the right parameterization `g_l(f, z) = W_f[l] · (e_z - mean_e)`, with `W_f` of shape `(rank, hidden_size)`. Thus right features need no lookup table over all future/token pairs. Training uses SparseAdam and rejects output layers with bias.

### Workflow and defaults

```bash
uv run data-pipline build-wikitext-large
uv run exp4 train
uv run exp4 analysis
```

Both stages default to `--dataset wikitext-large`, `--model-path models/gpt2-small`, `--results-root data/experiment_results`, and `--device cuda`; use `--device cpu` for either stage without CUDA. Training uses all available histories and futures, rank 32, sequential nesting, history/future batches of 64 each, learning rate `1e-2`, seed 0, and logging every 100 steps. There is no fixed step limit: every `--eval-every 1000` steps, training estimates all learned singular values using the analysis formula. It stops when their maximum relative change is below `--tol 1e-3` for `--patience 3` consecutive evaluations. Set `--max-steps` to cap total steps across resumes; the default is unlimited. Use `--history-count` / `--future-count` for leading subsets, `--history-batch` / `--future-batch` for block sizes, `--rank`, `--lr`, `--log-every`, or `--nesting joint` to adjust training.

Runs are stored under `<results-root>/exp4_neural_svd/<dataset>__<model-directory-name>/`. Pass matching dataset, results-root, and model-path flags to both stages for nondefault runs. The provided Slurm workflows cover exp1–exp3 only, not exp4.

### Outputs and interpretation

- `train/`: while training, `checkpoint.pt` stores feature weights, Adam moments, RNG state, loss history, and convergence state every `--checkpoint-every 100` steps and when capped. A default-size checkpoint is about 3 GB. Training automatically resumes from it; interruption between checkpoints loses at most 99 steps at the default interval. Each checkpoint also refreshes six analysis-ready snapshot files: `features.pt` (history and future lookup weights), `token_gram.npy` (the centered token embedding Gram matrix), `loss_history.npy`, `histories.parquet`, `futures.parquet`, and `metadata.json` (matrix definition, training settings/results, convergence statistics, and source provenance). Each file is replaced atomically, with metadata last; concurrent analysis can read files from adjacent snapshots. At convergence, training writes the final snapshot and removes the checkpoint.
- `analysis/`: `singular_values.npy`, `left_singular_vectors.npy` (unit-norm columns), `spectrum.json`, and `neural_svd_spectrum.png`. The plot shows singular values on a log scale, cumulative captured energy, and training loss.

Analysis computes feature statistics in torch float64 on the selected device and reads only `train/{features.pt,token_gram.npy,loss_history.npy,metadata.json}`. Future weights remain on CPU and move to the device in chunks of at most 1024 futures, using a flat 2-D matrix product with the token Gram matrix. Downstream diagnostics and output processing remain in NumPy. `--model-path` supplies only the directory name for locating the run; no language model or sample file is loaded. The token Gram matrix gives right-feature norms and inner products without constructing the enormous right-feature array. Analysis does **not** compute an exact matrix SVD, save right singular vectors, or estimate numerical rank. It does not accept `--centering`: no secondary centering is available because the matrix is not stored.

For a capped feasibility smoke, run `uv run exp4 train --max-steps 2 --checkpoint-every 1 --log-every 1 --results-root /tmp/exp4-smoke`. It prints `stop_reason: max_steps` and leaves a checkpoint and analysis-ready snapshot. Run `uv run exp4 analysis --results-root /tmp/exp4-smoke` to analyze it; the summary reports the training step and convergence status. Rerun training with `--max-steps 4` to continue from step 2, or omit the cap to train until convergence. Rerun analysis with `--overwrite` to refresh its outputs. Add `--device cpu` to either command to run without CUDA.

Singular values retain **learned order**, not sorted order. `spectrum.json` reports maximum absolute off-diagonal inner products for normalized left/right features (`left_orthogonality_error`, `right_orthogonality_error`), a sampled squared Frobenius norm estimate, cumulative captured energy, and training metadata. Captured energy divides cumulative squared learned singular values by that sampled norm estimate; it can exceed 1 before convergence and is reported without clipping. These are learned estimates, not a guarantee of convergence or exact leading singular values; inspect orthogonality, ordering, energy, and loss together.

Training resumes by default when a checkpoint exists, including when snapshot outputs exist alongside it. Resume requires matching sample/model paths, sample counts, rank, nesting, batch sizes, learning rate, seed, and convergence settings; device, logging/checkpoint intervals, and the total-step cap may change. Existing training outputs without a checkpoint require `--overwrite`, which discards both checkpoint and training outputs and restarts at step 0. Analysis also requires `--overwrite` to replace its outputs. Replacing training outputs does not refresh analysis; rerun `exp4 analysis --overwrite` when needed. Use `uv run exp4 <subcommand> --help` or `uv run python -m experiments.exp4_neural_svd <subcommand>`.

## 8. Understand exp1–exp3 analysis outputs

Experiments exp1–exp3 share the same `analysis` workflow. It reads the matrix and optional sibling `metadata.json` from `<result-dir>/inference/`, then writes all six outputs to `<result-dir>/analysis/<centering>/`:

```text
<results-root>/<experiment>/<run>/
├── inference/
│   ├── <matrix file and other inference outputs>
│   └── metadata.json
└── analysis/
    ├── none/
    │   ├── svd_U.npy
    │   ├── svd_S.npy
    │   ├── svd_Vt.npy
    │   ├── svd_metadata.json
    │   ├── rank.json
    │   └── svd_spectrum.png
    ├── global/  # same six files
    ├── row/     # same six files
    └── deflate1/  # same six files
```

Analysis computes a thin float64 SVD with PyTorch on `--device cuda` by default. CUDA uses the accurate `gesvd` driver; `--device cpu` uses PyTorch's CPU SVD. Invalid devices and unavailable CUDA produce an error rather than silently switching devices. If GPU memory is insufficient, rerun that analysis with `--device cpu`. Saved factors remain NumPy arrays, and the numerical-rank tolerance is unchanged.

`<run>` is a flat, string-joined directory name: `<dataset>__<model>` for exp1/exp2 (for example, `wikitext__gpt2-small`), or `<dataset>__<model_a>__minus__<model_b>` for exp3 (for example, `wikitext__gpt2-small__minus__gpt2-medium`). Model labels are the final directory names of the model paths. The dataset is not a separate directory level. Only the requested strategy directory is created.

### Secondary centering

`analysis --centering {none,global,row,deflate1}` selects how the stored matrix `M` is transformed before SVD; the default is `none`:

| Strategy | Matrix decomposed |
| --- | --- |
| `none` | `M` |
| `global` | `M - mean(M)` |
| `row` | `M - mean(M, axis=1, keepdims=True)` |
| `deflate1` | `M - S[0] * outer(U[:, 0], Vt[0, :])`, where `M = U @ diag(S) @ Vt` |

Row centering subtracts each history's mean across all columns: futures × header tokens for exp1/exp3, or futures for exp2. This is **secondary** centering: it does not change the base vocabulary-mean logit subtraction in exp1/exp3, and exp2 inference still stores uncentered log-probability sums. Selecting only top-k header tokens can leave the stored exp1/exp3 matrix biased toward large values despite its base vocabulary centering.

`deflate1` removes the leading singular component of `M`. It computes that component with a separate thin float64 SVD of `M` on the analysis `--device`, so this strategy runs two SVDs. The transformed matrix is then analyzed like every other strategy. Its spectrum equals the `none` spectrum without `S[0]`, followed by one rounding-level singular value for the removed direction. Rank uses the standard rule on this spectrum, so its tolerance scales with the original second singular value.

For example, compare all strategies without rerunning inference:

```bash
MODEL=gpt2-small
DATASET=wikitext
RESULTS_ROOT=data/experiment_results
for CENTERING in none global row deflate1; do
  uv run exp2 analysis --model-path "models/$MODEL" \
    --dataset "$DATASET" --results-root "$RESULTS_ROOT" --centering "$CENTERING"
done
```

If `none` was already analyzed, add `--overwrite` to regenerate it.

- **SVD:** Float64 centering and thin SVD of the selected matrix, regardless of the original stored dtype. `(U * S) @ Vt` reconstructs that matrix, not the original stored matrix unless centering is `none`. The centered matrix is not saved separately. `svd_metadata.json` records `centering`, `centering_formula`, input path, shape and original dtype, computation dtype, singular-value extrema, the centered Frobenius norm and relative reconstruction error, and source matrix metadata under `matrix_metadata`. New SVD metadata does not contain rank fields. Inference and stored matrix dtypes are unchanged.
- **Numerical rank:** Counts singular values strictly greater than `S.max() * max(matrix.shape) * finfo(float64).eps` for all input dtypes and all four centering strategies. `rank.json` records the rank, tolerance, rule, matrix shape, input dtype, and singular-value count; `input_dtype` records provenance only and does not determine epsilon. Existing SVD metadata with legacy rank fields is accepted; rank is recomputed from the singular values.
- **Visualization:** `svd_spectrum.png` plots only singular values strictly greater than the numerical-rank tolerance for all four centering strategies. It contains singular-value decay with logarithmic index and singular-value axes, plus cumulative squared Frobenius energy normalized by the retained values' energy, with linear index and energy axes. Indices start at 1. Rank markers appear in both panels; no tolerance line is drawn. The full saved SVD and `rank.json`, including its tolerance, are unchanged by visualization. It uses Matplotlib's headless Agg backend. An all-zero spectrum is rejected because normalized energy is undefined, including when centering produces an all-zero matrix; SVD and rank outputs have already been written in that case. Existing manual `singular_value_decay.png` plots are untouched.

Older float32 analyses used float32 epsilon for rank, so refresh their `rank.json` and `svd_spectrum.png` before using them. The saved singular values already come from float64 SVD: reuse them to recompute rank and redraw the plot without loading a model, rerunning inference, or repeating SVD. For each existing strategy directory, use the Python API (replace the path with your own):

```python
from pathlib import Path
from utils.analysis import compute_rank, visualize_svd

svd_dir = Path("data/experiment_results/exp2_PFH/wikitext__gpt2-small/analysis/none")
compute_rank(svd_dir=svd_dir, overwrite=True)
visualize_svd(svd_dir=svd_dir, overwrite=True)
```

### Directory selection and existing results

**Analysis does not infer the model or dataset from your last `inference` call.** Both subcommands default to `--dataset wikitext` and `--results-root data/experiment_results`, with `--model-path models/gpt2-small` for exp1/exp2, or `--model-a-path models/gpt2-small --model-b-path models/gpt2-medium` for exp3. Pass matching dataset, results-root, and model flags to inference and analysis for nondefault runs, as shown above.

Analysis uses only the model paths' directory names to locate the run: it does not load a model, check that the model directory exists, or require/check the sample file. It validates only the dataset name's format. Inference resolves the named sample file in the fixed samples directory. `--results-root PATH` selects the parent of the experiment directory, not the run root or its `inference/` or `analysis/` child.

Existing WikiText results have moved to the dataset-prefixed run directories above, and prepared sample files have been renamed to the plain dataset filenames. Historical `metadata.json` files are unchanged: their original `samples_path` values remain records of the inputs used at the time, even though those files have since been renamed.

Inference refuses to replace its existing outputs without `--overwrite`. Analysis prechecks all six outputs for the selected strategy and refuses to start if any exists unless `--overwrite` is supplied; this prevents an existing rank or plot file from causing a late overwrite conflict after SVD. `analysis --overwrite` replaces only that strategy's analysis outputs, not inference files or other strategies. When regenerating inference, rerun analysis with `--overwrite` for every previously computed strategy you intend to use; overwriting inference does not refresh analysis.

Other useful options:

- `--dataset NAME`: select a named sample set for inference or analysis; default `wikitext`.
- `--results-root PATH`: select the results root for inference or analysis; default `data/experiment_results`.
- `inference --history-count N --future-count N`: use only the leading N samples of each group; omit both to use all available samples.
- `exp1 inference --top-k K`: select K header tokens per future; default 50.
- `exp2 inference --batch-size N`: score N futures per inference batch; default 64.
- `inference --device cpu`: use CPU instead of the default CUDA device.

Use `uv run exp1 <subcommand> --help`, `uv run exp2 <subcommand> --help`, or `uv run exp3 <subcommand> --help` for available options. Module entry points also work: `uv run python -m experiments.exp1_L_M_table <subcommand>`, `uv run python -m experiments.exp2_PFH <subcommand>`, and `uv run python -m experiments.exp3_L_M_diff <subcommand>`.

## Python inference API

`CausalLMInference` loads a local model once in float32 and reuses it for token-ID inputs. It does not download files or tokenize text. CUDA is the default device; CPU is shown here for portability:

```python
from utils.inference import CausalLMInference

inference = CausalLMInference("models/gpt2-medium", device="cpu")
logits = inference.next_token_logits([15496, 11])
batch_logits = inference.next_token_logits_batch([[15496, 11], [995, 0]])
scores = inference.continuation_log_probs(
    [15496, 11], [[995, 0]], batch_size=1
)
```

`next_token_logits` returns unnormalized logits as a CPU float32 tensor of shape `(vocab_size,)`. `continuation_log_probs` returns one summed conditional log probability per continuation as a CPU float64 tensor, preserving input order. Inputs must be non-empty integer token-ID sequences within the model vocabulary and context limit; continuations in one call may have different lengths. Scores are sums, not length-normalized averages. Load models sequentially when memory is limited.

`next_token_logits_batch` takes a non-empty batch of non-empty token-ID sequences with **equal lengths** (no padding); each sequence is validated against the vocabulary and context limit. It returns float32 logits of shape `(batch_size, vocab_size)` **on the inference device**, unlike the CPU return from `next_token_logits`. Unequal lengths raise `ValueError`; this is why exp4 uses fixed-length windows rather than variable-length task/document inputs.

## NeuralSVD utility API

`utils.NestedLoRASVDLoss` ports the SVD loss and nesting masks from Ryu et al.'s
NestedLoRA (ICML 2024). It accepts row features `f` of shape `(B_rows, rank)`,
column features `g` of shape `(B_cols, rank)`, and the corresponding matrix block
`a_block` of shape `(B_rows, B_cols)`. Sample rows and columns independently.
Features may come from trainable lookup tables or encoders; the utility does not
load or save matrices.

```python
from utils.neural_svd import NestedLoRASVDLoss, neural_svd_spectrum

loss_fn = NestedLoRASVDLoss(rank=3)  # joint nesting; optionally step=2
# f, g are model outputs; a_block contains A(row_i, column_j).
loss = loss_fn(f, g, a_block)
loss.backward()
# After training, evaluate features on the desired rows and columns:
spectrum = neural_svd_spectrum(f.detach(), g.detach())
```

Move `loss_fn` to the features' device with `.to(device)`. It computes both
normalized operator products internally, converts `a_block` to `f`'s dtype and
device, and does not differentiate through the block. `sequential=True` selects
the original asymmetric prefix gradients and requires `step=1`.
`nesting_masks(rank, step=1, sequential=False)` exposes the masks separately.

`neural_svd_spectrum` returns a `NeuralSVDSpectrum` with `singular_values`, `left`,
and `right`; columns of the latter two have unit mean square. For a finite
`m × n` matrix evaluated on all rows and columns uniformly, multiply the returned
singular values by `sqrt(m * n)` to recover matrix singular values, and divide
`left` and `right` by `sqrt(m)` and `sqrt(n)` for unit-norm vectors. Results retain
the learned column order: they are not sorted, so unconverged ordering remains
visible. exp4 uses the same learned-order convention with closed-form right-feature moments; exp1–exp3 retain their exact matrix-analysis workflows.

## Tests

```bash
uv run python -m unittest discover -s tests
```
