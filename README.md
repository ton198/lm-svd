# LM-SVD: GPT-2 Matrix Experiments

Build matrices from local GPT-2 inference, compute their singular value decompositions (SVDs), measure numerical rank, and plot singular-value decay and cumulative energy.

Experiments use sampled history/future token sequences but construct different matrices:

| Experiment | Matrix entry | Matrix shape | Matrix file |
| --- | --- | --- | --- |
| `exp1` | Centered next-token logit for a future-specific header token | histories × (futures × top-k) | `matrix.npy` |
| `exp2` | No-BOS pointwise mutual information of history and future | histories × futures | `pmi.npy` |
| `exp3` | Difference between two centered-logit matrices using model A's future-specific headers | histories × (futures × top-k) | `matrix.npy` |

Experiments **exp1–exp3** have two subcommands, run in order: `inference` builds the matrix, and `analysis` runs SVD → numerical rank → visualization together with a selectable secondary centering strategy. Once a matrix exists, their analysis does not load a language model. Their analysis defaults to CUDA; use `--device cpu` to run without a GPU.

## Navigation

- [Install dependencies](#1-install-dependencies) · [Download models](#2-download-models-for-offline-inference) · [Prepare data](#3-prepare-shared-historyfuture-samples) · [Sample generation and examples](docs/samples.md)
- [exp1](#4-experiment-1-centered-logit-matrix) · [exp2](#5-experiment-2-no-bos-historyfuture-pmi-matrix) · [exp3](#6-experiment-3-difference-between-centered-logit-matrices) · [exp1–exp3 analysis](#7-understand-exp1exp3-analysis-outputs)
- [Python inference API](#python-inference-api) · [Tests](#tests)

## Project layout

| Path | Purpose |
| --- | --- |
| `src/experiments/` | Experiment commands and workflows |
| `src/data_pipline/` | Data preparation, rolling training-sample generation and random sampling |
| `src/utils/` | Shared inference and matrix-analysis utilities |
| `scripts/` | Download and result synchronization scripts |
| `bluehive_run/` | BlueHive Slurm job scripts and logs |
| `tests/` | Automated tests |
| `docs/` | Additional documentation, including [sample generation](docs/samples.md) |
| `data/`, `models/` | Local datasets, results, token streams, and model files |

## 1. Install dependencies

Requirements: Python 3.14 or newer and [uv](https://docs.astral.sh/uv/). Run all commands below from the repository root in a Bash-compatible shell.

```bash
uv sync
```

Matrix generation defaults to CUDA and uses float32 model weights. Use `--device cpu` if CUDA is unavailable. Experiment 2's batch size defaults to 64; reduce `--batch-size` if inference runs out of GPU memory, especially with GPT-2 Medium.

## 2. Download models for offline inference

```bash
# Download or reuse inference files for GPT-2 Small, Medium, and Large.
uv run python scripts/download_models.py

# Optional: download just one model.
uv run python scripts/download_models.py --models medium
```

The script downloads **complete inference files**, not every file in the upstream repositories. Each GPT-2 directory contains `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and `merges.txt`. These include the full model weights, not just tokenizers. Duplicate PyTorch, TensorFlow, Flax, Rust, and ONNX weight formats are not downloaded.

| Local directory | Hugging Face repository | Pinned revision |
| --- | --- | --- |
| `models/gpt2-small` | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| `models/gpt2-medium` | [openai-community/gpt2-medium](https://huggingface.co/openai-community/gpt2-medium) | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` |
| `models/gpt2-large` | [openai-community/gpt2-large](https://huggingface.co/openai-community/gpt2-large) | `32b71b12589c2f8d625668d2335a01cac3249519` |

Re-running the script reuses files already downloaded at the pinned revisions. `--output-dir PATH` changes the parent directory of all three model directories; by default it is the repository's `models/` directory. Use `--models large` for GPT-2 Large. Internet access is required to fetch missing files. Download the source dataset in the next section before enabling offline mode.

After downloading the required models and source data, all tokenization and model inference can run offline.

GPT-2 Small and Medium have been verified with local-only tokenizer loading and CPU inference, including next-token logits and conditional continuation log probabilities.

## 3. Prepare shared history/future samples

For how each sample set is generated, with examples traced back to the raw sources, see [docs/samples.md](docs/samples.md).

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

The output is `data/history_future_samples/wikitext.parquet`. If it already exists, reuse it or explicitly pass `--overwrite` to rebuild it. Experiments exp1 and exp3 select this file by default with `--dataset wikitext`; exp2 samples raw WikiText internally. GPT-2 Small and Medium use the same GPT-2 token vocabulary, so the same sample file can be reused for comparisons between them.

To build a larger window set from the same downloaded source:

```bash
uv run data-pipline build-wikitext-large
```

This writes `data/history_future_samples/wikitext-large.parquet` with **10000 histories × 64 tokens and 10000 futures × 32 tokens** by default. It accepts the same window flags as `build-windows`, including `--history-count`, `--future-count`, `--dataset-dir`, `--tokenizer-path`, `--seed` (default 0), `--output`, and `--overwrite`. Only the default counts and output path differ.

The pipeline processes sorted shards and rows, encodes each text row independently, concatenates token IDs without inserting separators or EOS tokens, and samples distinct non-overlapping windows. Windows may cross source-row boundaries. Output columns are `sample_id`, `group`, `token_count`, `start_token`, `token_ids`, and `text`. Experiments use **`token_ids`**, not re-encoded text. Sampling settings and source information are stored in the Parquet schema's `lm_svd` metadata.

Experiments using sample sets load them only from the fixed directory `data/history_future_samples/<dataset>.parquet`, relative to the working directory. Use `--dataset NAME`, not a file path. Names must contain only lowercase letters, digits, and hyphens, starting with a letter or digit (`[a-z0-9][a-z0-9-]*`). Prepared names include `wikitext`, `wikitext-large`, and `task-single-text`. If a producer's output directory is changed, place its Parquet file in this fixed directory before using it in an experiment.

### Task × document samples

For task-conditioned prompts instead of unrelated windows, build from the frozen task catalog at `data/original/task_sources/single_text.jsonl`, C4 documents and a local tokenizer. The builder needs no other task-source files. Download the pinned sources as needed:

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
uv run python scripts/download_task_sources.py
# The frozen catalog must already exist at data/original/task_sources/single_text.jsonl.
uv run data-pipline build-task-inputs --tokenizer-path models/gpt2-small
```

The downloader reuses existing files under `data/original/`; `--output-dir` changes that source root. The builder does not use the downloaded Natural Instructions (SNI), FLAN or tasksource-instruct files and does not generate instructions.

Each nonblank catalog line contains `id`, `source`, `category` and `instruction`; IDs must be unique. The frozen catalog contains 1309 histories: 213 SNI, 88 tasksource, 8 FLAN and 1000 synthetic instructions. The builder writes only `data/history_future_samples/task-single-text.parquet`, pairing these histories with 1309 C4 document futures by default for a 1309 × 1309 matrix. Every catalog task must pass deterministic deduplication and token-length checks; rejected catalog tasks cause an error rather than being silently dropped. The C4 revision and catalog's SHA-256 digest are recorded in the output's `lm_svd` Parquet metadata, together with selection settings and passage rejection counts.

Original-source classification is heuristic: it can miss multi-field requirements. Synthetic instructions are reviewed for use with one arbitrary English prose document, but this is **not a semantic compatibility guarantee** between every instruction and every document. Task deduplication, token-length checks, and passage filtering still use deterministic rules.

Histories render as `Instruction: {instruction}` without trailing whitespace; futures render as `\n\nDocument: {passage}\n\nResponse:`. Their concatenation places the instruction before the document and response cue. The builder checks tokenization boundaries and model context length. Histories are ordered by seeded category round-robin, so leading prefixes selected with inference's `--history-count N` are category-balanced.

Builder controls include `--task-catalog` (default `data/original/task_sources/single_text.jsonl`), `--input-count` (1309 futures), `--history-max-tokens` (256), `--passage-min-tokens` / `--passage-max-tokens` (96 / 160 passage tokens), and `--seed` (0). `--tokenizer-path` selects the tokenizer, recorded as `tokenizer_path` in `lm_svd` metadata rather than a filename suffix; rebuilding with another tokenizer targets the same filename and requires `--overwrite` to replace existing files. `--output-dir` relocates the generated Parquet file. Use `--c4-paths` to select local C4 files, including after changing the downloader's destination. Existing outputs are refused unless `--overwrite` is explicit.

Select the task dataset by name for exp1 and exp3; dataset-prefixed run directories keep its results separate:

```bash
DATASET=task-single-text
RESULTS_ROOT=data/experiment_results
uv run exp1 inference --device cuda --dataset "$DATASET" --results-root "$RESULTS_ROOT"
uv run exp3 inference --device cuda --dataset "$DATASET" --results-root "$RESULTS_ROOT"
```

Omit sample limits to use all rows, or set `--history-count N --future-count N` for leading subsets. exp2 independently samples WikiText and does not use these task/document samples.

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

The submission creates three independently scheduled array tasks (`--array=1-3`), mapping task IDs 1, 2 and 3 to `exp1`, `exp2` and `exp3`, respectively. Each task requests one H100, 4 CPUs, 16 GiB host memory and a maximum of three hours (`03:00:00`). Each runs only its assigned experiment's inference, then its matching analysis; any failed command stops that task, so failed inference skips analysis. exp1/exp3 use all task dataset rows; exp2 samples WikiText directly with the default 512 × 512 counts and `--run-name wikitext-direct`. The script uses default result directories without `--overwrite`: existing outputs cause failure rather than being silently replaced. Do not submit overlapping arrays or other jobs against those same result directories.

The `h100` partition and `gpu:H100:1` GRES spelling must match the cluster configuration. Each task's resource request is not a measured guarantee that its experiment's full dataset will fit in memory or finish within two hours; reduce inference batch size for GPU-memory pressure or adjust the host-memory/time request based on observed usage.

### Sync analysis results from BlueHive

Run locally to pull only `data/experiment_results/<exp>/<run>/analysis/` trees:

```bash
scripts/sync_circ_analysis.sh -n --itemize-changes  # Preview transfers and deletions.
scripts/sync_circ_analysis.sh --max-delete=0 --itemize-changes  # Guard against deletions.
```

The script uses the SSH alias `circ` and remote repository `/scratch/lli81/projects/ur_research/lm-svd`. Override either with `REMOTE` or `REMOTE_ROOT`. It runs from the local repository root regardless of the calling directory.

`--delete` is enabled: local analysis files absent from the remote are removed unless guarded by `--max-delete=0`. Excluded `inference/`, `train/`, and loose files are neither copied nor deleted. Empty experiment/run directories without analysis are pruned from the transfer list. Additional arguments are passed to rsync.

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

## 5. Experiment 2: no-BOS history/future PMI matrix

For each history `h` and future `f`, exp2 stores pointwise mutual information using no BOS token:

```text
PMI[h, f] = log P(h + f) - log P(h) - log P(f)
log P(x) = sum_{t >= 2} log p(x_t | x_<t)
          = 0 for a one-token sequence
```

The first token of each sequence is unscored. Causality makes the history contribution to `log P(h + f)` equal to `log P(h)`, so it cancels: exp2 computes the sum of future-token log probabilities conditioned on the history and preceding future tokens, minus `log P(f)`. It does not calculate `log P(h)` separately. In particular, a singleton future has `log P(f) = 0`, though its conditional score given the history is still included.

These are differences of log-probability **sums**, not averages, probabilities, or centered logits. Rows correspond to histories; columns correspond to futures. Histories and futures may have different token lengths, including within a batch.

### Direct WikiText inference

exp2 reads original WikiText text shards from `data/original/wikitext/wikitext-103-raw-v1/train-*.parquet`, tokenizes them with the local model's tokenizer, and samples windows internally. These are raw source text shards, not prepared history/future Parquet samples. exp2 does not read the training sample store and does not accept `--dataset`, `--stream`, or `--sample-store`.

```bash
uv run exp2 inference --device cuda --run-name wikitext-direct
uv run exp2 analysis --device cuda --run-name wikitext-direct --centering none
```

Defaults: 512 histories × 512 futures, 64-token histories, 32-token futures, seed 0, and independent uniform starts with replacement. Flags `--dataset-dir`, `--config`, `--split`, `--history-count`, `--future-count`, `--history-tokens`, `--future-tokens`, `--seed`, and `--batch-size` control inference. The tokenizer always comes from `--model-path`, with local-only loading. Inputs contain no inserted BOS or separator tokens.

`--pairing adjacent` samples histories with enough room for their following futures, and requires equal history/future counts. The full PMI matrix is computed; diagonal entries are matched continuations. Choose a separate `--run-name` for each configuration, pairing or seed; inference and analysis must use the same run name.

Outputs are `<results-root>/exp2_PFH/<run-name>__<model>/inference/{pmi.npy,metadata.json}`. Metadata records the raw source, tokenizer, pairing, seed and exact window starts. PMI computation and matrix-analysis formulas are unchanged. Adjacent analysis retains pairing diagnostics, `pairing_truncation.png`, and `top5_sparse.png`; analysis discovers pairing from saved metadata.

A small CPU run using local data and model files:

```bash
RESULTS_ROOT=$(mktemp -d /tmp/lm-svd-exp2.XXXXXX)
uv run exp2 inference --device cpu --results-root "$RESULTS_ROOT" \
  --history-count 2 --future-count 2 --batch-size 2
uv run exp2 analysis --device cpu --results-root "$RESULTS_ROOT"
```

On BlueHive, submit `bluehive_run/exp2-inference.sbatch`, then `bluehive_run/exp2-analysis.sbatch` after successful inference. Both accept `RUN_NAME` (default `wikitext-direct`), and analysis runs all four centering strategies. They require the local model, tokenizer and original WikiText training text. Inference and analysis scripts use `--overwrite`.

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

## 7. Understand exp1–exp3 analysis outputs

Experiments exp1–exp3 each own their `analysis` workflow. It reads the matrix and optional sibling `metadata.json` from `<result-dir>/inference/`, then writes five standard outputs to `<result-dir>/analysis/<centering>/` (adjacent exp2 runs additionally write `pairing_truncation.png` and `top5_sparse.png`):

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
    │   ├── metadata.json
    │   └── svd_spectrum.png
    ├── global/  # same five files
    ├── row/     # same five files
    └── deflate1/  # same five files
```

Analysis computes a thin float64 SVD with PyTorch on `--device cuda` by default. CUDA uses the accurate `gesvd` driver; `--device cpu` uses PyTorch's CPU SVD. Invalid devices and unavailable CUDA produce an error rather than silently switching devices. If GPU memory is insufficient, rerun that analysis with `--device cpu`. Saved factors remain NumPy arrays, and the numerical-rank tolerance is unchanged.

`<run>` is `<dataset>__<model>` for exp1, `<run-name>__<model>` for exp2 (default `wikitext-direct__gpt2-small`), or `<dataset>__<model_a>__minus__<model_b>` for exp3. Model labels are the final directory names of the model paths. Only the requested centering directory is created.

### Secondary centering

`analysis --centering {none,global,row,deflate1}` selects how the stored matrix `M` is transformed before SVD; the default is `none`:

| Strategy | Matrix decomposed |
| --- | --- |
| `none` | `M` |
| `global` | `M - mean(M)` |
| `row` | `M - mean(M, axis=1, keepdims=True)` |
| `deflate1` | `M - S[0] * outer(U[:, 0], Vt[0, :])`, where `M = U @ diag(S) @ Vt` |

Row centering subtracts each history's mean across all columns: futures × header tokens for exp1/exp3, or futures for exp2. This is **secondary** centering: it does not change the base vocabulary-mean logit subtraction in exp1/exp3, and exp2 inference still stores uncentered no-BOS PMI values. Selecting only top-k header tokens can leave the stored exp1/exp3 matrix biased toward large values despite its base vocabulary centering.

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

The single `metadata.json` contains six top-level sections in order: `input`, `centering`, `svd`, `rank`, `spectrum`, and `loglog_fit`:

- **Input and centering:** `input` records the matrix path, shape, original dtype, and optional inference metadata verbatim under `inference_metadata` (null if absent). `centering` records the selected `strategy` and `formula`.
- **SVD:** Float64 centering and thin SVD of the selected matrix, regardless of the original stored dtype. `(U * S) @ Vt` reconstructs that matrix, not the original stored matrix unless centering is `none`. The centered matrix is not saved separately. `svd` records factor filenames, decomposition form, computation dtype/device, singular-value count and extrema, the centered Frobenius norm, and relative reconstruction error. Inference and stored matrix dtypes are unchanged.
- **Numerical rank:** Counts singular values strictly greater than `S.max() * max(matrix.shape) * finfo(float64).eps` for all input dtypes and all four centering strategies. `rank` records `value`, `tolerance`, and `tolerance_rule`; the original input dtype is provenance only and does not determine epsilon.
- **Visualization:** `svd_spectrum.png` plots only singular values strictly greater than the numerical-rank tolerance. It contains singular-value decay with logarithmic index and singular-value axes, plus cumulative squared Frobenius energy normalized by the retained values' energy, with linear index and energy axes. Indices start at 1. Rank markers appear in both panels; no tolerance line is drawn. `spectrum` records the plot filename, selection rule, index base, retained singular values, and cumulative energy fractions and formula. Both lists have length `rank.value`; the full spectrum remains in `svd_S.npy`. Plotting uses Matplotlib's headless Agg backend.
- **Log-log fit:** The shared spectrum panel fits exactly the plotted values using `log10(σ_i) = k * log10(i) + b`, with indices starting at 1, and draws a dotted fit line labeled with `k` and `b`. Least squares weights each squared residual by its log-index cell width, `weight_i = log10((i + 0.5) / (i - 0.5))`, balancing contributions across log-index intervals. The top-level `loglog_fit` section records `slope`, `intercept`, `point_count`, `log_base` (10), `formula`, and `weighting`; the CLI prints the coefficients and metadata path. Fewer than two retained values skips the fit, saves null slope/intercept while retaining `weighting`, and omits the fitted line. The fit is not nested inside `spectrum`.

Factor and plot references in metadata are bare filenames relative to the directory containing `metadata.json`; `input.matrix_path` records the inference matrix path as constructed from the result directory. Analysis writes the plot first, then the three factors, and `metadata.json` last as the completion marker. If no singular values exceed the rank tolerance (including an all-zero matrix after centering), analysis raises an error before writing any files or creating the analysis directory.

The shared Python helpers operate on arrays, not saved analysis directories: `compute_svd(matrix, centering=..., device=...)` returns `(svd, summary)` with rank and tolerance on `svd`; `visualize_svd(singular_values, rank_tolerance=..., matrix_shape=..., plot_path=...)` writes the plot and returns `(spectrum, fit)` metadata dictionaries. Each experiment's `analysis.run_analysis` handles input/output and assembles its metadata.

Refresh older analyses, including those that used float32 epsilon for rank, by rerunning the experiment's analysis command with `--overwrite`. This recomputes SVD from the stored inference matrix without loading a model or rerunning inference:

```bash
uv run exp2 analysis --model-path models/gpt2-small \
  --run-name wikitext-direct --results-root data/experiment_results \
  --centering none --overwrite
```

Rerunning analysis does not remove preexisting legacy JSON artifacts. Manually remove obsolete analysis JSON files after refreshing existing exp1–exp3 results, retaining the new `metadata.json`.

### Directory selection and existing results

**Analysis does not infer the model or source from your last inference call.** Pass matching results-root and model flags, plus `--dataset` for exp1/exp3 or `--run-name` for exp2. exp2 defaults to `wikitext-direct`.

Analysis only uses model directory names and stored inference outputs; it does not load the model, tokenizer, raw WikiText, or prepared samples. `--results-root PATH` selects the parent of the experiment directory.

Existing WikiText results have moved to the dataset-prefixed run directories above, and prepared sample files have been renamed to the plain dataset filenames. Historical `metadata.json` files are unchanged: their original `samples_path` values remain records of the inputs used at the time, even though those files have since been renamed.

Inference refuses to replace its existing outputs without `--overwrite`. Analysis prechecks all five outputs for the selected strategy and refuses to start if any exists unless `--overwrite` is supplied; this prevents an existing factor, metadata, or plot file from causing a late overwrite conflict after SVD. `analysis --overwrite` replaces only that strategy's analysis outputs, not inference files or other strategies. When regenerating inference, rerun analysis with `--overwrite` for every previously computed strategy you intend to use; overwriting inference does not refresh analysis.

Other useful options:

- `exp1/exp3 --dataset NAME`: select a named sample set; default `wikitext`.
- `exp2 --run-name NAME`: select an inference/analysis run; default `wikitext-direct`.
- `--results-root PATH`: select the results root for inference or analysis; default `data/experiment_results`.
- `inference --history-count N --future-count N`: exp1/exp3 select leading prepared samples; exp2 independently draws these counts from raw WikiText (default 512 each).
- `exp1 inference --top-k K`: select K header tokens per future; default 50.
- `exp2 inference --pairing {independent,adjacent}`: choose window pairing; default `independent`.
- `exp2 inference --batch-size N`: score N futures per inference batch; default 64.
- `inference --device cpu`: use CPU instead of the default CUDA device.

Use `uv run exp1 <subcommand> --help`, `uv run exp2 <subcommand> --help`, or `uv run exp3 <subcommand> --help` for available options. Module entry points also work: `uv run python -m experiments.exp1_L_M_table <subcommand>`, `uv run python -m experiments.exp2_PFH <subcommand>`, and `uv run python -m experiments.exp3_L_M_diff <subcommand>`.

## Training sample generation and sampling

Both parts live in `data_pipline`. First tokenize local WikiText, then start continuous generation:

```bash
uv run data-pipline build-token-stream
uv run data-pipline sample-store produce --device cuda --worker-id 0
# In another terminal:
uv run data-pipline sample-store status
```

The generator independently draws h and f windows uniformly with replacement (overlap and duplicates are allowed), concatenates their token IDs without BOS or separators, and evaluates the model's last-position prediction. Each sample stores h, f and the **full-vocabulary raw float32 logits**, without softmax, centering or top-k truncation. Defaults are H=64, F=32, 64 paired samples per block, batch size 64, and capacity 64 GiB. This is 64 pairs per block, not a Cartesian history × future grid.

The token-stream builder writes `data/token_streams/wikitext-103-train.npy` and sibling JSON. Build another split using `build-token-stream --split validation --output data/token_streams/wikitext-103-validation.npy`, then use `--stream wikitext-103-validation`. New streams record a tokenizer fingerprint; generation checks it against the local teacher tokenizer. Older stream metadata uses its recorded tokenizer path for the check. Store configuration also records model file sizes and modification times to reject changed checkpoints on restart. All loads are local; neither command downloads files.

The default store is `data/sample_stores/wikitext-103-train__gpt2-small__h64-f32`. Override it with `--store-dir`. Blocks are uncompressed structured `.npy` files supporting memory mapping, with h/f int64 fields and a float32 logits field. Capacity includes block headers; metadata and one temporary staging block per active producer require additional space. All workers share one slot pool: first fill unclaimed empty slots, then refresh the oldest published unclaimed slot. The total capacity is independent of worker count.

`state.json` records each worker's claimed slot and next generation sequence, each published slot's commit order, and the next commit number. A fixed `.state.lock` uses POSIX `fcntl.lockf` only while initializing or reading/modifying/writing this shared state. Model loading and generation happen outside the lock. State snapshots and completed blocks are published by atomic replacement; readers can keep sampling a complete old block during its refresh. If every slot is claimed, producers retry after 0.1 seconds.

```python
from data_pipline import SampleSampler

sampler = SampleSampler(
    "data/sample_stores/wikitext-103-train__gpt2-small__h64-f32", seed=0
)
sampler.wait_until_ready()  # Wait for the first block after the store is initialized.
h, f, logits = sampler.sample()
# CPU tensors: int64[64], int64[32], float32[vocab_size].
```

The sampler uniformly draws from all currently published samples, with replacement. It loads no LLM or tokenizer and needs no original WikiText files. Returned tensors own their memory, so later slot replacement or caller mutation is safe. Sampling an initialized but empty store raises `ValueError`; a missing store also raises an error. A seed reproduces choices only while the store contents remain unchanged.

`--max-gib`, `--samples-per-block`, `--batch-size`, `--history-tokens`, `--future-tokens`, and `--seed` control production. `--max-blocks N` stops after N additional completed blocks in this invocation; otherwise production continues indefinitely. Each process requires a non-negative integer `--worker-id`. New IDs register automatically, so workers can be added without rebuilding the store. Restart with the same ID and store configuration to resume. An unfinished block is regenerated using the same `(seed, worker_id, sequence)`; sequence advances and the claim is released only after the complete block is replaced and its publication is committed. A crash after replacement but before that commit also redoes the same block.

On BlueHive, submit one producer per job; the script defaults the worker ID to `SLURM_JOB_ID`. Requeue retains this ID. A fresh `sbatch` gets a new ID; set `WORKER_ID` to continue a previous worker instead. `MAX_GIB` controls the entire store, and optional `MAX_BLOCKS` limits each invocation for smoke tests:

```bash
sbatch bluehive_run/sample-store-produce.sbatch
sbatch bluehive_run/sample-store-produce.sbatch
# Resume a previous worker in a newly submitted job:
WORKER_ID=12345 sbatch bluehive_run/sample-store-produce.sbatch
```

Run at most one live process per worker ID. There is no duplicate-ID detection, heartbeat, or automatic claim reclamation. Claims survive process exit and await a restart of the same ID; retained worker records do not indicate liveness. `sample-store status` reports each worker's claim and sequence, published blocks/samples, capacity and commit count.

Stop all producers and samplers before deleting or rebuilding a store with `data-pipline sample-store produce --worker-id 0 --overwrite`. This is an offline operation; active-process detection is not performed. The coordination format is `lm-svd-sample-store-v3`; v1 hidden-state stores and v2 sharded stores require an explicit rebuild, with no migration. The token stream and model can be reused, and v3 keeps the v2 `.npy` sample fields unchanged. exp2 does not consume this store.

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

`next_token_logits_batch` takes a non-empty batch of non-empty token-ID sequences with **equal lengths** (no padding); each sequence is validated against the vocabulary and context limit. It returns float32 logits of shape `(batch_size, vocab_size)` **on the inference device**, unlike the CPU return from `next_token_logits`. Unequal lengths raise `ValueError`.

`final_hidden_states(input_ids)` accepts a non-empty 2-D integer tensor of equal-length token-ID sequences, validates vocabulary/context bounds, and returns float32 `(batch_size, sequence_length, hidden_size)` states on the inference device under `torch.no_grad()`. For GPT-2 these are post-final-LayerNorm states; multiplying them by the unembedding transpose reproduces logits without materializing full-sequence logits inside the method.

## Tests

```bash
uv run python -m unittest discover -s tests
```

The suite uses tiny random GPT-2 fixtures and synthetic matrices; no downloaded models or datasets are needed.
