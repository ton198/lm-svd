# LM-SVD: GPT-2 Matrix Experiments

Build matrices from local GPT-2 inference, compute their singular value decompositions (SVDs), measure numerical rank, and plot singular-value decay and cumulative energy.

Experiments use sampled history/future token sequences but construct different matrices:

| Experiment | Matrix entry | Matrix shape | Matrix file |
| --- | --- | --- | --- |
| `exp1` | Centered next-token logit for a future-specific header token | histories × (futures × top-k) | `matrix.npy` |
| `exp2` | No-BOS pointwise mutual information of history and future | histories × futures | `pmi.npy` |
| `exp3` | Difference between two centered-logit matrices using model A's future-specific headers | histories × (futures × top-k) | `matrix.npy` |
| `exp4` | Same entry as `exp1`, in a training table and a disjoint validation table | histories × (futures × top-k), twice | `train/matrix.npy`, `val/matrix.npy` |
| `exp5` | Same entry as `exp1` minus the header token's centered logit after the future alone, one (history, future) pair at a time | no stored matrix; evaluated on the `exp4` tables | — |
| `exp6` | Same entry and pairs as `exp5`; the right network is bilinear in the header token's unembedding row | no stored matrix; evaluated on the `exp4` tables | — |
| `exp7` | Same entry and pairs as `exp5`; one independent pair of sub-networks per mode, trained one mode at a time | no stored matrix; evaluated on the `exp4` tables | — |

Experiments **exp1–exp3** have two subcommands, run in order: `inference` builds the matrix, and `analysis` runs SVD → numerical rank → visualization together with a selectable secondary centering strategy. Once a matrix exists, their analysis does not load a language model. Their analysis defaults to CUDA; use `--device cpu` to run without a GPU. **exp4** instead runs `inference` → `train`: it builds two tables with their exact SVDs, trains NeuralSVD on the training table, and compares the learned modes with both exact SVDs during training. **exp5** has only `train`: it trains two MLPs on frozen GPT-2 hidden states with pairs from the sample store and compares them with the exact SVDs of the exp4 tables. **exp6** also has only `train`: the same pairs and target as exp5, with a trainable copy of GPT-2 on each side and a right side that is bilinear in the header token's unembedding row. **exp7** also has only `train`: the same pairs and target, with its own pair of sub-networks for every mode, trained and frozen one mode at a time.

## Navigation

- [Install dependencies](#1-install-dependencies) · [Download models](#2-download-models-for-offline-inference) · [Prepare data](#3-prepare-shared-historyfuture-samples) · [Sample generation and examples](docs/samples.md)
- [exp1](#4-experiment-1-centered-logit-matrix) · [exp2](#5-experiment-2-no-bos-historyfuture-pmi-matrix) · [exp3](#6-experiment-3-difference-between-centered-logit-matrices) · [exp1–exp3 analysis](#7-understand-exp1exp3-analysis-outputs) · [exp4](#experiment-4-neuralsvd-on-an-explicit-table) · [exp5](#experiment-5-gpt-2-hidden-states-and-mlps-on-the-l_m-table) · [exp6](#experiment-6-a-right-side-bilinear-in-the-header-token) · [exp7](#experiment-7-an-independent-sub-network-per-singular-function)
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

# RoBERTa-base for exp7's --subnet roberta; only downloaded when named.
uv run python scripts/download_models.py --models roberta
```

The script downloads **complete inference files**, not every file in the upstream repositories. Each GPT-2 directory contains `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and `merges.txt`. The RoBERTa directory contains `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and `merges.txt`. These include the full model weights, not just tokenizers. Duplicate PyTorch, TensorFlow, Flax, Rust, and ONNX weight formats are not downloaded.

| Local directory | Hugging Face repository | Pinned revision |
| --- | --- | --- |
| `models/gpt2-small` | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| `models/gpt2-medium` | [openai-community/gpt2-medium](https://huggingface.co/openai-community/gpt2-medium) | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` |
| `models/gpt2-large` | [openai-community/gpt2-large](https://huggingface.co/openai-community/gpt2-large) | `32b71b12589c2f8d625668d2335a01cac3249519` |
| `models/roberta-base` | [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` |

Re-running the script reuses files already downloaded at the pinned revisions. `--output-dir PATH` changes the parent directory of the model directories; by default it is the repository's `models/` directory. Without `--models` the script downloads the three GPT-2 models; `--models roberta` downloads RoBERTa-base (about 500 MB, used only by exp7). Internet access is required to fetch missing files. Download the source dataset in the next section before enabling offline mode.

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

## Experiment 4: NeuralSVD on an explicit table

exp4 tests whether NeuralSVD training can fit one explicit table, and whether what it learns carries over to a second table it never sees. Both tables use the exp1 entry and the exp1 column layout (top-k header tokens per future). Their exact SVDs are computed first and serve as the reference.

```bash
uv run exp4 inference   # build both tables and their exact SVDs
uv run exp4 train       # train on the training table, compare every 1000 steps
```

**`inference`** reads `wikitext-large` by default. The training table takes the first 2048 histories and 1024 futures; the validation table takes the next 1024 histories and 512 futures, so the two tables share no history and no future. With top-k 50 the shapes are `2048 × 51200` and `1024 × 25600`, which costs about 2.6 million language-model forward passes. The SVD is a float64 thin SVD of the stored matrix with no secondary centering. Outputs under `<results-root>/exp4_table_neural_svd/<dataset>__<model>/inference/`:

| File | Content |
| --- | --- |
| `{train,val}/matrix.npy` | The table, float32 |
| `{train,val}/tokens.npz` | `histories`, `futures`, `header_token_ids`, and the sample IDs of both |
| `{train,val}/svd.npz` | All singular values `S`, and the leading 256 singular vectors `U`, `Vt` |
| `metadata.json` | Settings and table shapes; written last, so it marks a finished build |

**`train`** uses the `svd_inference` network (`get_nested_lora_network`). Each step draws 32 distinct rows and 512 distinct columns of the training table and feeds the full `32 × 512` sub-table between them to the NestedLoRA loss. Every 1000 steps it evaluates the learned features of every row and column of both tables against that table's exact SVD, for ranks 1, 2, 4, …, 128:

| Printed column | Meaning |
| --- | --- |
| `sv_rel_err` | Learned singular value of mode `rank` over the exact one, minus 1. The learned value is the product of that mode's left and right feature norms. |
| `left_overlap`, `right_overlap` | `‖U_kᵀ Q_k‖²_F / k`, where `Q_k` is an orthonormal basis of the first `k` learned features. 1 means the same subspace; a random subspace scores about `k / rows` (or `k / columns`). |
| `energy` | `1 − ‖A − L_k R_kᵀ‖² / ‖A‖²` for the first `k` learned modes |
| `exact_energy` | The same quantity for the exact rank-`k` truncation, which is the upper bound of `energy` |

The first mode holds almost all of the energy, so compare `energy` with `exact_energy` rather than with 1.

Outputs under `.../train/`: `eval_history.jsonl` (one line per evaluation, with the `train` and `val` comparisons), `checkpoint.pt` and `metadata.json`. Running `train` again with the same settings continues from the checkpoint up to `--steps`; `--overwrite` starts over. Batch sizes, rank and learning rate are set with `--history-batch`, `--column-batch`, `--neigs` and `--lr`.

`train --centering column-mean` subtracts from every column its mean over the table's own histories, in both tables, and recomputes their exact SVDs before training. The leading mode of the stored table is close to a constant over histories times the column means and holds over 99% of the energy; this option removes it so that training and the comparison cover the modes after it. Results go to `.../train-column-mean/`, next to the uncentered run. The validation table is centered with its own column means, which a model would not have for new futures, so this option is a diagnostic rather than a deployable target.

`train --switch-every STEPS` trains on a sequence of fresh tables instead of the stored training table. Every `STEPS` steps it builds a new table of `--table-histories` × `--table-futures` (default 512 × 256) with the language model, from distinct random histories and futures of the sample file that are not in the validation table, and applies `--centering` to it. The two stored tables are still evaluated every 1000 steps, so the `val` comparison shows whether the modes carry over to unseen histories and futures. The `train` comparison then covers a table whose histories and futures can appear in the fresh tables. Each evaluation also records `table_index`. Table `i` depends only on `--seed` and `i`, so a resumed run rebuilds the table it stopped in. Results go to `.../train[-column-mean]-switch-<histories>x<futures>-every<STEPS>/`. A 512 × 256 table costs about 131 thousand language-model forward passes.

On BlueHive, submit `bluehive_run/exp4-inference.sbatch`, then `bluehive_run/exp4-train.sbatch`. Arguments after the script name are passed to the command.

Small smoke run:

```bash
RESULTS_ROOT=$(mktemp -d /tmp/lm-svd-exp4.XXXXXX)
uv run exp4 inference --results-root "$RESULTS_ROOT" --batch-size 256 \
  --train-histories 128 --train-futures 32 --val-histories 64 --val-futures 16
uv run exp4 train --results-root "$RESULTS_ROOT" --steps 400 --eval-every 200
```

## Experiment 5: GPT-2 hidden states and MLPs on the L_M table

exp5 tests whether a network that reads only frozen GPT-2 hidden states can learn the modes of the exp1 table, with the right side producing every header column of a future in one pass.

```text
left:   history -> frozen GPT-2 -> hidden state at the last token -> 2-layer MLP -> (rank,)
right:  future  -> frozen GPT-2 -> hidden state at the last token -> 2-layer MLP -> (top-k, rank)

A[history, (future, j)] ≈ g(future, j) + left(history) · right(future)[j]
```

The hidden state is the final layer after `ln_f`. Each side passes through GPT-2 on its own; the networks never see `history + future`. Slot `j` of the right output is the future's `j`-th header token (descending future-only logit, as in exp1). The right MLP is not given the header token ids. Only the two MLPs are trained. Each is `LayerNorm → Linear → GELU → Linear`, and `--mlp-width` is the size of the middle layer.

### The fixed first mode

`g(future, j)` is the centered logit of header token `j` after the future alone. exp5 fixes the first mode to `1 × g` and trains the NestedLoRA loss on the rest, `A − g`. `g` comes from the same GPT-2 pass that gives the right network its input, so it is also available for futures the network has never seen. On the exp1 table (`wikitext`, GPT-2 Small, 512 × 25,600):

| Removed from the table | Energy left | Share of the largest remaining mode | σ₁/σ₂, about |
| --- | ---: | ---: | ---: |
| nothing | 100% | 99.6% | 46 |
| `g` (exp5) | 1.22% | 67% | 4 |
| each column's mean over the histories | 0.42% | 11% | 1.2 |

`g` and the column means correlate at 0.93. Their difference is still constant over the histories and remains as the first mode of what exp5 trains on.

### Run

exp5 needs the exp4 tables and a sample store that holds published blocks from the same model (see [Training sample generation and sampling](#training-sample-generation-and-sampling)). A producer may keep running during training.

```bash
uv run exp4 inference                                              # the evaluation tables
uv run data-pipline sample-store produce --device cuda --worker-id 0   # in another terminal
uv run exp5 train
```

Each step draws `--batch-size` (default 256) independent (history, future) pairs from the store. For each pair it takes the stored logits of `history + future` on the future's header tokens, centers them over the vocabulary and subtracts `g`. The loss uses those entries for its operator term and every history and column of the batch for its metric term. In expectation this is the loss of a full history × future grid, because the store draws histories and futures independently.

Every `--eval-every` steps the learned features of both exp4 tables are compared with the exact SVD of that table minus `g`, in the same columns as exp4 (`sv_rel_err`, `left_overlap`, `right_overlap`, `energy`, `exact_energy`). The header tokens stored by exp4 are kept: a batched pass can order near-tied tokens differently, so exp5 only checks that they are among the model's top-k.

Outputs under `<results-root>/exp5_gpt2_mlp_table/<dataset>__<model>/train/`: `eval_history.jsonl` (one line per evaluation with `train_loss`, `kernel_energy`, `seconds_per_step` and the `train` and `val` comparisons), `checkpoint.pt` (the latest trained weights, without the frozen GPT-2), `checkpoint-best.pt` (the weights of the evaluation with the highest validation `energy` at the largest rank) and `metadata.json`, which also records how much energy `g` leaves in each table. `--dataset` and `--model-path` select the exp4 run to evaluate on; `--store-dir` selects the store. Rank, MLP width and learning rate are set with `--neigs`, `--mlp-width` and `--lr`.

Things to keep in mind when reading the numbers:

- **They are not comparable with exp4's.** exp5 is measured on the tables minus `g`; exp4's `train/` and `train-column-mean/` runs are measured on the stored tables and on the tables minus their column means. Compare each run's `energy` with its own `exact_energy`.
- **Both tables are held out to the same degree.** Store pairs come from the whole training stream and are not excluded from the tables; a pair overlaps a validation window with a probability of about 0.1%. The `train` table is no longer "seen" data, it is just the larger table.
- **A loss far below `−E[K²]` means the store is too small or too stale.** `kernel_energy` is `E[K²]`, and a fit of the table ends near `−E[K²]`. When the same pairs are drawn many times, the network can instead fit which history came with which future, and the loss drops well below that while the `energy` of both tables falls. An evaluation whose loss is below `−1.2 × E[K²]` prints a warning and gets `"memorizing": true` in `eval_history.jsonl`; training continues, and `checkpoint-best.pt` keeps the weights from before. Keep a producer running so the store is refreshed.
- **A continued run draws other pairs.** Running `train` again continues from the checkpoint up to `--steps` (`--overwrite` starts over), but the store changes while producers run, so the pairs are not reproduced.

### Training the right side's GPT-2

`train --unfreeze-right` gives the right side its own encoder: a copy of the model's transformer that is trained together with the two MLPs, with learning rate `--encoder-lr` (default 2e-5) and dropout off. The right MLP then reads the last-token hidden state of that copy instead of the frozen model's. The frozen model is still used for the header tokens, `g` and the history hidden states, so the training target and the evaluation tables do not change. Results go to `.../train-unfreeze-right/`, and its checkpoints also hold the encoder (about 0.5 GB each for GPT-2 Small).

The reason for this option is a probe on the exp4 training table (`scripts/probe_right_function.py`): with the exact left basis, regressing each column's exact coordinates on frozen future-only hidden states reaches about half of the attainable energy on held-out futures (0.47–0.50 of 0.94 at rank 128, with the header token's embedding given as an input), and more training futures do not raise it.

### Training a copy of GPT-2 on each side

`train --finetune` replaces both MLPs: each side is its own trainable copy of the model's transformer (learning rate `--encoder-lr`, dropout off) with a `LayerNorm → Linear` head on its last hidden state. The left copy reads the history. The right copy reads a column as its future followed by the header token, so the token is part of the input; the header tokens of one future share one sequence, in which each sees the future and itself only and sits at the position after the future. `--mlp-width` is unused. The frozen model still gives the header tokens and `g`. Results go to `.../train-finetune/`. A step with `--batch-size 128` needs about 23 GB of GPU memory.

On BlueHive, submit `bluehive_run/exp5-train.sbatch` after `exp4-inference.sbatch`, with `sample-store-produce.sbatch` running or its store already filled. Arguments after the script name are passed to the command.

Small smoke run on CPU, after the commands above have produced the tables and at least one block:

```bash
RESULTS_ROOT=$(mktemp -d /tmp/lm-svd-exp5.XXXXXX)
mkdir -p "$RESULTS_ROOT/exp4_table_neural_svd"
ln -s "$PWD/data/experiment_results/exp4_table_neural_svd/wikitext-large__gpt2-small" \
  "$RESULTS_ROOT/exp4_table_neural_svd/"
uv run exp5 train --results-root "$RESULTS_ROOT" --device cpu \
  --steps 20 --eval-every 10 --neigs 8 --mlp-width 64 --batch-size 16
```

## Experiment 6: a right side bilinear in the header token

exp6 keeps exp5's pairs, target, loss and evaluation and changes the network. It follows from what the target is. With `s(x)` the final hidden state after `x` and `e_z` the unembedding row of token `z` minus the mean row,

```text
A[history, (future, z)] − g(future, z) = ⟨s(history + future) − s(future), e_z⟩
```

so every right singular function of exp5's target has the form `⟨w_l(future), e_z⟩`: a vector that depends on the future, meeting the header token only through an inner product with its unembedding row. exp5's right networks do not have that form. The MLPs are not told which token a slot holds, and `--finetune` reads the token as one more input token and puts a linear head on the result. In the `--finetune` run the energy at rank 128 stayed next to the energy at rank 1 (validation table, step 15,000: 0.433 and 0.448, where the exact SVD has 0.654 and 0.946).

```text
left:   history -> trainable GPT-2 copy -> last hidden state -> LayerNorm -> Linear -> (rank,)
right:  future  -> trainable GPT-2 copy -> last hidden state -> LayerNorm -> Linear -> B(future), (rank, d)
        header token z -> e_z -> P e_z, (d,)
        right(future, z)[l] = ⟨B_l(future), P e_z⟩ / sqrt(d)
```

A head that gave `w_l(future)` directly would need rank × hidden outputs (75 M weights for GPT-2 Small at rank 128). It is factorised as `w_l(future) = B_l(future) P`: `P` is one trained `d × hidden` projection of the unembedding rows, shared by all modes and futures, and the head only gives the `rank × d` matrix `B(future)`. `d` is `--head-rank` (default 128, 12.8 M weights). A future has top-k header tokens, so any `d ≥ top-k` can still reach every value on them; what `d` limits is how `B(future)` may vary with the future. `e_z` comes from the frozen model, divided by the root mean square of all its entries, and is neither trained nor saved. The right copy reads the future alone, so a step is lighter than exp5's `--finetune`: about 16 GB of GPU memory at the default `--batch-size 128` (extrapolated from smaller batches).

```bash
uv run exp6 train
```

It needs what exp5 needs: the exp4 tables and a sample store from the same model. The heads and `P` use `--lr` (default 3e-4), the two copies `--encoder-lr` (default 2e-5). Outputs go to `<results-root>/exp6_gpt2_bilinear_table/<dataset>__<model>/train/`, with the same files and the same columns as exp5, so its numbers compare directly with exp5's (both are measured on the tables minus `g`). On BlueHive, submit `bluehive_run/exp6-train.sbatch`.

## Experiment 7: an independent sub-network per singular function

exp7 keeps exp5's and exp6's pairs, target and evaluation and changes how the modes are learned. In exp5 and exp6 every mode is one output of a shared network (one GPT-2 copy per side, the 128 modes its 128 outputs), and all modes are trained together with joint nesting. On both exp4 tables (minus `g`) their energy stays at 0.47–0.51 at any rank, where the exact SVD has 0.65 at rank 1 and 0.95 at rank 128. exp6's record shows signs of the modes interfering: on the report table (below) the energy at rank 1 was 0.470 at step 50,000 and 0.393 at step 100,000, while the energy at rank 128 stayed at 0.47–0.50.

What exp7 changes:

- **One pair of sub-networks per mode.** Mode `l` has its own left encoder and head, which read the history, and its own right encoder and head, which read the future. No trainable parameter is shared between the two sides or between two modes; `L` modes have `2L` encoders. The frozen model's unembedding is a constant shared by all modes and is not trained.
- **Stages.** Stage `l` trains only mode `l`, on each pair's target minus the output of the frozen modes `1..l−1` on the same pair, then freezes it. Only one pair of sub-networks has gradients at a time.
- **No shared projection.** Each mode's right head gives the full `E`-dimensional vector `w_l(future)`, so exp6's projection `P` is gone.

This is a staged form of the sequential nesting of the NeuralSVD paper (arXiv:2402.03655), in which the gradient of mode `l` is a rank-one fit of the residual after the earlier modes, and the earlier modes receive no gradient from it. The paper trains all modes at once with gradient masks (`utils.nestedlora.NestedLoRA(sequential=True)` is that code). exp7 freezes instead, for two reasons: all sub-networks with gradients do not fit on one GPU, and subtracting the frozen modes' output on the same pair cancels the noise those modes bring, which the masked form cannot do.

### One mode

```text
left u_l:   history (64 tokens) -> encoder_l^L -> pooled vector
                                -> LayerNorm (no affine) -> Linear(·, 1) -> a number

right v_l:  future (32 tokens)  -> encoder_l^R -> pooled vector
                                -> LayerNorm (no affine) -> Linear(·, E) -> w_l(future), (E,)
            v_l(future, z) = ⟨w_l(future), e_z⟩ / sqrt(E)
```

`E` is the width of the frozen model's unembedding (768 for GPT-2 Small). `e_z` is the unembedding row of token `z` minus the mean row, divided by the root mean square of all entries, as in exp6; it is neither trained nor saved. The right side has this form because the target itself is `⟨s(history + future) − s(future), e_z − ē⟩` (see [exp6](#experiment-6-a-right-side-bilinear-in-the-header-token)), so every right singular function is the inner product of a vector that depends only on the future with `e_z`.

`--subnet` selects the encoders. Every encoder turns a sequence of tokens into one vector, the pooled vector. Sequences have a fixed length (history 64, future 32) and no padding, so no attention mask is needed.

| `--subnet` | Each encoder | Pooled vector | Learning rates |
| --- | --- | --- | --- |
| `scratch` (default) | A randomly initialised BERT transformer (`transformers.BertModel` without its pooler), bidirectional, trained from scratch, dropout 0, with its own token and position embeddings; it reads the model's token ids directly; `--scratch-layers` 6, `--scratch-width` 384, `--scratch-heads` 6 by default | The mean of the last hidden state over all positions | everything `--lr` (default 3e-4), with a linear warmup over the first 1000 steps of each stage |
| `roberta` | A trainable copy of the pretrained RoBERTa-base in `--roberta-path` (default `models/roberta-base`; `transformers.RobertaModel` without its pooler), bidirectional, dropout 0. The model's GPT-2 token ids are mapped to RoBERTa's ids by token string, and `<s>` is put in front of the sequence and `</s>` behind it | The mean of the last hidden state over all positions, `<s>` and `</s>` included | encoders `--encoder-lr` (default 2e-5), heads `--lr` (default 3e-4), no warmup |
| `gpt2` (kept, not used in the planned runs) | A trainable copy of the model's transformer, causal, dropout off: exp6's encoder, so only "independent or shared" changes | The hidden state of the last position | encoders `--encoder-lr` (default 2e-5), heads `--lr` (default 3e-4) |

Each stage starts a new Adam optimizer.

Why RoBERTa and not the original BERT: RoBERTa uses the same byte-level BPE as GPT-2. All 50257 GPT-2 tokens are in RoBERTa's vocabulary of 50265, under other ids, so a table from GPT-2 id to RoBERTa id built from the token strings is all it takes, and no text is decoded and tokenized again. BERT uses WordPiece and would need the text tokenized again, and a history often ends in the middle of a word. A GPT-2 token missing from RoBERTa's vocabulary, or a missing `--roberta-path` directory, is an error before anything is written. RoBERTa-base has 514 position embeddings, of which a sequence without padding can use 512: 64 history tokens or 32 future tokens plus `<s>` and `</s>` fit, and a longer sequence is an error before training.

### One stage

A step of stage `l` draws `--batch-size` (default 128) pairs, computes their target as exp5 does, runs the frozen modes without gradient, and fits the current mode to the residual `r = target − Σ_{i<l} f_i g_i` with the rank-one paired loss

```text
loss = −2 · mean_b[ f_b · mean_j(r_bj · g_bj) ] + mean_b(f_b²) · mean_{b,j}(g_bj²)
```

which is exp5's `paired_loss` at `neigs=1`. Only the current mode's parameters are updated.

A stage evaluates every `--eval-every` (default 1000) of its steps and always at its last step. The **selection metric** is the captured energy at rank `l` of the `l` modes (the frozen ones and the current one, not aligned) on the selection table. The stage remembers the current mode's weights at the evaluation with the highest selection metric, and ends when either

- it has run `--stage-steps` (default 30000) steps, or
- it has run at least `--min-stage-steps` (default 5000) steps and the selection metric has not reached a new best for `--patience` (default 5) evaluations in a row.

It then loads those best weights, freezes the mode and saves it to `modes/mode-XX.pt`, aligns all `l` modes, and the next stage starts. `--patience 0` is a fixed number of steps: the stage runs `--stage-steps` steps and freezes the weights of its last step, with no selection. `--min-stage-steps` may not exceed `--stage-steps`.

### The two tables

exp4's directory names `train` and `val` are misleading here, so exp7's logs, records and this section call them:

| exp7 name (record key) | exp4 table | Shape | Role |
| --- | --- | --- | --- |
| **selection table** (`selection`) | `train` | 2048 × 51200 | Decides when a stage ends and which weights are frozen |
| **report table** (`report`) | `val` | 1024 × 25600 | Reported only |

exp7 trains on neither table; both are held out to the same degree (see [exp5](#experiment-5-gpt-2-hidden-states-and-mlps-on-the-l_m-table)). Both are the exp4 tables minus `g`, each with its exact SVD, as in exp5 and exp6. Only the selection table decides when a mode is frozen. exp4's files and directories keep their names.

### The alignment

The learned functions are not orthogonal and have no common scale. At the end of every stage, stage 1 included (where it only rescales), exp7 draws `--align-pairs` (default 16384) fresh pairs from the store, runs all `l` frozen modes forward only, and estimates three `l × l` matrices:

```text
Λ_f = E[f fᵀ]        Λ_g = E[g gᵀ] (over all (future, z) columns)        C_ij = E[f_i(h) · target(h, f, z) · g_j(f, z)]

M = Λ_f^{-1/2} C Λ_g^{-1/2} = P S Qᵀ        T_f = Λ_f^{-1/2} P S^{1/2}        T_g = Λ_g^{-1/2} Q S^{1/2}
```

`C` uses the target, not the residual. The aligned features `F T_f` and `G T_g` are orthogonal and ordered by `S`, largest first, and their product is the least-squares fit of the target within the span of the learned functions. **`aligned`** in `stages.jsonl` is the comparison of these aligned features with the exact SVDs. The alignment only recombines what has been learned: it uses only store pairs, not the evaluation tables, changes no weights, and does not affect later stages, whose residuals use the unaligned frozen modes.

### Run

exp7 needs what exp5 and exp6 need: the exp4 tables and a sample store from the same model. The first step runs stage 1 alone, once with `scratch` and once with `roberta`:

```bash
uv run python scripts/download_models.py --models roberta   # once per machine
uv run exp7 train --subnet scratch --modes 1
uv run exp7 train --subnet roberta --modes 1
```

Then, with the kind chosen from the table in [How to read the first stage](#how-to-read-the-first-stage), the same command with a larger `--modes` continues that kind's directory:

```bash
uv run exp7 train --subnet scratch --modes 8     # continues train-scratch
```

`--subnet scratch` is the default. `--roberta-path` (default `models/roberta-base`) names the local RoBERTa of `--subnet roberta`; nothing is downloaded during training. It is one of the settings a continued run must repeat (see below), for every kind.

Outputs go to `<results-root>/exp7_independent_subnets/<dataset>__<model>/train-<subnet>/`, so each kind has its own directory. On BlueHive, submit `bluehive_run/exp7-train.sbatch`; arguments after the script name are passed to the command, and resubmitting continues the current stage.

Running the same command again continues: the frozen modes are loaded from `modes/`, and an interrupted stage continues at its last evaluation with its best weights and its count of evaluations without a new best. The sampler is reseeded from `--seed` and the global step, so a continued run draws other pairs. Between runs of one directory:

- **Only `--modes` may change** (and `--device`). Any other changed setting (`--stage-steps`, `--patience`, `--lr`, `--store-dir`, `--roberta-path`, …) is an error that names `--overwrite`, which deletes the outputs and starts over.
- A `--modes` below the number of finished modes is an error. Equal to it, with no stage in progress, the command returns at once and writes nothing. Equal to it while a later stage is in progress is an error asking for a larger `--modes` or `--overwrite`.
- **A finished stage cannot be lengthened**, only redone with `--overwrite`.

### Outputs

| File | Content |
| --- | --- |
| `eval_history.jsonl` | One line per evaluation, over all stages |
| `stages.jsonl` | One line per finished stage |
| `modes/mode-XX.pt` | The frozen mode `XX` (two encoders and two heads): `state_dict`, `stage`, `best_step` |
| `alignment.pt` | The latest alignment: `T_f`, `T_g`, `S` and `modes`, the number of modes it covers |
| `checkpoint.pt` | `config` (the settings a continued run must repeat), `completed_modes`, the global `step`, the evaluation `history`, and `stage`: empty between stages, otherwise the stage step, the current mode's weights, its best weights, the optimizer, the warmup schedule of a `scratch` mode and the state of the stopping rule (best value, its step, evaluations since) |
| `metadata.json` | All settings, `modes`, `warmup_steps`, descriptions of the `encoder` and of the two sides, and exp5's run metadata; `trainable_parameters` counts one mode, `tables` and `table_shapes` use the keys `selection` and `report`, and `ranks` reads `1..l at stage l` |

`eval_history.jsonl`:

| Field | Meaning |
| --- | --- |
| `step` | Global step, counted over all stages |
| `stage`, `stage_step` | The stage and its own step count |
| `train_loss` | Mean stage loss since the previous evaluation |
| `kernel_energy` | Mean square of the target, `E[K²]`, since the previous evaluation |
| `residual_energy` | Mean square of the residual `r` the current mode is fitted to, since the previous evaluation; equal to `kernel_energy` in stage 1 |
| `seconds_per_step` | Mean time of a step since the previous evaluation |
| `selection`, `report` | Comparison of the `l` modes (frozen and current, not aligned) with that table's exact SVD: `singular_values`, `singular_value_relative_error`, `ranks`, `left_subspace_overlap`, `right_subspace_overlap`, `captured_energy`, `exact_captured_energy` (exp4's columns) |

`stages.jsonl`:

| Field | Meaning |
| --- | --- |
| `stage` | The stage |
| `steps` | Steps the stage ran |
| `best_step` | The stage step whose weights were frozen |
| `reason` | `steps` (reached `--stage-steps`) or `patience` |
| `selection_energy` | The selection metric of the frozen weights |
| `selection`, `report` | Comparison of the frozen modes `1..l`, not aligned |
| `aligned` | `selection` and `report` comparisons of the aligned modes |
| `singular_values` | `S` of the alignment |
| `left_correlation`, `right_correlation` | Largest absolute off-diagonal entry of `Λ_f` and `Λ_g` as a correlation: how far from orthogonal the learned modes are |

Things to keep in mind when reading the records:

- **Ranks are listed one by one, 1..l**, not 1, 2, 4, 8, … as in exp5 and exp6, so that every stage has a value at its own rank `l`.
- **The numbers compare directly with exp5's and exp6's**: all three are measured on the tables minus `g`.
- **Two kinds of singular values.** `singular_values` in `stages.jsonl` is `S` of the alignment. Its moments are means over pairs and columns, so `S` is per entry. The `singular_values` inside a table comparison (`selection`, `report`, `aligned`) are norms over the whole table, so they are larger by `sqrt(N·C)` of that table (`N` rows, `C` columns).
- **Two kinds of `residual_energy`.** In `metadata.json`, `tables.*.residual_energy` is exp5's energy of the table left after removing `g`. The `residual_energy` field of `eval_history.jsonl` is something else: the mean square of the stage's residual target on the training pairs.
- **An aligned rank `k < l` uses all `l` pairs of networks**, because each aligned function is a combination of all the learned ones. It is therefore not comparable with exp6's rank `k`, which uses `k` outputs.
- **`aligned` can be slightly below the unaligned value.** The alignment is estimated from `--align-pairs` sampled pairs, not from the table.

### How to read the first stage

Reference values on the report table (exp4's `val` table, minus `g`): the shared networks of exp5 and exp6 reach 0.45–0.47 at rank 1 and at best 0.51 at any rank; the exact SVD has 0.654 at rank 1.

Stage 1 is run once with `scratch` and once with `roberta` (see [Run](#run)). Read the table below on the **unaligned** `report` captured energy at rank 1, from `eval_history.jsonl` or `stages.jsonl`. With one mode the alignment only rescales, so `aligned` should agree up to estimation noise.

| Stage 1 result (report table, energy at rank 1) | Meaning | Next step |
| --- | --- | --- |
| Above 0.51 | One independent sub-network learning the first mode alone already exceeds what the shared network reaches with 128 modes. Independence works | Continue in the same directory with the winning kind of sub-network and a larger `--modes` |
| 0.47–0.51 | Independence removed the interference, but the gain is small | Still continue to 8 modes and see whether the total energy exceeds 0.51 |
| Below 0.47 | The bottleneck is not the sharing but predicting GPT-2's response to unseen futures from the future alone | Stop and reconsider the input of the right function (for example anchor histories) |
| `scratch` far below `roberta` | Pretraining is necessary | Use only `roberta` from then on |
| `scratch` close to or above `roberta` | A small network is enough and more modes are affordable | Use `scratch` from then on, with a larger `--modes` |
| Both below 0.47 | A bidirectional encoder with pooling did not solve the problem | Run stage 1 once with the kept `--subnet gpt2` for comparison, then reconsider |

With several stages, look also at the difference between the energy before and after the alignment (a large difference means later modes are repairing the errors of earlier ones), and at the energy each stage adds in `stages.jsonl` compared with what the exact SVD adds at that rank.

### Resources (estimated, not measured)

- `gpt2`: each encoder has about 125 million parameters. A step needs the same gradient memory as exp6 (about 16 GB at batch 128). Each frozen mode adds about 1 GB of weights and two forward-only GPT-2 passes, so stage 8 needs about 23 GB and a step takes about two to three times as long as in stage 1.
- `roberta`: each encoder also has about 125 million parameters, so its memory, time and disk are expected to be close to `gpt2`'s.
- `scratch` (default size, a 6-layer BERT of width 384): each encoder has about 30 million parameters, about 19 million of them in the token embedding of GPT-2's 50257 tokens.
- Disk: every evaluation rewrites the whole checkpoint (the current mode, its best weights and the optimizer state). For `gpt2`, a run directory with 8 modes is estimated at about 16 GB, including the checkpoint's temporary file. This is an estimate, not a measurement.

### Small smoke run

On CPU, after the exp4 tables and at least one store block exist:

```bash
RESULTS_ROOT=$(mktemp -d /tmp/lm-svd-exp7.XXXXXX)
mkdir -p "$RESULTS_ROOT/exp4_table_neural_svd"
ln -s "$PWD/data/experiment_results/exp4_table_neural_svd/wikitext-large__gpt2-small" \
  "$RESULTS_ROOT/exp4_table_neural_svd/"
uv run exp7 train --results-root "$RESULTS_ROOT" --device cpu --subnet scratch \
  --modes 1 --stage-steps 20 --min-stage-steps 0 --patience 0 --eval-every 10 \
  --align-pairs 64 --scratch-layers 1 --scratch-width 64 --scratch-heads 2 --batch-size 16
# The same command with --modes 2 adds a second stage in the same directory.
```

## Training sample generation and sampling

Sample generation, storage, coordination and CLI code live in the `data_pipline.sample_store` package (`producer.py`, `store.py`, `state.py` and `cli.py`). First tokenize local WikiText, then start continuous generation:

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

The sampler uniformly selects 100 distinct published blocks (or all published blocks if fewer are available), reads them into one RAM pool, and samples rows uniformly with replacement. It reloads the pool after as many draws as the pool contains rows. With the defaults, 100 blocks contain 6400 rows and use about 1.2 GiB per sampler; one state read and 100 block reads serve those 6400 draws. The old pool is released before allocating its replacement. On an unchanged store each row still has uniform marginal probability, but consecutive samples share a pool; producer updates are seen at the next pool load. It loads no LLM or tokenizer and needs no original WikiText files. Returned tensors own their memory, so later slot replacement or caller mutation is safe. Sampling an initialized but empty store raises `ValueError`; a missing store also raises an error. A seed reproduces choices only while the store contents remain unchanged.

`--max-gib`, `--samples-per-block`, `--batch-size`, `--history-tokens`, `--future-tokens`, and `--seed` control production. `--max-blocks N` stops after N additional completed blocks in this invocation; otherwise production continues indefinitely. Each process requires a non-negative integer `--worker-id`. New IDs register automatically, so workers can be added without rebuilding the store. Restart with the same ID and store configuration to resume. An unfinished block is regenerated using the same `(seed, worker_id, sequence)`; sequence advances and the claim is released only after the complete block is replaced and its publication is committed. A crash after replacement but before that commit also redoes the same block.

On BlueHive, submit one producer per job; the script always uses `SLURM_JOB_ID` as the worker ID. Requeue retains this ID and resumes its progress; a fresh `sbatch` gets a new ID and registers a new worker. `MAX_GIB` controls the entire store, and optional `MAX_BLOCKS` limits each invocation for smoke tests:

```bash
sbatch bluehive_run/sample-store-produce.sbatch
sbatch bluehive_run/sample-store-produce.sbatch
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
