# LM-SVD: GPT-2 Matrix Experiments

Build matrices from local GPT-2 inference, compute their singular value decompositions (SVDs), measure numerical rank, and plot singular-value decay and cumulative energy. For matrices too large to store, learn leading singular components online with NeuralSVD.

Experiments use sampled history/future token sequences but construct different matrices:

| Experiment | Matrix entry | Matrix shape | Matrix file |
| --- | --- | --- | --- |
| `exp1` | Centered next-token logit for a future-specific header token | histories × (futures × top-k) | `matrix.npy` |
| `exp2` | Joint log probability of history + future given BOS | histories × futures | `feature_log_prob.npy` |
| `exp3` | Difference between two centered-logit matrices using model A's future-specific headers | histories × (futures × top-k) | `matrix.npy` |
| `exp4` | Centered next-token logit for every vocabulary token | histories × (futures × vocabulary) | Not stored; online NeuralSVD |
| `exp5` | Same stored top-k centered-logit matrix as exp1 | histories × (futures × top-k) | Reads exp1's `matrix.npy`; NeuralSVD decomposition |
| `exp6` | Full-vocabulary centered logit on iid stream windows; frozen GPT-2 + trainable MLP features | Streaming history × (future × vocabulary) operator | Not stored; streaming NeuralSVD |
| `exp7` | Same streaming operator; trainable small Transformer features | Streaming history × (future × vocabulary) operator | Not stored; streaming NeuralSVD |
| `exp8` | Same streaming operator; trainable token embeddings + MLP features | Streaming history × (future × vocabulary) operator | Not stored; streaming NeuralSVD |
| `exp9` | Same streaming operator; frozen MiniLM sentence embeddings + trainable MLP features | Streaming history × (future × vocabulary) operator | Not stored; streaming NeuralSVD |
| `exp10` | Same operator and frozen MiniLM + MLP features as exp9; joint NestedLoRA nesting | Streaming history × (future × vocabulary) operator | Not stored; streaming NeuralSVD |

Experiments **exp1–exp3** have two subcommands, run in order: `inference` builds the matrix, and `analysis` runs SVD → numerical rank → visualization together with a selectable secondary centering strategy. Once a matrix exists, their analysis does not load a language model. Their analysis defaults to CUDA; use `--device cpu` to run without a GPU. **exp4** instead uses `train` → `analysis`, defaults to the larger `wikitext-large` dataset, and analyzes learned features on CPU without loading a model or materializing the matrix.

**exp5** has only `analysis`: it learns singular components from exp1's stored matrix and always compares them with exp1's exact SVD for the selected centering. It performs no inference and loads no language model.

**exp6/exp7/exp8/exp9/exp10** use `train` → `analysis` on a token stream rather than a fixed sample table. Both stages load the local language model; analysis compares learned features against an exact fixed-probe reference.

## Navigation

- [Install dependencies](#1-install-dependencies) · [Download models](#2-download-models-for-offline-inference) · [Prepare data](#3-prepare-shared-historyfuture-samples) · [Sample generation and examples](docs/samples.md)
- [exp1](#4-experiment-1-centered-logit-matrix) · [exp2](#5-experiment-2-joint-historyfuture-log-probability-matrix) · [exp3](#6-experiment-3-difference-between-centered-logit-matrices) · [exp4](#7-experiment-4-online-full-vocabulary-neuralsvd) · [exp5](#8-experiment-5-neuralsvd-on-the-exp1-matrix) · [exp1–exp3 analysis](#9-understand-exp1exp3-analysis-outputs)
- [Streaming exp6–exp10](#experiments-6-7-8-9-and-10-streaming-full-vocabulary-neuralsvd) · [Python inference API](#python-inference-api) · [NeuralSVD utility API](#neuralsvd-utility-api) · [Tests](#tests)

## Project layout

| Path | Purpose |
| --- | --- |
| `src/experiments/` | Experiment commands and workflows |
| `src/data_pipline/` | Sample and token-stream preparation |
| `src/utils/` | Shared inference, analysis, and NeuralSVD utilities |
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
# Download or reuse inference files for GPT-2 Small, GPT-2 Medium, and MiniLM.
uv run python scripts/download_models.py

# Optional: download just one model.
uv run python scripts/download_models.py --models medium
```

The script downloads **complete inference files**, not every file in the upstream repositories. Each GPT-2 directory contains `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and `merges.txt`. MiniLM contains `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, and `vocab.txt`. These include the full model weights, not just tokenizers. Duplicate PyTorch, TensorFlow, Flax, Rust, and ONNX weight formats are not downloaded.

| Local directory | Hugging Face repository | Pinned revision |
| --- | --- | --- |
| `models/gpt2-small` | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| `models/gpt2-medium` | [openai-community/gpt2-medium](https://huggingface.co/openai-community/gpt2-medium) | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` |
| `models/all-MiniLM-L6-v2` | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |

Re-running the script reuses files already downloaded at the pinned revisions. `--output-dir PATH` changes the parent directory of all three model directories; by default it is the repository's `models/` directory. Use `--models minilm` to download only the exp9/exp10 encoder. Internet access is required to fetch missing files. Download the source dataset in the next section before enabling offline mode.

After downloading the required models and source data, all tokenization and model inference can run offline.

Both GPT-2 models have been verified with local-only tokenizer loading and CPU inference, including next-token logits and conditional continuation log probabilities. MiniLM has been verified with local-only sentence encoding and an exp9 CUDA train/analysis/resume smoke.

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

### Sync analysis results from BlueHive

Run locally to pull only `data/experiment_results/<exp>/<run>/analysis/` trees:

```bash
scripts/sync_circ_analysis.sh -n  # Preview transfers and deletions.
scripts/sync_circ_analysis.sh
```

The script uses the SSH alias `circ` and remote repository `/scratch/lli81/projects/ur_research/lm-svd`. Override either with `REMOTE` or `REMOTE_ROOT`. It runs from the local repository root regardless of the calling directory.

`--delete` is enabled: local analysis files absent from the remote are removed. Excluded `inference/`, `train/`, and loose files are neither copied nor deleted. Empty experiment/run directories without analysis are pruned from the transfer list. Additional arguments are passed to rsync.

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


## 5. Experiment 2: joint history/future log-probability matrix

For each history/future pair, sum the log probabilities of the history and future tokens starting from BOS:

```text
B[history, future] = log p(history + future | BOS) = sum_t log p(history_t | BOS + history_<t) + sum_t log p(future_t | BOS + history + future_<t)
```

These are log-probability **sums**, not averages, probabilities, or centered logits. Rows correspond to histories; columns correspond to futures. Histories and futures may have different token lengths, including within a batch; both lengths affect the summed scores.

The model configuration must provide `bos_token_id`; there is no fallback if it is missing. GPT-2 uses `<|endoftext|>` (50256) as the document-start context, even for histories sampled from the middle of documents. Inference metadata records `bos_token_id` and the score definition.

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

`inference` writes `feature_log_prob.npy` and `metadata.json` under the unchanged `$RESULT_DIR/inference/` path; `exp2 analysis` reads `inference/feature_log_prob.npy`. Legacy conditional results are not migrated. Replacing existing outputs requires `--overwrite`; the metadata's score definition and `bos_token_id` distinguish joint runs from legacy runs.

On CIRC BlueHive, `bluehive_run/exp2-inference.sbatch` and `bluehive_run/exp2-analysis.sbatch` both request partition `h100`, QoS `h100`, and one H100 GPU. They retain 4 CPUs and 16G memory, with 6 hours for inference and 1 hour for analysis. Submit from the repository root with matching `DATASET` values and an `afterok` dependency; analysis runs all four centering modes. Both scripts overwrite existing outputs, so cancel any older jobs for the same dataset before resubmitting.

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

For a bias-free output layer with token embeddings `e_z`, centered logits equal `s(h, f) · (e_z - mean_e)`, where `s` is the final hidden state. exp4 uses a dense history lookup for left features and the right parameterization `g_l(f, z) = W_f[l] · (e_z - mean_e)`, with `W_f` of shape `(rank, hidden_size)`. Thus right features need no lookup table over all future/token pairs. Training uses Adam with dense embedding gradients and rejects output layers with bias.

### Workflow and defaults

```bash
uv run data-pipline build-wikitext-large
uv run exp4 train
uv run exp4 analysis
```

Both stages default to `--dataset wikitext-large`, `--model-path models/gpt2-small`, `--results-root data/experiment_results`, and `--device cuda`; use `--device cpu` for either stage without CUDA. Training uses all available histories and futures, rank 256 when `--rank` is omitted, sequential nesting, history/future batches of 64 each, base learning rate `--lr 1e-2`, seed 0, and logging every 100 steps. The default `--lr-schedule cosine` decays without warmup to `--min-lr 1e-4` over `--lr-decay-steps 30000` updates, starting at step 0 for fresh training or at the resumed step of a constant-lr checkpoint. Cosine training stops at the schedule end (`stop_reason: lr_schedule_end`), not on convergence. Every `--eval-every 1000` steps, training estimates all learned singular values using the analysis formula and records convergence when their maximum relative change is below `--tol 1e-3` for `--patience 3` consecutive evaluations. Use `--lr-schedule constant` to stop on that convergence criterion instead. Set `--max-steps` to cap total steps across resumes; it defaults to no additional cap. Use `--history-count` / `--future-count` for leading subsets, `--history-batch` / `--future-batch` for block sizes, `--rank`, `--lr`, `--log-every`, or `--nesting joint` to adjust training.

Runs are stored under `<results-root>/exp4_neural_svd/<dataset>__<model-directory-name>/`; the directory name does not include rank. Pass matching dataset, results-root, and model-path flags to both stages for nondefault runs. Analysis reads rank from the saved training metadata and needs no rank argument. Training against an existing checkpoint with a different rank fails the resume config check rather than silently restarting.

### BlueHive

After installing `.venv` and preparing `models/gpt2-small` and `data/history_future_samples/wikitext-large.parquet` on the cluster, submit the training-only exp4 job from the repository root:

```bash
cd /scratch/lli81/lm-svd
mkdir -p bluehive_run/logs
sbatch bluehive_run/exp4-train.sbatch
```

Before the first rank-256 submission, preserve an existing default-dataset rank-32 run by moving it aside on the cluster (only if the source directory exists, and before running `sbatch` above):

```bash
mv data/experiment_results/exp4_neural_svd/wikitext-large__gpt2-small \
   data/experiment_results/exp4_neural_svd/wikitext-large__gpt2-small__rank32
```

The training script accepts `DATASET` and `RANK` environment overrides at submission time; unset values default to `wikitext-large` and `256`, respectively:

```bash
RANK=128 sbatch bluehive_run/exp4-train.sbatch
DATASET=task-single-text RANK=128 sbatch bluehive_run/exp4-train.sbatch
```

For a nondefault dataset, submit analysis with the same `DATASET`, for example `DATASET=task-single-text sbatch bluehive_run/exp4-analysis.sbatch`. Analysis uses the saved training rank, not `RANK`.

The training job requests one H100 on the `preempt` partition, 4 CPUs, 64 GiB host memory, and one day, with requeue enabled and Slurm logs appended. It runs only `exp4 train` with the defaults above unless overridden as shown. Training does not use `--overwrite`: it resumes an existing checkpoint, while completed training outputs without a checkpoint cause an error. Do not submit overlapping jobs against the same result directory.

Resubmit the same training job after a timeout or failure to resume from the latest checkpoint. An existing constant-lr checkpoint starts a 30,000-step cosine decay from its saved step, in place, without copying the old state. Subsequent resumes and Slurm requeues reuse the saved decay start; they neither restart nor extend the schedule. A rerun after the schedule end performs no updates. Submit `sbatch bluehive_run/exp4-analysis.sbatch` separately to run `exp4 analysis --overwrite`, including on a saved snapshot before convergence. The analysis job requests one GPU on the `gpu` partition, 4 CPUs, 32 GiB host memory, and one hour. Exp4 analysis has no `--centering` option and runs once on the learned centered-logit matrix. Resource requests are not a measured guarantee that the full dataset fits or converges within the time limit; adjust the script's batch sizes and memory/time requests based on observed usage.

Slurm stdout/stderr go to `bluehive_run/logs/slurm-lm-svd-exp4-train-<job-id>.out` and `.err` for training, or `slurm-lm-svd-exp4-analysis-<job-id>.out` and `.err` for analysis. Stage logs go to the matching `bluehive_run/logs/slurm-<job-name>-<job-id>/` directory as `exp4-train.out` or `exp4-analysis.out`. Create the log directory before submission so Slurm can open its files.

### Outputs and interpretation

- `train/`: while training, `checkpoint.pt` stores feature weights, Adam moments, RNG state, loss history, convergence state, and learning-rate schedule state every `--checkpoint-every 100` steps, when capped, and at the cosine schedule end. At the default rank 256 and dataset size, the checkpoint is estimated at about 24 GB and `features.pt` at about 7.9 GB. Training automatically resumes from it; interruption between checkpoints loses at most 99 steps at the default interval. Each checkpoint also refreshes six analysis-ready snapshot files: `features.pt` (history and future lookup weights), `token_gram.npy` (the centered token embedding Gram matrix), `loss_history.npy`, `histories.parquet`, `futures.parquet`, and `metadata.json` (matrix definition, training settings/results, `lr_schedule`, `final_lr`, convergence statistics, and source provenance). Each file is replaced atomically, with metadata last; concurrent analysis can read files from adjacent snapshots. Constant-schedule convergence writes the final snapshot and removes the checkpoint. Cosine training records convergence without stopping or deleting the checkpoint, and retains it at the schedule end.
- `analysis/`: exactly four files: `singular_values.npy`, `left_singular_vectors.npy` (unit-norm columns), `neural_svd_spectrum.png`, and `metadata.json`. The two arrays are written first, then the plot, and `metadata.json` last as the completion marker. Its four top-level sections, in order, are `input`, `spectrum`, `orthogonality`, and `loglog_fit`. `input` records the training directory, input filenames, matrix shape, and the complete training metadata under `train_metadata`; `spectrum` records factor and plot filenames, learned rank/order, singular values, the squared Frobenius norm estimate, and cumulative captured energy with their formulas. `orthogonality` contains `left_error`, `right_error`, and their definition. Input filenames are relative to `input.train_dir`; factor and plot filenames are relative to the analysis metadata directory.

The plot shows singular values on log-log axes with a fitted line labeled by slope `k` and intercept `b`, cumulative captured energy, and training loss. The fit uses every learned singular value in learned order, with indices starting at 1: `log10(σ_i) = k * log10(i) + b`. Least squares weights each squared residual by its log-index cell width, `weight_i = log10((i + 0.5) / (i - 0.5))`, so densely sampled high indices do not dominate the fit. The top-level `loglog_fit` section stores `slope`, `intercept`, `point_count`, `log_base`, `formula`, and `weighting`; the CLI also prints `k` and `b`. With fewer than two points the fit is skipped and its slope/intercept are null, but `weighting` is still recorded. Loss history is read from the training snapshot, not embedded in analysis metadata. Spectrum and energy panels share the exp1–exp3 plotting layer, but exp4 has no numerical-rank markers or fixed energy limits.

Analysis computes feature statistics in torch float64 on the selected device and reads only `train/{features.pt,token_gram.npy,loss_history.npy,metadata.json}`. Future weights remain on CPU and move to the device in chunks of at most 1024 futures, using a flat 2-D matrix product with the token Gram matrix. Downstream diagnostics and output processing remain in NumPy. `--model-path` supplies only the directory name for locating the run; no language model or sample file is loaded. The token Gram matrix gives right-feature norms and inner products without constructing the enormous right-feature array. Analysis does **not** compute an exact matrix SVD, save right singular vectors, or estimate numerical rank. It does not accept `--centering`: no secondary centering is available because the matrix is not stored.

For a capped feasibility smoke, run `uv run exp4 train --max-steps 2 --checkpoint-every 1 --log-every 1 --results-root /tmp/exp4-smoke`. It prints `stop_reason: max_steps` and leaves a checkpoint and analysis-ready snapshot. Run `uv run exp4 analysis --results-root /tmp/exp4-smoke` to analyze it; the summary reports the training step and convergence status. Rerun training with `--max-steps 4` to continue from step 2, or omit the cap to train until the cosine schedule ends. Rerun analysis with `--overwrite` to refresh its outputs. Add `--device cpu` to either command to run without CUDA.

Singular values retain **learned order**, not sorted order. In `metadata.json`, `orthogonality.left_error` and `orthogonality.right_error` report maximum absolute off-diagonal inner products for normalized left/right features. `spectrum.frobenius_norm_sq_estimate` is the sampled squared Frobenius norm estimate, and `spectrum.captured_energy_fraction` divides cumulative squared learned singular values by that estimate; it can exceed 1 before convergence and is reported without clipping. Both the singular-value and captured-energy lists have length `spectrum.learned_rank`. These are learned estimates, not a guarantee of convergence or exact leading singular values; inspect orthogonality, ordering, energy, and loss together.

Training resumes by default when a checkpoint exists, including when snapshot outputs exist alongside it. Resume requires matching sample/model paths, sample counts, rank, nesting, batch sizes, base learning rate, seed, and convergence settings; device, logging/checkpoint intervals, and the total-step cap may change. Constant-lr checkpoints, including legacy checkpoints without schedule state, may switch to cosine. Once cosine is saved, its minimum learning rate and decay length must match, and switching back to constant is rejected. Existing training outputs without a checkpoint require `--overwrite`, which discards both checkpoint and training outputs and restarts at step 0; do not use it to continue an existing run. Analysis also requires `--overwrite` to replace its outputs. Replacing training outputs does not refresh analysis; rerun `exp4 analysis --overwrite` when needed. Use `uv run exp4 <subcommand> --help` or `uv run python -m experiments.exp4_neural_svd <subcommand>`.

Legacy SparseAdam checkpoints are incompatible with dense Adam and fail the resume config check. Start the replacement run once with `uv run exp4 train --overwrite`; subsequent runs and BlueHive requeues resume normally without `--overwrite`.

## 8. Experiment 5: NeuralSVD on the exp1 matrix

exp5 isolates the decomposition method: it reads `<results-root>/exp1_L_M_table/<dataset>__<model-directory-name>/inference/matrix.npy` and its required `metadata.json`, then fits dense row/column lookup tables with sequential NeuralSVD and Adam. It does not run inference, load a model, or read sample files; `--model-path` supplies only the directory name used to locate the run.

### Prerequisites and commands

Both **exp1 inference and exp1 exact analysis for the selected centering** must already exist under the same results root, dataset, and model name. Missing inputs or exact factors are errors before training starts. For example, after the exp1 inference workflow above:

```bash
# Prepare the required exact reference, then fit and compare NeuralSVD.
uv run exp1 analysis --centering none
uv run exp5 analysis

# A different centering requires its own exact reference.
uv run exp1 analysis --centering deflate1
uv run exp5 analysis --centering deflate1

uv run exp5 analysis --help
```

These commands default to `--dataset wikitext`, `--model-path models/gpt2-small`, `--results-root data/experiment_results`, `--centering none`, and `--device cuda`. Pass matching dataset, model-path, and results-root flags to exp1 and exp5 for nondefault runs. Use `--device cpu` without CUDA; use `--overwrite` to replace existing analysis outputs.

Training defaults are `--rank 64`, `--history-batch 64`, `--future-batch 64`, `--steps 30000`, `--lr 1e-2`, `--min-lr 1e-4`, `--seed 0`, and `--log-every 1000`. The learning rate follows cosine decay over all steps. Each step samples histories and futures uniformly without replacement and includes **all top-k columns** for each sampled future. Batch sizes must not exceed the stored history/future counts.

### Centering and interpretation

`--centering {none,global,row,deflate1}` supports the same four strategies as exp1. `none`, `global`, and `row` use exactly the same transformed matrix as exp1. For `deflate1`, exp5 instead trains `rank + 1` components on the original stored matrix, drops learned component 0, and measures reconstruction and captured energy against `M - S[0] * outer(U[:, 0], Vt[0, :])`. **exp5 performs no exact SVD**, including for deflation; its required exp1 reference is computed separately.

Singular values remain in **learned order**, not sorted order. Metadata reports reconstruction relative error, maximum absolute off-diagonal Gram entries for unit-norm left/right factors, and the log-log fit over all retained components. Captured energy is `cumsum(S**2) / ||X||_F**2` for the analyzed matrix `X`; it is not normalized to the retained spectrum or clipped and can exceed 1. These learned estimates do not guarantee convergence.

Comparison is always against exp1's matching centering: index-matched singular-value relative errors, left/right absolute vector cosines, and left/right subspace overlaps. Index matching compares learned order with exp1's descending order; subspace overlap is useful when nearby components swap. The exact captured-energy overlay uses exp1's full spectrum as its denominator.

### Outputs and BlueHive

Results go to `<results-root>/exp5_L_M_neural_svd/<dataset>__<model-directory-name>/analysis/<centering>/`:

- `svd_U.npy`, `svd_S.npy`, and `svd_Vt.npy`: retained unit-norm factors and learned singular values.
- `loss_history.npy`: one loss value per training step.
- `svd_spectrum.png`: four panels showing the log-log spectrum and fit, cumulative captured energy, alignment with exp1's left/right vectors, and training loss. Spectrum and energy panels include green dashed exp1 exact-SVD overlays.
- `metadata.json`: input provenance, centering formula, training settings, factor diagnostics, spectrum, log-log fit, and exp1 comparison. It is written last as the completion marker.

After preparing exp1 inference and exact analysis for **all four centerings**, submit from the repository root:

```bash
mkdir -p bluehive_run/logs
sbatch bluehive_run/exp5-analysis.sbatch
```

The batch script runs all four exp5 centerings with `--overwrite` and a two-hour time limit, writing separate `exp5-analysis-<centering>.out` logs. It does not generate the exp1 prerequisites.

## 9. Understand exp1–exp3 analysis outputs

Experiments exp1–exp3 each own their `analysis` workflow. It reads the matrix and optional sibling `metadata.json` from `<result-dir>/inference/`, then writes exactly five outputs to `<result-dir>/analysis/<centering>/`:

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

`<run>` is a flat, string-joined directory name: `<dataset>__<model>` for exp1/exp2 (for example, `wikitext__gpt2-small`), or `<dataset>__<model_a>__minus__<model_b>` for exp3 (for example, `wikitext__gpt2-small__minus__gpt2-medium`). Model labels are the final directory names of the model paths. The dataset is not a separate directory level. Only the requested strategy directory is created.

### Secondary centering

`analysis --centering {none,global,row,deflate1}` selects how the stored matrix `M` is transformed before SVD; the default is `none`:

| Strategy | Matrix decomposed |
| --- | --- |
| `none` | `M` |
| `global` | `M - mean(M)` |
| `row` | `M - mean(M, axis=1, keepdims=True)` |
| `deflate1` | `M - S[0] * outer(U[:, 0], Vt[0, :])`, where `M = U @ diag(S) @ Vt` |

Row centering subtracts each history's mean across all columns: futures × header tokens for exp1/exp3, or futures for exp2. This is **secondary** centering: it does not change the base vocabulary-mean logit subtraction in exp1/exp3, and exp2 inference still stores uncentered joint log-probability sums. Selecting only top-k header tokens can leave the stored exp1/exp3 matrix biased toward large values despite its base vocabulary centering.

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
  --dataset wikitext --results-root data/experiment_results \
  --centering none --overwrite
```

Rerunning analysis does not remove preexisting legacy JSON artifacts. Manually remove obsolete analysis JSON files after refreshing existing exp1–exp4 results, retaining the new `metadata.json`.

### Directory selection and existing results

**Analysis does not infer the model or dataset from your last `inference` call.** Both subcommands default to `--dataset wikitext` and `--results-root data/experiment_results`, with `--model-path models/gpt2-small` for exp1/exp2, or `--model-a-path models/gpt2-small --model-b-path models/gpt2-medium` for exp3. Pass matching dataset, results-root, and model flags to inference and analysis for nondefault runs, as shown above.

Analysis uses only the model paths' directory names to locate the run: it does not load a model, check that the model directory exists, or require/check the sample file. It validates only the dataset name's format. Inference resolves the named sample file in the fixed samples directory. `--results-root PATH` selects the parent of the experiment directory, not the run root or its `inference/` or `analysis/` child.

Existing WikiText results have moved to the dataset-prefixed run directories above, and prepared sample files have been renamed to the plain dataset filenames. Historical `metadata.json` files are unchanged: their original `samples_path` values remain records of the inputs used at the time, even though those files have since been renamed.

Inference refuses to replace its existing outputs without `--overwrite`. Analysis prechecks all five outputs for the selected strategy and refuses to start if any exists unless `--overwrite` is supplied; this prevents an existing factor, metadata, or plot file from causing a late overwrite conflict after SVD. `analysis --overwrite` replaces only that strategy's analysis outputs, not inference files or other strategies. When regenerating inference, rerun analysis with `--overwrite` for every previously computed strategy you intend to use; overwriting inference does not refresh analysis.

Other useful options:

- `--dataset NAME`: select a named sample set for inference or analysis; default `wikitext`.
- `--results-root PATH`: select the results root for inference or analysis; default `data/experiment_results`.
- `inference --history-count N --future-count N`: use only the leading N samples of each group; omit both to use all available samples.
- `exp1 inference --top-k K`: select K header tokens per future; default 50.
- `exp2 inference --batch-size N`: score N futures per inference batch; default 64.
- `inference --device cpu`: use CPU instead of the default CUDA device.

Use `uv run exp1 <subcommand> --help`, `uv run exp2 <subcommand> --help`, or `uv run exp3 <subcommand> --help` for available options. Module entry points also work: `uv run python -m experiments.exp1_L_M_table <subcommand>`, `uv run python -m experiments.exp2_PFH <subcommand>`, and `uv run python -m experiments.exp3_L_M_diff <subcommand>`.

## Experiments 6, 7, 8, 9, and 10: streaming full-vocabulary NeuralSVD

These independent experiments replace exp4's per-sample lookup tables with functions of token content. Each step independently draws fresh history and future window starts **uniformly with replacement** from the token stream (duplicates are allowed). Histories and futures are not required to be adjacent in the original text. Every sampled future contributes **all vocabulary tokens** (50,257 for GPT-2); there is no top-k selection, secondary centering, or first-component removal.

```text
A[h, (f, z)] = logit(z | h + f) - mean_vocabulary(logits | h + f)
             = s(h + f) · (e_z - mean_e)
(T v)(h) = E_{f,z}[A[h, (f, z)] v(f, z)]
```

Here `h` and `f` have independent uniform-start window distributions, `z` is uniform over the vocabulary, and `s` is GPT-2's final hidden state. The bias-free centered-unembedding identity computes the exact full-vocabulary block without storing the infinite table. The teacher model is frozen; NestedLoRA with Adam trains the feature functions.

### Prepare a token stream and run

Download the local model and WikiText source as above, then tokenize the entire split once:

```bash
uv run data-pipline build-token-stream
uv run exp6 train
uv run exp6 analysis
uv run exp7 train
uv run exp7 analysis
uv run exp8 train
uv run exp8 analysis
uv run exp9 train
uv run exp9 analysis
uv run exp10 train
uv run exp10 analysis
```

The builder defaults to `--dataset-dir data/original/wikitext --config wikitext-103-raw-v1 --split train --tokenizer-path models/gpt2-small --output data/token_streams/wikitext-103-train.npy`. It writes a 1-D integer `.npy` token stream and sibling `.json` with tokenizer, vocabulary, split, dtype, and token-count metadata. Use `--overwrite` to replace them. Training memory-maps the array; it does not use the Parquet history/future sample tables. For another stream, write `data/token_streams/NAME.npy` and its JSON, and pass `--stream NAME`. Use a tokenizer matching the teacher model.

All five experiments default to `--stream wikitext-103-train`, `--model-path models/gpt2-small`, `--results-root data/experiment_results`, and `--device cuda`; `--device cpu` is supported. Shared training defaults, except exp10's fixed joint nesting:

| Flags | Defaults |
| --- | --- |
| `--rank`, `--nesting` | `256`, `sequential` for exp6–exp9 (`joint` also supported); exp10 requires `joint` |
| `--history-tokens`, `--future-tokens` | `64`, `32` |
| `--history-batch`, `--future-batch` | `64`, `64` |
| `--probe-histories`, `--probe-futures`, `--probe-seed` | `1024`, `1024`, `1` |
| `--lr`, `--lr-schedule`, `--min-lr`, `--lr-decay-steps` | `1e-3`, `cosine`, `1e-5`, `30000` |
| `--eval-every`, `--tol`, `--patience`, `--convergence-rank` | `1000`, `1e-3`, `3`, `32` |
| `--checkpoint-every`, `--log-every`, `--seed` | `100`, `100`, `0` |
| `--max-steps` | No cap |

`--probe-histories` must be at least the rank, and the concatenated history/future must fit the teacher's context.

- **exp6** (`exp6_stream_gpt2_mlp`): frozen GPT-2 encodes each window; concatenate the final-token hidden state and mean hidden state. Separate left/right MLPs use non-affine LayerNorm, two GELU hidden layers, and output heads. `--mlp-width 1024` controls both hidden layers. GPT-2 weights are neither optimized nor saved in the feature state.
- **exp7** (`exp7_stream_transformer`): separate history/future Transformers train from scratch on token IDs, with learned token/position embeddings, pre-norm layers, GELU, no dropout, and unmasked attention within each window. Concatenated final-token/mean pooling feeds left/right linear heads. Defaults are `--model-dim 256 --layers 4 --heads 4`; model dimension must be divisible by heads. GPT-2 only supplies the target blocks and centered token embeddings.
- **exp8** (`exp8_stream_token_mlp`): independent history/future token embeddings and MLPs train from scratch. Each window's embeddings are concatenated in position order into `window_tokens * embedding_dim` features, preserving order without additional positional encoding. Defaults are `--embedding-dim 256 --mlp-width 1024`. Each MLP uses non-affine LayerNorm, two `Linear → GELU` hidden layers, and a left rank-output or right rank-times-teacher-hidden-size output head, matching exp6's MLP structure. GPT-2 does not compute the input features: it only supplies target blocks, centered token embeddings, and their vocabulary/hidden-size and token-RMS information.
- **exp9** (`exp9_stream_sentence_mlp`): decode each entire GPT-2 history/future window to text, then encode it with frozen `sentence-transformers/all-MiniLM-L6-v2`, attention-mask mean pooling, and L2 normalization into one 384-dimensional vector. Windows may cross sentence boundaries or start inside words; no splitting or repair is applied. Defaults are `--encoder-path models/all-MiniLM-L6-v2 --max-seq-length 256 --mlp-width 1024`. MiniLM retokenization truncates at this sequence limit. One shared float32 encoder stays in eval mode, outside feature parameters, state dicts, and module counts; independent left/right MLPs use exp6's structure with 384-dimensional inputs. Both the local GPT-2 tokenizer and MiniLM model/tokenizer are required. Encoder path and architecture are saved in `features` metadata, so analysis needs no encoder flags.
- **exp10** (`exp10_stream_sentence_mlp_joint`): exp9's control with identical frozen MiniLM features, trainable MLPs, and training defaults, except NestedLoRA nesting is fixed to `joint` (equal weights on all prefixes, `step=1`). `exp10 train --nesting sequential` is rejected before creating outputs; use exp9 for sequential nesting. The CLI and analysis feature builder are reused from exp9, not copied.

All five right heads produce `(future_batch, rank, teacher_hidden_size)` weights, contracted with the centered unembedding for vocabulary features, with a fixed initialization scale based on hidden size and token RMS.

With GPT-2's vocabulary/hidden size, default windows 64/32, rank 256, embedding dimension 256, and MLP width 1024, exp8's trainable parameters are:

| Component | Parameters |
| --- | ---: |
| Two token embeddings | 25,731,584 |
| History MLP, including left output layer | 18,090,240 |
| Future MLP, excluding right output layer | 9,439,232 |
| Right output layer `Linear(1024, 196608)` | 201,523,200 |
| **Total** | **254,784,256** |

The total is confirmed by `sum(p.numel() for p in features.parameters())` for the default architecture; the large right output head dominates.

For exp9 and exp10 with GPT-2 Small, rank 256, and MLP width 1024, measured trainable counts exclude both frozen models:

| Component | Parameters |
| --- | ---: |
| Left MLP | 1,706,240 |
| Right MLP | 202,967,040 |
| **Total** | **204,673,280** |

### Smoke, resume, and schedules

For a short capped CUDA run of each architecture, using a separate result root:

```bash
for e in exp6 exp7 exp8 exp9 exp10; do
  uv run "$e" train --rank 4 --history-batch 4 --future-batch 4 \
    --probe-histories 32 --probe-futures 8 --max-steps 20 \
    --eval-every 10 --checkpoint-every 10 --log-every 5 \
    --results-root /tmp/stream-smoke
  uv run "$e" analysis --results-root /tmp/stream-smoke --reference-future-batch 2
done

# Continue the exp9 smoke from step 20; retain its configuration.
uv run exp9 train --rank 4 --history-batch 4 --future-batch 4 \
  --probe-histories 32 --probe-futures 8 --max-steps 30 \
  --eval-every 10 --checkpoint-every 10 --log-every 5 \
  --results-root /tmp/stream-smoke
```

The capped runs report `stop_reason: max_steps`; continuation prints `resuming from step 20`. This is a feasibility smoke, not a convergence claim. Training automatically restores feature weights, Adam moments, sampling RNG, loss history, and probe-stability state from `train/checkpoint.pt`.

BlueHive sentence-MLP templates are `bluehive_run/exp9-{train,analysis}.sbatch` and `bluehive_run/exp10-{train,analysis}.sbatch`. Submit from the repository root after provisioning the local models and stream. Training intentionally omits `--overwrite` to resume checkpoints; analysis reads the encoder path from training metadata.

Cosine decay starts at step 0 without warmup, reaches `--min-lr` after `--lr-decay-steps` (30,000 by default), and holds that rate until convergence. It does not stop before the decay endpoint even if the probe stabilizes earlier; resuming does not restart or extend the decay. `--lr-schedule constant` stops on convergence without a decay wait. The optional `--max-steps` caps either schedule. Probe stability means the maximum relative change among the leading `--convergence-rank` singular values in learned order stays below `--tol` for `--patience` evaluations. The convergence rank defaults to 32 and is capped at `--rank`; this stability check is not an exact-reference accuracy criterion.

Resume configuration must match, including architecture, rank, windows, batches, stream/model paths, optimizer/schedule settings, seed, and probe selection settings. Mismatches raise `checkpoint config mismatch`; there is **no** constant-to-cosine migration. The convergence settings `--eval-every`, `--tol`, `--patience`, and `--convergence-rank` can change on resume, resetting the probe-stability count. Legacy checkpoints with convergence settings stored in their configuration are also accepted and reset probe stability on resume. The step cap and logging/checkpoint intervals can change. Existing training outputs without a checkpoint require `--overwrite`, which restarts at step 0 rather than resuming. Analysis likewise requires `--overwrite` to replace its outputs; replacing training does not refresh existing analysis. Do not run overlapping writers against the same directory.

### Paths, outputs, and probe interpretation

Run directories are `<results-root>/<experiment>/<stream>__<model-directory-name>/`, where `<experiment>` is `exp6_stream_gpt2_mlp`, `exp7_stream_transformer`, `exp8_stream_token_mlp`, `exp9_stream_sentence_mlp`, or `exp10_stream_sentence_mlp_joint`. Rank and architecture flags are not encoded in the directory name. Pass matching stream, results-root, and model-path flags to locate a nondefault run for analysis.

- `train/features.pt`: trainable feature state and buffers.
- `train/loss_history.npy`: one loss per completed step.
- `train/probe.npz`: fixed history/future start indices, sampled once using the probe seed.
- `train/metadata.json`: configuration, stream metadata, steps, loss/entry-energy estimates, convergence state, and latest probe spectrum (null before the first evaluation); written last. Shared `parameter_counts` records `trainable_total` and `modules` (trainable counts by top-level child module) for exp6–exp10. These counts are metadata only, not part of the resume configuration, so adding them does not invalidate existing checkpoints. Frozen teacher and sentence-encoder parameters are excluded.
- `train/checkpoint.pt`: resumable state, saved with a refreshed snapshot every checkpoint interval and at the step cap. Convergence removes the checkpoint.
- `analysis/`: six files: `singular_values.npy`, `reference_singular_values.npy`, `left_singular_vectors.npy`, `reference_left_singular_vectors.npy`, `neural_svd_spectrum.png`, and `metadata.json` (written last). Vector arrays are probe-histories × rank, with unit-norm columns. The plot compares learned/reference spectra and cumulative energy alongside training loss.

Unlike exp4 analysis, **analysis reloads both the token stream and the local teacher model from paths saved in training metadata** to reconstruct the probe and exact reference. Analysis `--model-path` only names the run directory; it does not override the saved teacher path. Keep those inputs available. `--reference-future-batch 4` controls futures per reference forward: with `Hp` probe histories, each forward contains `q * Hp` concatenated sequences. Lower `q` to reduce peak memory (at the cost of more forwards); the reference still evaluates every probe history/future pair.

For exp9 and exp10, analysis also reloads the local sentence encoder using the path and sequence limit saved in training `features` metadata. Keep that directory and the teacher tokenizer available alongside the stream and teacher weights.

Singular values are in **operator units**, with no finite-matrix `sqrt(m*n)` multiplier. For a probe with `Hp` histories, `Fp` futures and vocabulary `V`, the exact reference is the SVD of the dense probe table divided by `sqrt(Hp * Fp * V)`. Analysis obtains it through the history Gram matrix and token Gram matrix rather than materializing the vocabulary-wide table. Its values are sorted descending; learned values retain **learned order**. These units differ from exp4's finite-table matrix singular values.

The fixed probe is an empirical sub-table, not an exact SVD of the full stream operator. Analysis reports:

- Per-component relative error `learned_sigma / reference_sigma - 1`, comparing learned order to descending reference order.
- Left-vector alignment `abs(dot(learned_u, reference_u))`; near-degenerate reference components can rotate, so also inspect prefix subspace overlap `||U_ref[:, :k].T @ Q_k||_F**2 / k`, where `Q_k` orthonormalizes the first `k` learned vectors.
- Learned and exact-reference cumulative energy `cumsum(sigma**2) / probe.energy`, where `probe.energy` is the mean squared entry of the full probe table, **not** the energy of just the retained rank. Learned fractions can exceed 1 and are not clipped.
- Maximum absolute off-diagonal normalized left/right Gram entries and a learned-order log-log spectrum fit.

Stable probe norms, low loss, or a completed learning-rate schedule alone do not guarantee convergence, correct ordering, orthogonality, or full-stream generalization. Use the exact-reference errors, alignments, overlap, and energy together.

### Comparing architectures fairly

Run exp6, exp7, exp8, and exp9 with the same stream, rank, history/future windows and batches, learning-rate schedule, training steps, and probe seed. Compare singular-value relative errors, left-vector alignment, prefix subspace overlap, captured energy, trainable parameter counts, time per step, and peak GPU memory. Keep the teacher and probe sizes fixed as well so the target operator and exact-reference comparison are shared.

**exp8 versus exp6 asks whether frozen GPT-2 input representations help** when using the same MLP structure. exp8 versus exp7 compares complete architectures, not parameter-matched MLP and Transformer backbones. A parameter-matched capacity comparison requires a separate width adjustment while keeping the output heads fixed.

**exp10 versus exp9 isolates joint versus sequential nesting.** Compare at matched training steps with the same stream, teacher, rank, architecture, batches, schedule, seeds, and fixed probe; exp10 starts independently rather than restarting exp9.

### BlueHive

Prepare `.venv`, the local GPT-2 model, the MiniLM encoder for exp9/exp10, and token-stream `.npy`/`.json` on the cluster, then submit from the repository root:

```bash
mkdir -p bluehive_run/logs
sbatch bluehive_run/exp6-train.sbatch
sbatch bluehive_run/exp7-train.sbatch
sbatch bluehive_run/exp8-train.sbatch
sbatch bluehive_run/exp9-train.sbatch
sbatch bluehive_run/exp10-train.sbatch
# Submit analysis separately after a snapshot is available.
sbatch bluehive_run/exp6-analysis.sbatch
sbatch bluehive_run/exp7-analysis.sbatch
sbatch bluehive_run/exp8-analysis.sbatch
sbatch bluehive_run/exp9-analysis.sbatch
sbatch bluehive_run/exp10-analysis.sbatch

# Optional training overrides; use the same STREAM for analysis.
STREAM=wikitext-103-train RANK=128 sbatch bluehive_run/exp6-train.sbatch
STREAM=wikitext-103-train sbatch bluehive_run/exp6-analysis.sbatch
STREAM=wikitext-103-train RANK=128 sbatch bluehive_run/exp8-train.sbatch
STREAM=wikitext-103-train sbatch bluehive_run/exp8-analysis.sbatch
```

Training scripts default to `STREAM=wikitext-103-train`, `RANK=256`; all five accept these overrides. They request one H100 on `preempt`, 4 CPUs, 64 GiB host memory, and two days. Slurm sends `USR1` 600 seconds before the time limit, triggering automatic requeue and resume from the latest checkpoint; both training stage and Slurm logs are appended across requeues. Training errors fail the job without requeueing. Use `scancel <job-id>` to stop repeated requeues. The scripts do not pass `--overwrite`. If rank 256 exhausts H100 memory, use `RANK=128` for any experiment in a fresh run (or explicitly restart incompatible existing outputs); code defaults stay unchanged. Analysis scripts use the same `STREAM` override, read the saved rank rather than `RANK`, and run with `--overwrite`. They request one GPU on `gpu`, 4 CPUs, 32 GiB host memory, and **four hours** for the exact probe reference.

Slurm logs are `bluehive_run/logs/slurm-lm-svd-exp{6,7,8,9,10}-{train,analysis}-<job-id>.{out,err}`; stage logs live under the corresponding `slurm-<job-name>-<job-id>/` directory as `exp{6,7,8,9,10}-{train,analysis}.out` (for example, `exp10-train.out` or `exp10-analysis.out`). Use `uv run expN <subcommand> --help` for all flags (`N` is 6–10); module entry points are `experiments.exp6_stream_gpt2_mlp`, `experiments.exp7_stream_transformer`, `experiments.exp8_stream_token_mlp`, `experiments.exp9_stream_sentence_mlp`, and `experiments.exp10_stream_sentence_mlp_joint`.

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

`final_hidden_states(input_ids)` accepts a non-empty 2-D integer tensor of equal-length token-ID sequences, validates vocabulary/context bounds, and returns float32 `(batch_size, sequence_length, hidden_size)` states on the inference device under `torch.no_grad()`. For GPT-2 these are post-final-LayerNorm states; multiplying them by the unembedding transpose reproduces logits without materializing full-sequence logits inside the method.

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
visible. exp4 uses the same learned-order convention with closed-form right-feature moments, and exp5 applies it to exp1's stored finite matrix; exp1–exp3 retain their exact matrix-analysis workflows.

## Tests

```bash
uv run python -m unittest discover -s tests
```

The default suite keeps short-step training tests using a tiny random GPT-2 fixture. The convergence/SVD-recovery test is opt-in because it trains to convergence:

```bash
RUN_SLOW=1 uv run python -m unittest \
  tests.test_exp4_neural_svd.Exp4NeuralSVDTests.test_training_and_analysis_recover_top_two_singular_values -v
```

Streaming exp6/exp7/exp8 also have opt-in tiny-model top-two singular-value recovery tests (30,000-step cosine decay followed by training until convergence, with a 60,000-step guard for each architecture; no downloaded model or dataset needed):

```bash
RUN_SLOW=1 uv run python -m unittest tests.test_stream_neural_svd -v
```

This command exercises the streaming tests including the slow recovery check against the exact all-starts operator spectrum at 2% relative tolerance; it is not a claim that a real-data run has converged.

For fast log-log spectrum verification without training, model downloads, or sample data, run:

```bash
uv run python -m unittest tests.test_analysis tests.test_exp4_neural_svd.Exp4AnalysisTests -v --durations 5
```

These tests cover power-law fits, learned component order, rank-one skipped fits, saved analysis artifacts, and plotting artists using small matrices and synthetic snapshots.
