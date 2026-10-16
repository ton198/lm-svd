# LM-SVD: GPT-2 Matrix Experiments

Build matrices from local GPT-2 inference, compute their singular value decompositions (SVDs), measure numerical rank, and plot singular-value decay and cumulative energy.

Both experiments use the same sampled history/future token sequences but construct different matrices:

| Experiment | Matrix entry | Matrix shape | Matrix file |
| --- | --- | --- | --- |
| `exp1` | Centered next-token logit for a future-specific header token | histories × (futures × top-k) | `matrix.npy` |
| `exp2` | Conditional log probability of an entire future given a history | histories × futures | `feature_log_prob.npy` |

Each experiment has four independent subcommands, run in order: `get_matrix`, `svd`, `rank`, `visualize`. Once a matrix exists, the analysis stages do not load a language model or need a GPU.

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

The output is `data/history_future_samples/wikitext-103-raw-v1-train-gpt2-history-future.parquet`. If it already exists, reuse it or explicitly pass `--overwrite` to rebuild it. Both experiments use this path by default. GPT-2 Small and Medium use the same GPT-2 token vocabulary, so the same sample file can be reused for comparisons between them.

The pipeline processes sorted shards and rows, encodes each text row independently, concatenates token IDs without inserting separators or EOS tokens, and samples distinct non-overlapping windows. Windows may cross source-row boundaries. Output columns are `sample_id`, `group`, `token_count`, `start_token`, `token_ids`, and `text`. Experiments use **`token_ids`**, not re-encoded text. Sampling settings and source information are stored in the Parquet schema's `lm_svd` metadata.

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
RESULT_DIR="data/experiment_results/exp1_L_M_table/$MODEL"

uv run exp1 get_matrix \
  --model-path "models/$MODEL" \
  --device cuda \
  --top-k 50 \
  --output-dir "$RESULT_DIR"

uv run exp1 svd --input-dir "$RESULT_DIR"
uv run exp1 rank --svd-dir "$RESULT_DIR"
uv run exp1 visualize --svd-dir "$RESULT_DIR"
```

`get_matrix` writes `matrix.npy`, `rows.parquet`, `columns.parquet`, and `metadata.json`. The Parquet files map rows and columns to sample IDs, header ranks, token IDs, and future-only centered logits.

### Small CPU smoke run

This uses GPT-2 Small and the leading samples from the same input file. `mktemp` creates a fresh directory so existing full results are not touched. The expected shape is `1 × 2`.

```bash
RESULT_DIR=$(mktemp -d /tmp/lm-svd-exp1.XXXXXX)
uv run exp1 get_matrix --device cpu \
  --history-count 1 --future-count 1 --top-k 2 \
  --output-dir "$RESULT_DIR"
uv run exp1 svd --input-dir "$RESULT_DIR"
uv run exp1 rank --svd-dir "$RESULT_DIR"
uv run exp1 visualize --svd-dir "$RESULT_DIR"
echo "Results: $RESULT_DIR"
```

## 5. Experiment 2: conditional future log-probability matrix

For each history/future pair, sum the conditional log probabilities of the future's tokens:

```text
B[history, future] = sum_t log p(future_t | history + future_<t)
```

These are log-probability **sums**, not averages, probabilities, or centered logits. Rows correspond to histories; columns correspond to futures.

### Full workflow

Set `MODEL` to `gpt2-small` or `gpt2-medium`. All histories and futures in the sample file are used. The default sample file produces a `512 × 512` matrix. Lower `--batch-size` if necessary; it controls how many futures are scored together for each history.

```bash
MODEL=gpt2-small
RESULT_DIR="data/experiment_results/exp2_PFH/$MODEL"

uv run exp2 get_matrix \
  --model-path "models/$MODEL" \
  --device cuda \
  --batch-size 64 \
  --output-dir "$RESULT_DIR"

uv run exp2 svd --input-dir "$RESULT_DIR"
uv run exp2 rank --svd-dir "$RESULT_DIR"
uv run exp2 visualize --svd-dir "$RESULT_DIR"
```

`get_matrix` writes `feature_log_prob.npy` and `metadata.json`. Keep the matrix filename unchanged when passing this directory to `exp2 svd`.

### Small CPU smoke run

The expected shape is `2 × 2`.

```bash
RESULT_DIR=$(mktemp -d /tmp/lm-svd-exp2.XXXXXX)
uv run exp2 get_matrix --device cpu \
  --history-count 2 --future-count 2 --batch-size 2 \
  --output-dir "$RESULT_DIR"
uv run exp2 svd --input-dir "$RESULT_DIR"
uv run exp2 rank --svd-dir "$RESULT_DIR"
uv run exp2 visualize --svd-dir "$RESULT_DIR"
echo "Results: $RESULT_DIR"
```

## 6. Understand the analysis outputs

Both experiments share the same analysis implementation:

| Stage | Reads | Writes |
| --- | --- | --- |
| `svd` | The experiment's matrix; optional sibling `metadata.json` | `svd_U.npy`, `svd_S.npy`, `svd_Vt.npy`, `svd_metadata.json` |
| `rank` | `svd_S.npy`, `svd_metadata.json` | `rank.json` |
| `visualize` | `svd_S.npy`, `rank.json` | `svd_spectrum.png` |

- **SVD:** Float64 thin SVD, with no additional centering. Reconstruct the stored matrix as `(U * S) @ Vt`. Metadata records input path, shape and dtype, computation dtype, singular-value extrema, Frobenius norm, relative reconstruction error, and the source matrix metadata under `matrix_metadata`. New SVD metadata does not contain rank fields.
- **Numerical rank:** Counts singular values strictly greater than `S.max() * max(matrix.shape) * finfo(input_dtype).eps`. The original matrix dtype determines epsilon, even though the SVD is computed in float64; non-floating input uses float64 epsilon. `rank.json` records the rank, tolerance, rule, matrix shape, input dtype, and singular-value count. Existing SVD metadata with legacy rank fields is accepted; rank is recomputed from the singular values.
- **Visualization:** `svd_spectrum.png` contains logarithmic singular-value decay with rank and tolerance markers, plus cumulative squared Frobenius energy with a rank marker. It uses Matplotlib's headless Agg backend. An all-zero spectrum is rejected because normalized energy is undefined. Existing manual `singular_value_decay.png` plots are untouched.

### Directory selection and existing results

**Analysis commands do not infer the model from your last `get_matrix` call.** If no directory options are supplied, they always use the GPT-2 Small result directory for that experiment. Always pass explicit directories for Medium, custom outputs, or smoke runs, as shown above.

`get_matrix` defaults to `models/gpt2-small` and writes under `data/experiment_results/<experiment>/<model-directory-name>/`. A different `--model-path` changes that default result directory. `svd --output-dir PATH` can write factors elsewhere; subsequent `rank` and `visualize` calls must use that output directory as `--svd-dir`.

Every stage refuses to replace its own existing outputs unless `--overwrite` is supplied. If a stage has already completed, reuse its files and start at the next stage. When regenerating a matrix, rerun **all downstream stages** to avoid mixing results from different inputs; overwriting one stage does not refresh the others.

Other useful options:

- `get_matrix --samples-path PATH`: use another history/future sample file.
- `get_matrix --history-count N --future-count N`: use only the leading N samples of each group; omit both to use all available samples.
- `exp1 get_matrix --top-k K`: select K header tokens per future; default 50.
- `exp2 get_matrix --batch-size N`: score N futures per inference batch; default 64.
- `get_matrix --device cpu`: use CPU instead of the default CUDA device.

Use `uv run exp1 <subcommand> --help` or `uv run exp2 <subcommand> --help` for available options. Module entry points also work: `uv run python -m experiments.exp1_L_M_table <subcommand>` and `uv run python -m experiments.exp2_PFH <subcommand>`.

## Python inference API

`CausalLMInference` loads a local model once in float32 and reuses it for token-ID inputs. It does not download files or tokenize text. CUDA is the default device; CPU is shown here for portability:

```python
from utils import CausalLMInference

inference = CausalLMInference("models/gpt2-medium", device="cpu")
logits = inference.next_token_logits([15496, 11])
scores = inference.continuation_log_probs(
    [15496, 11], [[995, 0]], batch_size=1
)
```

`next_token_logits` returns unnormalized logits as a CPU float32 tensor of shape `(vocab_size,)`. `continuation_log_probs` returns one summed conditional log probability per continuation as a CPU float64 tensor. Inputs must be non-empty integer token-ID sequences within the model vocabulary and context limit; continuations in one call must have equal lengths. Load models sequentially when memory is limited.

## Tests

```bash
uv run python -m unittest discover -s tests
```
