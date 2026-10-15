# Building history/future token windows

Download the pinned GPT-2 tokenizer files locally (no model weights):

```bash
hf download openai-community/gpt2 config.json tokenizer_config.json tokenizer.json vocab.json merges.txt --repo-type model --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --local-dir models/gpt2-small
```

Build the default table offline after the dependencies are installed:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run data-pipline build-windows --tokenizer-path models/gpt2-small
```

The CLI reads sorted local `samples/wikitext/<config>/<split>-*.parquet` shards. It GPT-2-encodes each `text` row independently, concatenates the token IDs in shard and row order, and inserts no separator or EOS token. Windows may cross row boundaries. `--tokenizer-path` must be a local tokenizer directory; generation does not download tokenizers. Counts, lengths, split, seed, and output can be changed with the command options; the default output is `data/history_future_samples/wikitext-103-raw-v1-train-gpt2-history-future.parquet`. `--overwrite` is required to replace an existing output.

The Parquet output contains `sample_id`, `group`, `token_count`, `start_token`, `token_ids`, and decoded `text`, in that order. History/future windows are sampled from exclusive fixed-width token slots and have distinct token sequences. `token_ids` are the authoritative experiment input; decoded text may not re-encode identically when a window starts inside a tokenized word boundary. Configuration and sampling values are recorded as JSON in the `lm_svd` schema metadata.

## Local GPT-2 models

Append Small weights to the tokenizer directory above, and download Medium's complete inference files at fixed revisions:

```bash
hf download openai-community/gpt2 model.safetensors generation_config.json --repo-type model --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --local-dir models/gpt2-small
hf download openai-community/gpt2-medium config.json generation_config.json tokenizer_config.json tokenizer.json vocab.json merges.txt model.safetensors --repo-type model --revision 6dcaa7a952f72f9298047fd5137cd6e4f05f41da --local-dir models/gpt2-medium
uv sync
```

PyTorch is a project runtime dependency. Both directories support offline `AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32)` and `AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)`; set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to enforce offline execution. Move the model and input tensors to CUDA for GPU inference.

Verified with PyTorch 2.14.1 / CUDA 13.0 on an RTX 5060: both models ran float32 CUDA forwards on the first history (64 tokens) and future (32 tokens) rows of the default table, with finite logits and causal-LM loss. Small has 124,439,808 parameters; Medium has 354,823,168. Load models sequentially and release each with `del model; torch.cuda.empty_cache()` to limit GPU memory use.

## Next-token logits

Import `CausalLMInference` from `utils` to load a local causal language model once and reuse it for token-ID sequences. CUDA is the default device; pass `"cpu"` for CPU inference. The model loads in float32 and locally only; no tokenizer or text input is used.

```python
from utils import CausalLMInference

inference = CausalLMInference("models/gpt2-small")
logits = inference.next_token_logits([15496, 11])
```

`next_token_logits` accepts one non-empty sequence of integer token IDs, no longer than `inference.max_length`, with IDs in `[0, inference.vocab_size)`. It returns the unnormalized next-token logits as a CPU float32 tensor of shape `(vocab_size,)`. The result is not softmax- or log-softmax-normalized.

## Experiment 1: centered-logit table

Build the default GPT-2 small table offline:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -m experiments.exp1_L_M_table
```

The command reads `data/history_future_samples/wikitext-103-raw-v1-train-gpt2-history-future.parquet` and writes `matrix.npy`, `rows.parquet`, `columns.parquet`, and `metadata.json` under `data/experiment_results/exp1_L_M_table/gpt2-small/`. `--history-count` and `--future-count` limit the leading samples; `--top-k` controls the number of future-specific header tokens. Use `--samples-path`, `--model-path`, `--device`, or `--output-dir` to override defaults. Existing output files are protected; pass `--overwrite` to replace them.

Each future's header is selected from future-only next-token logits. Matrix values are `logit(token | history + future) - mean(logits over the full vocabulary)`. `matrix.npy` rows correspond to histories; columns are grouped by future, then descending future-only logit rank. The Parquet index files map matrix coordinates to sample IDs, ranks, and token IDs.

## Experiment 2: future-feature log probabilities

Build a history-by-future matrix of conditional future log-probability sums:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -m experiments.exp2_PFH --device cuda
```

The command reads the same history/future sample table and local GPT-2 small model as Experiment 1. It writes `feature_log_prob.npy` and `metadata.json` under `data/experiment_results/exp2_PFH/gpt2-small/` by default. Rows correspond to histories and columns to futures; each value is the sum of the token conditional log probabilities `sum_t log p(future_t | history + future_<t>)`. Set `--device cpu` when CUDA is unavailable. `--batch-size` controls how many futures are scored together per history (default 64); reduce it to lower inference memory use. `--history-count`, `--future-count`, `--samples-path`, `--model-path`, and `--output-dir` select inputs and output location. Existing output files are protected; pass `--overwrite` to replace them. This is a separate experiment and does not change Experiment 1's centered-logit table or outputs.

### Rank and thin SVD

Compute the SVD of the stored matrix without additional centering:

```bash
uv run python -m experiments.exp1_L_M_table.svd
```

The default input is `data/experiment_results/exp1_L_M_table/gpt2-small/matrix.npy`. Outputs are `svd_U.npy`, `svd_S.npy`, `svd_Vt.npy`, and `svd_metadata.json` in the input directory. Use `--input-dir` and `--output-dir` to select other directories; `--overwrite` is required if any SVD output already exists. The table-generation command and its files are unchanged.

The shared `utils.matrix_svd` API computes float64 thin factors with NumPy; `utils.numerical_rank` uses `S.max() * max(matrix.shape) * finfo(input_dtype).eps`, with float64 epsilon for non-floating input. Thus a float32 input retains a float32-based rank threshold even though the factors are float64. `utils.MatrixSVD` holds the factors, rank, and threshold. Inputs must be finite, non-empty 2-D arrays. Metadata records the input shape and dtype, rank threshold, singular-value extrema, Frobenius norm, relative reconstruction error, and the source table metadata when present.