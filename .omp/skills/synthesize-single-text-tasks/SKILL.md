---
name: synthesize-single-text-tasks
description: Generate single-document task instructions one at a time with smol subagents, rejecting near-duplicates by embedding recall, and freeze them for task-single-text.
---

# Synthesize single-text tasks

Run from the repository root. Generate and review instructions exclusively with the built-in `sonic` subagent. This installation maps sonic to `@smol`, `openai-codex/gpt-6-luna` with low thinking. Verify a writer's `history://<id>` transcript records that model and thinking level before the full run. If sonic cannot resolve to smol, stop and report; do not substitute another model. The orchestrator must never write or review candidate instructions itself.

Use the Python eval kernel (project `.venv/bin/python`, with torch, transformers and huggingface_hub available), not a CLI. Set eval `timeout=0` for the generation loop. Read `xd://eval/agents` before first using its `agent`/`wait` interfaces. Calls are serial, one writer and, when appropriate, one reviewer per attempt. The helper does not launch checks or rebuild the pipeline.

## Rubric

```text
An instruction is ACCEPTABLE only if all hold:
1. It is a reusable task that operates on exactly ONE input: an arbitrary English prose document (a ~120-token web or news passage of any topic) that will follow it as "Document: ...".
2. It needs no other input: no question to answer, no answer options, no second text, no target word/entity/topic/letter supplied by a user, no list of items, no external data or tools.
3. It is well-defined for ANY such document regardless of topic or genre (do not assume the document is a review, recipe, story, email, dialogue, or code).
4. It refers to the input as "the document" or "the text" and does not mention specific named entities, topics, dates or quoted example content.
5. Any label set, output format, or length limit it uses is stated inside the instruction itself.
6. It is one or two sentences, 6-60 words, plain English, no placeholders, no line breaks, and does not contain the words "Instruction:", "Document:" or "Response:".
7. It does not ask for unsafe, deceptive, or harmful content.
```

## Writer prompt

```text
{rubric}

These kinds of instructions already exist (number of existing instructions per kind in parentheses):
{kinds}

Reference tasks sampled from other instruction datasets, for inspiration only. Many need extra inputs or a different kind of input, so do not copy them; instead you may adapt an operation from one of them to a single arbitrary document when that operation is not a listed kind:
{references}

{feedback}
Write ONE new ACCEPTABLE instruction of a kind that is not listed above. Do not write a paraphrase or a minor variant (different count, label wording, length limit, ordering or formatting) of any listed kind.
```

`kinds` contains the entire kind catalog formatted by `format_kinds(kinds)`, one `- K012 (7): <name>: <description>` per line sorted by id, or `- (none yet)` when empty. `references` is generated for each attempt by `sample_references(load_references(sni_headers, tasksource_parquet, flan_templates), attempt)`: 10 `- <text>` lines sampled from all candidates loaded directly from the raw SNI, tasksource and FLAN files, including tasks that are not single-text. Candidates are sorted by `(SOURCE_RANK[t.source], t.source_id)` before deduplication by `normalize_instruction`, keeping the first occurrence; whitespace is collapsed, and instructions longer than 80 words are truncated to 80 words with ` (truncated)` appended. Sampling uses `random.Random(attempt)`, seeded by the global attempt number for reproducible resumes. These writer-only references enter neither the Index nor the kind catalog, and the assignment map still does not enter the writer prompt. `feedback` includes all rejected attempts for the current item in attempt order, retaining only records with a string instruction (so `writer_failed` without candidate text is skipped). It is empty when no such records exist; otherwise:

```text
Your previous attempts for this instruction were all rejected. Do not repeat, paraphrase or make a minor variant of any of them:
1. [too_close, max similarity 0.83] <instruction> | closest existing: <nearest[0] text>
2. [word_count] <instruction>
```

Number entries from 1. Include `, max similarity {max_similarity:.2f}` only when similarity is non-null, and append ` | closest existing: {nearest[0][1]}` only when neighbors exist. Include only the first neighbor per failure; there is no limit on the number of failed attempts.

## Reviewer prompt

```text
{rubric}

A candidate is also unacceptable as "too_close" only if it is a paraphrase or minor variant of one specific neighbor below: it applies the same operation to the same aspect of the document and asks for the same kind of output, differing only in wording, counts, label wording, length limits, ordering or formatting. A candidate that analyzes a different aspect of the document, uses a substantively different label set, or produces a different kind of output is NOT too_close, even if it shares a general form such as choosing one label from a set and justifying it.

Candidate: {candidate}

Nearest existing instructions (cosine similarity):
{neighbors}

Judge the candidate. reason_code is "ok" when accepted.
```

`neighbors` has one `- 0.83 | <instruction>` per line.

## Structured outputs

```python
WRITER_SCHEMA = {"type": "object", "properties": {"instruction": {"type": "string"}}, "required": ["instruction"]}
REVIEW_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["accept", "reject"]}, "reason_code": {"type": "string", "enum": ["ok", "too_close", "extra_input", "self_contained", "genre_specific", "specific_content", "underspecified_output", "format", "unsafe"]}}, "required": ["verdict", "reason_code"]}
```

## Kind catalog

The librarian uses the same sonic subagent and serial call wrapper as the writer and reviewer. Its exact prompt and schema are:

```python
KIND_PROMPT = '''You maintain a catalog of KINDS of document-processing instructions. Split kinds whose operations differ. A kind is one operation applied to an arbitrary document plus one kind of output, described generally enough to cover paraphrases and minor variants (different counts, label wording, length limits, ordering or formatting), but never merging different operations or different output kinds.

Existing kinds:
{kinds}

Assign each numbered instruction below to an existing kind id when it performs the same operation with the same kind of output. Otherwise create a new kind with a temporary key N1, N2, ... (name at most 8 words, description at most 25 words, both general: no specific counts, labels or examples) and assign it there. Instructions in this batch may share a new kind.

Instructions:
{batch}'''
KIND_SCHEMA = {"type": "object", "properties": {
    "new_kinds": {"type": "array", "items": {"type": "object", "properties": {"key": {"type": "string"}, "name": {"type": "string"}, "description": {"type": "string"}}, "required": ["key", "name", "description"]}},
    "assignments": {"type": "array", "items": {"type": "object", "properties": {"index": {"type": "integer"}, "kind": {"type": "string"}}, "required": ["index", "kind"]}}},
    "required": ["new_kinds", "assignments"]}
```

`{kinds}` uses the same `format_kinds` representation as the writer. `{batch}` has one `<index>. <instruction>` per line, starting at 1 for each batch.

`RUN/kinds.json` stores general kind descriptions, counts and normalized instruction assignments:

```json
{"kinds": [{"id": "K001", "name": "...", "description": "...", "count": 3}],
 "assignments": {"<normalize_instruction(text)>": "K001"}}
```

`assignments` is resume state only and never enters a prompt. `count` is the number of reference instructions assigned to that kind. IDs follow creation order as `f"K{n:03d}"`.

`sync_kinds(run_dir, texts, call)` starts from `{"kinds": [], "assignments": {}}` when no file exists, then classifies normalized, previously unassigned instructions in original order, in batches of 20. Each batch calls `call(KIND_PROMPT.format(...), f"kinds-{len(kinds)+1:03d}-{start:04d}", KIND_SCHEMA)`, where `start` is the zero-based batch offset in the pending list.

The result must be a dict whose assignments cover exactly `1..len(batch)` once each. Each assigned kind must be an existing ID or a new key declared in that result; new keys must be unique `N1`, `N2`, etc. New names and descriptions must be nonempty strings of at most 8 and 25 words respectively. Only referenced new keys receive permanent IDs. Counts and normalized assignments are updated, then every successful batch writes `kinds.json.tmp` and atomically replaces `kinds.json`. A missing or invalid result retries the same batch at most three times total, then raises `RuntimeError(f"kind catalog update failed for batch starting {start}")` and stops the run. Already saved batches are not repeated on resume.

The catalog is synchronized after constructing the Index, before the generation loop, and immediately after every accepted instruction is added to the Index. Startup synchronization also repairs an interruption between accepting an instruction and catalog synchronization. Only the writer uses catalog context for generation; the reviewer still sees five real nearest instructions, and the similarity threshold and acceptance rules are unchanged.

## Deterministic helpers

Load the adjacent `synth.py` by `importlib.util.spec_from_file_location`; it is not a package. It imports `SOURCE_RANK`, `load_flan_tasks`, `load_sni_tasks`, `load_task_catalog`, `load_tasksource_tasks` and `normalize_instruction` from `data_pipline.task_inputs` after putting `src` on `sys.path`.

- `format_reason(raw)` returns the first failing reason in this order: `line_break` (contains `\n`); `word_count` (normalized whitespace, outside 6–60); `template_marker` (`r"\b(?:Instruction|Document|Response)\s*:"`, case-insensitive); `placeholder` (`r"[{}<>\[\]]|\.\.\.|_{2,}"`); `no_document_reference` (no `r"\b(?:document|text)\b"`, case-insensitive); `exact_duplicate` (same normalized instruction as a reference). Otherwise it returns `None`. The semantic rubric, including sentence count, is judged by sonic, not guessed by extra deterministic heuristics.
- `load_references(sni_headers: Path, tasksource_parquet: Path, flan_templates: Path) -> list[str]` reads all candidates from `load_sni_tasks`, `load_tasksource_tasks` and `load_flan_tasks`, without filtering by `reject_reason` or `format_class`. It stably sorts by `(SOURCE_RANK[t.source], t.source_id)` and deduplicates by `normalize_instruction`, preserving the first occurrence, then collapses whitespace with `" ".join(instruction.split()[:80])`. Instructions longer than 80 words receive the suffix ` (truncated)`. Missing source files naturally raise `FileNotFoundError` from their loader. These writer-only references do not enter the Index or kind catalog.
- `sample_references(references: list[str], attempt: int)` uses `random.Random(attempt).sample(references, min(10, len(references)))` and joins the sampled texts as `- <text>` lines. The global attempt seed makes sampling reproducible across resumes; an empty list returns `- (none)`.
- `load_task_catalog(catalog)` loads the frozen catalog's `id`, `source`, `category` and `instruction` fields as task candidates. `run` converts them back to catalog rows and excludes IDs already present in this run's `accepted.jsonl` to form its baseline (`base`). The baseline includes both original and previously frozen synthetic instructions; excluding this run's accepted IDs prevents double-counting on resume after freezing. It is not regenerated from the raw reference sources.
- `Embedder()` loads `sentence-transformers/all-MiniLM-L6-v2`, pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (Apache-2.0). If `models/all-MiniLM-L6-v2/model.safetensors` is missing it downloads that revision, allowing only `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`. `encode(texts)` runs on CPU, batches of 64, max length 128, attention-mask-weighted mean pooling and L2 normalization.
- `Index(texts, vectors)` holds reference texts/vectors and resets the module's normalized duplicate-reference set. `add(text, vec)` updates both. `nearest(vec, k=5)` returns descending `(cosine_similarity, instruction)` pairs. Use one active Index per loaded helper module; `run` constructs a fresh one including accepted instructions.

## Generation loop

`run(agent, *, wait=None, usage_reader=None, run_dir=Path("data/task_synthesis/run-1"), catalog=CATALOG, sni_headers=SNI_HEADERS, tasksource_parquet=TASKSOURCE_PARQUET, flan_templates=FLAN_TEMPLATES, target=1000, max_attempts=6000, freeze_output=True, embedder=None)` implements the full loop below. `target` is the number of newly accepted instructions for this run, not the catalog's total size. Default paths are:

- `CATALOG = Path("data/original/task_sources/single_text.jsonl")`
- `SNI_HEADERS = Path("data/original/natural_instructions/task_headers.jsonl")`
- `TASKSOURCE_PARQUET = Path("data/original/tasksource-instruct/data/test-00000-of-00001.parquet")`
- `FLAN_TEMPLATES = Path("data/original/flan/templates.py")`

1. **Prepare/resume.** Load `accepted.jsonl` and `attempts.jsonl` if the directory already exists. Load the frozen catalog and set `run_ids = {row["id"] for row in accepted}` and `base = [r for r in catalog_rows if r["id"] not in run_ids]`. Validate every synthetic baseline ID with `re.fullmatch(r"syn-(\d+)", id)`; otherwise raise `ValueError(f"synthetic task id is not syn-<number>: {id}")`. Set `offset` to the maximum baseline synthetic ID number, or 0 if there are none. Accepted IDs must be contiguous as `[f"syn-{offset + i:04d}" for i in range(1, len(accepted) + 1)]`. Embed `[r["instruction"] for r in base] + [row["instruction"] for row in accepted]` into the Index. Load writer references from the three raw source files. The attempts log is a write-ahead log: an accepted attempt is flushed before its accepted row, so resuming can recover an interrupted accepted append with the same offset-generated ID without regenerating that item. Each append is flushed and fsynced. Malformed JSON (including an incomplete final append) stops with its file and line number; preserve the valid prefix and repair that incomplete tail before resuming. Do not run concurrent orchestrators against the same directory.
   Synchronize `kinds.json` with every instruction in the Index before entering the loop, even when the target is already reached.
2. **Generate one.** While accepted count is below `target` and total attempts below `max_attempts`, let `n = len(accepted) + 1`. `item` and writer/reviewer labels retain run-local numbering, independent of the synthetic ID offset. Make at most five attempts per cycle. For each attempt:
   - Increment the global attempt count for this attempt. Give the writer the entire current catalog via `format_kinds(kinds)` and 10 reference tasks via `sample_references(references, total_attempts)`.
   - Call `agent(writer_prompt, agent="sonic", label=f"write-{n:04d}-{trial}", schema=WRITER_SCHEMA)` and wait for its parsed result. A missing result is rejected as `writer_failed`.
   - Run `format_reason`. If non-None, reject using its reason.
   - Otherwise encode the candidate and retrieve five nearest references. Reject a highest cosine similarity **>= 0.90** as `too_close`, without calling the reviewer.
   - Otherwise call `agent(reviewer_prompt, agent="sonic", label=f"review-{n:04d}-{trial}", schema=REVIEW_SCHEMA)` and wait. Accept `verdict="accept", reason_code="ok"`; reject other results with `reason_code`. A missing result is `reviewer_failed`.
   - Record every attempt in `attempts.jsonl`: `attempt`, `item`, `try`, `instruction`, `result` (`accepted`/`rejected`), `reason`, `max_similarity`, `nearest`. Similarity is null and neighbors empty if rejection happened before embedding.
   - On acceptance, append to `accepted.jsonl`: `id` (`f"syn-{offset + n:04d}"`), `instruction`, `max_similarity`, `nearest`, `attempts` (total tries on that item across cycles). Add the candidate and vector to the Index immediately, then synchronize `kinds.json` before generating another instruction.
   - On every writer call, include all failed candidates for the current item in the feedback above, never failures from other items. Accumulate across five-attempt cycles without clearing; rebuild feedback from `attempts.jsonl` on resume. Five-attempt cycles determine only try numbering. Labels follow the required item/try form and may repeat across cycles; handle IDs distinguish calls.
3. **Freeze.** On completion or reaching `max_attempts`, call `freeze(base, accepted, catalog=CATALOG)` with this run's catalog path. Write the unchanged baseline rows first, then each accepted instruction as `{"id": row["id"], "source": "synthetic", "category": "synthetic", "instruction": row["instruction"]}`. Write to the catalog's `.tmp` file and atomically replace the catalog. Re-entering the same run excludes its accepted IDs from the baseline and freezes the same rows without appending duplicates. If the cap is reached below `target`, freeze the actual count and explicitly report the shortfall and reason distribution. Interrupted/failed runs can be resumed; they do not freeze automatically before the loop returns. `freeze_output=False` disables freezing for a pilot. `freeze(base, accepted, catalog)` is also available separately.
4. **Report.** `run` returns `report(RUN)`: accepted count, total attempts, attempts per accepted item, rejection counts, median and nearest-rank 95th percentile accepted maximum similarity, the ten highest-similarity accepted records in `[0.85, 0.90)` with neighbors, and twenty accepted records sampled with `random.Random(0)`. It also includes `kinds` (total kind count) and `top_kinds` (the top ten kinds by descending count, projected to `id`, `name`, `count`); these default to `0` and `[]` if `kinds.json` does not exist. Include token totals and cost when available, and disclose incomplete usage.

## Eval entry points

Load once in the persistent Python kernel:

```python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "single_text_synth", ".omp/skills/synthesize-single-text-tasks/synth.py")
synth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth)
```

Before the pilot, the parent should exercise the deterministic helper smoke checks (format failures, duplicate state, catalog baseline and resume handling, raw-source references, Index ordering and normalized embeddings); this skill's implementation does not run checks implicitly.

Run the pilot in a **separate directory** with eval `timeout=0`:

```python
pilot_report = synth.run(
    agent, run_dir=Path("data/task_synthesis/pilot"), target=5,
    freeze_output=False)
display(pilot_report)
```

The default wait uses `handle.wait()`, which returns parsed schema JSON. Any `RuntimeError` from agent creation or waiting is recorded as a missing result, with its message in `usage.jsonl`'s `call_error` (null on success). This covers `schema?`, structured-output schema violations and aborted yields. Each missing result increments a consecutive-failure counter; any nonmissing result resets it. Ten consecutive missing results stop with `RuntimeError("10 consecutive sonic calls failed; last error: ...")`. Other agent/wait exception types propagate; usage-reader errors are logged separately. No model fallback is permitted. If adapting another interface, pass `wait=lambda handle: ...` returning that same parsed dict or None, not a list of results.

Inspect the pilot's five accepted rows and attempt records, confirm rejection/retry records, and inspect a writer transcript for the smol model. If the pilot contains no rejection, inspect further attempts rather than inventing a rejection record. Inspect high-similarity acceptances for obvious duplicates. If evidence warrants a threshold change, use one revised threshold consistently in `synth.py` and this skill before the formal run, not per-item exceptions. After confirming the pilot, remove only its throwaway directory; never clear the formal run to resume it.

Formal generation (eval `timeout=0`):

```python
summary = synth.run(agent)
display(summary)
```

Re-enter the same call to resume. For a deliberately paused intermediate run, use the same `run_dir` with a smaller target and `freeze_output=False`, then raise target to 1000 on resume. Do not accidentally freeze the pilot into the production catalog.

### Subagent usage

`usage.jsonl` records each completed call's label, handle ID, aggregate usage, `usage_error` and `call_error`. With `usage_reader=None`, `run` automatically uses `lambda h: usage_from_transcript(find_transcript(h.id))`.

`find_transcript(handle_id, sessions_root=Path.home() / ".omp/agent/sessions/-workspace-organization-ur_research-lm-svd")` searches `sessions_root.glob(f"*/{handle_id}.jsonl")` and selects the newest file by modification time, since handle IDs may recur in older sessions. No match raises `FileNotFoundError(f"no transcript for handle {handle_id} under {sessions_root}")`, which the call wrapper records in `usage_error`.

For another sessions root, supply an explicit reader:

```python
summary = synth.run(
    agent,
    usage_reader=lambda handle: synth.usage_from_transcript(
        synth.find_transcript(handle.id, sessions_root=Path("/actual/sessions/root"))))
```

`usage_from_transcript(path)` sums `message.role == "assistant"` messages' `usage.input`, `output`, `cacheRead`, `cacheWrite`, `totalTokens`, and `usage.cost.total`. No input/output double-counting is added to totalTokens. A custom callback may alternatively use usage fetched through the history tool and return the same aggregate dictionary. Usage-reader errors are logged, not counted as zero-cost calls. If any call lacks measured usage, totals are explicitly partial/unknown (`usage_complete=False`); never present them as the full token cost. Calls interrupted before their usage append may be absent even from the call count. After interruption, reconcile such handles with session history if a complete cost total is required.

After formal synthesis, the parent verifies catalog size, unique IDs and all-pairs cosine < 0.90, rebuilds the single-text dataset, and runs the approved plan's pipeline checks. Those are separate from this generation helper. Once synthesis is complete and the frozen catalog has been validated, delete that completed run's directory, including its attempts, accepted rows, usage and kind-catalog state. Preserve it while a run still needs to be resumed.
