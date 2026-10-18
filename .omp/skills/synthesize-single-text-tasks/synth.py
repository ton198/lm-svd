"""Deterministic support for the single-text synthesis skill; import by file path."""
from __future__ import annotations

import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

sys.path.insert(0, "src")
from data_pipline.task_inputs import (
    SOURCE_RANK,
    load_flan_tasks,
    load_sni_tasks,
    load_task_catalog,
    load_tasksource_tasks,
    normalize_instruction,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CATALOG = Path("data/original/task_sources/single_text.jsonl")
SNI_HEADERS = Path("data/original/natural_instructions/task_headers.jsonl")
TASKSOURCE_PARQUET = Path("data/original/tasksource-instruct/data/test-00000-of-00001.parquet")
FLAN_TEMPLATES = Path("data/original/flan/templates.py")
THRESHOLD = 0.90
REFERENCE_COUNT = 10
REFERENCE_WORDS = 80
REFERENCES: set[str] = set()
RUBRIC = '''An instruction is ACCEPTABLE only if all hold:
1. It is a reusable task that operates on exactly ONE input: an arbitrary English prose document (a ~120-token web or news passage of any topic) that will follow it as "Document: ...".
2. It needs no other input: no question to answer, no answer options, no second text, no target word/entity/topic/letter supplied by a user, no list of items, no external data or tools.
3. It is well-defined for ANY such document regardless of topic or genre (do not assume the document is a review, recipe, story, email, dialogue, or code).
4. It refers to the input as "the document" or "the text" and does not mention specific named entities, topics, dates or quoted example content.
5. Any label set, output format, or length limit it uses is stated inside the instruction itself.
6. It is one or two sentences, 6-60 words, plain English, no placeholders, no line breaks, and does not contain the words "Instruction:", "Document:" or "Response:".
7. It does not ask for unsafe, deceptive, or harmful content.'''
WRITER_PROMPT = '''{rubric}

These kinds of instructions already exist (number of existing instructions per kind in parentheses):
{kinds}

Reference tasks sampled from other instruction datasets, for inspiration only. Many need extra inputs or a different kind of input, so do not copy them; instead you may adapt an operation from one of them to a single arbitrary document when that operation is not a listed kind:
{references}

{feedback}
Write ONE new ACCEPTABLE instruction of a kind that is not listed above. Do not write a paraphrase or a minor variant (different count, label wording, length limit, ordering or formatting) of any listed kind.'''
REVIEWER_PROMPT = '''{rubric}

A candidate is also unacceptable as "too_close" only if it is a paraphrase or minor variant of one specific neighbor below: it applies the same operation to the same aspect of the document and asks for the same kind of output, differing only in wording, counts, label wording, length limits, ordering or formatting. A candidate that analyzes a different aspect of the document, uses a substantively different label set, or produces a different kind of output is NOT too_close, even if it shares a general form such as choosing one label from a set and justifying it.

Candidate: {candidate}

Nearest existing instructions (cosine similarity):
{neighbors}

Judge the candidate. reason_code is "ok" when accepted.'''
WRITER_SCHEMA = {"type": "object", "properties": {"instruction": {"type": "string"}}, "required": ["instruction"]}
REVIEW_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["accept", "reject"]}, "reason_code": {"type": "string", "enum": ["ok", "too_close", "extra_input", "self_contained", "genre_specific", "specific_content", "underspecified_output", "format", "unsafe"]}}, "required": ["verdict", "reason_code"]}
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


def format_kinds(catalog: dict) -> str:
    return "\n".join(
        f"- {kind['id']} ({kind['count']}): {kind['name']}: {kind['description']}"
        for kind in sorted(catalog["kinds"], key=lambda kind: kind["id"])
    ) or "- (none yet)"


def _valid_kind_result(result, batch_size: int, existing_ids: set[str]) -> bool:
    if not isinstance(result, dict):
        return False
    new_kinds, assignments = result.get("new_kinds"), result.get("assignments")
    if not isinstance(new_kinds, list) or not isinstance(assignments, list):
        return False
    new_keys = set()
    for kind in new_kinds:
        if not isinstance(kind, dict):
            return False
        key, name, description = kind.get("key"), kind.get("name"), kind.get("description")
        if not isinstance(key, str) or not re.fullmatch(r"N[1-9]\d*", key) or key in new_keys:
            return False
        if not isinstance(name, str) or not 1 <= len(name.split()) <= 8:
            return False
        if not isinstance(description, str) or not 1 <= len(description.split()) <= 25:
            return False
        new_keys.add(key)
    seen = set()
    for assignment in assignments:
        if not isinstance(assignment, dict):
            return False
        index, kind = assignment.get("index"), assignment.get("kind")
        if type(index) is not int or not 1 <= index <= batch_size or index in seen:
            return False
        if not isinstance(kind, str) or kind not in existing_ids | new_keys:
            return False
        seen.add(index)
    return len(seen) == batch_size


def sync_kinds(run_dir: Path, texts: list[str], call) -> dict:
    path = Path(run_dir) / "kinds.json"
    catalog = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"kinds": [], "assignments": {}}
    kinds, assignments = catalog["kinds"], catalog["assignments"]
    pending, seen = [], set(assignments)
    for text in texts:
        normalized = normalize_instruction(text)
        if normalized not in seen:
            pending.append(text)
            seen.add(normalized)
    for start in range(0, len(pending), 20):
        batch = pending[start:start + 20]
        existing_ids = {kind["id"] for kind in kinds}
        prompt = KIND_PROMPT.format(
            kinds=format_kinds(catalog),
            batch="\n".join(f"{index}. {text}" for index, text in enumerate(batch, 1)))
        for _ in range(3):
            result = call(prompt, f"kinds-{len(kinds)+1:03d}-{start:04d}", KIND_SCHEMA)
            if _valid_kind_result(result, len(batch), existing_ids):
                break
        else:
            raise RuntimeError(f"kind catalog update failed for batch starting {start}")
        referenced = {assignment["kind"] for assignment in result["assignments"]}
        new_ids = {}
        for kind in result["new_kinds"]:
            if kind["key"] in referenced:
                kind_id = f"K{len(kinds)+1:03d}"
                new_ids[kind["key"]] = kind_id
                kinds.append({"id": kind_id, "name": kind["name"],
                              "description": kind["description"], "count": 0})
        by_id = {kind["id"]: kind for kind in kinds}
        for assignment in result["assignments"]:
            kind_id = new_ids.get(assignment["kind"], assignment["kind"])
            assignments[normalize_instruction(batch[assignment["index"] - 1])] = kind_id
            by_id[kind_id]["count"] += 1
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(catalog, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    return catalog


def format_reason(raw: str) -> str | None:
    if "\n" in raw:
        return "line_break"
    s = " ".join(raw.split())
    if not 6 <= len(s.split()) <= 60:
        return "word_count"
    if re.search(r"\b(?:Instruction|Document|Response)\s*:", s, re.I):
        return "template_marker"
    if re.search(r"[{}<>\[\]]|\.\.\.|_{2,}", s):
        return "placeholder"
    if not re.search(r"\b(?:document|text)\b", s, re.I):
        return "no_document_reference"
    if normalize_instruction(s) in REFERENCES:
        return "exact_duplicate"
    return None


def load_references(sni_headers: Path, tasksource_parquet: Path, flan_templates: Path) -> list[str]:
    candidates = (
        load_sni_tasks(sni_headers)
        + load_tasksource_tasks(tasksource_parquet)
        + load_flan_tasks(flan_templates)
    )
    references, seen = [], set()
    for task in sorted(candidates, key=lambda task: (SOURCE_RANK[task.source], task.source_id)):
        key = normalize_instruction(task.instruction)
        if key in seen:
            continue
        seen.add(key)
        words = task.instruction.split()
        text = " ".join(words[:REFERENCE_WORDS])
        if len(words) > REFERENCE_WORDS:
            text += " (truncated)"
        references.append(text)
    return references


def sample_references(references: list[str], attempt: int) -> str:
    return "\n".join(
        f"- {text}" for text in random.Random(attempt).sample(
            references, min(REFERENCE_COUNT, len(references)))
    ) if references else "- (none)"


class Embedder:
    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        path = Path("models/all-MiniLM-L6-v2")
        if not (path / "model.safetensors").is_file():
            snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=str(path),
                              allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"])
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to("cpu").eval()

    def encode(self, texts: list[str]):
        torch = self.torch
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                batch = self.tokenizer(texts[start:start + 64], padding=True, truncation=True,
                                       max_length=128, return_tensors="pt")
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                chunks.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        return torch.cat(chunks) if chunks else torch.empty((0, self.model.config.hidden_size))


class Index:
    """One active reference universe; constructing an Index resets format_reason's set."""
    def __init__(self, texts=None, vectors=None):
        self.texts = list(texts or [])
        self.vectors = vectors
        if vectors is not None and len(vectors) != len(self.texts):
            raise ValueError("reference text/vector count mismatch")
        if self.texts and vectors is None:
            raise ValueError("reference texts require vectors")
        REFERENCES.clear()
        REFERENCES.update(map(normalize_instruction, self.texts))

    def add(self, text, vec):
        import torch
        vec = vec.detach().cpu().reshape(1, -1)
        self.vectors = vec if self.vectors is None else torch.cat((self.vectors, vec))
        self.texts.append(text)
        REFERENCES.add(normalize_instruction(text))

    def nearest(self, vec, k=5) -> list[tuple[float, str]]:
        if not self.texts or k <= 0:
            return []
        scores = self.vectors @ vec.detach().cpu().reshape(-1)
        values, indices = scores.topk(min(k, len(self.texts)), sorted=True)
        return [(float(score), self.texts[int(i)]) for score, i in zip(values, indices)]


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{n}: invalid JSON; repair interrupted final append before resuming") from exc
    return rows


def _append(path: Path, row: dict):
    import os
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def find_transcript(handle_id: str, sessions_root: Path = Path.home() / ".omp/agent/sessions/-workspace-organization-ur_research-lm-svd") -> Path:
    matches = sorted(sessions_root.glob(f"*/{handle_id}.jsonl"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"no transcript for handle {handle_id} under {sessions_root}")
    return matches[-1]


def usage_from_transcript(path: Path) -> dict:
    """Aggregate assistant usage in a handle's JSONL session transcript, if provided."""
    totals = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0, "cost": 0.0}
    messages = 0
    for row in _read_rows(Path(path)):
        message = row.get("message", {})
        usage = message.get("usage")
        if message.get("role") != "assistant" or not usage:
            continue
        messages += 1
        for key in ("input", "output", "cacheRead", "cacheWrite", "totalTokens"):
            totals[key] += usage.get(key, 0)
        totals["cost"] += (usage.get("cost") or {}).get("total", 0)
    return {**totals, "messages": messages} if messages else {}


def freeze(base: list[dict], accepted: list[dict], catalog=CATALOG):
    catalog = Path(catalog)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    temporary = catalog.with_name(catalog.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in base:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in accepted:
            stream.write(json.dumps({
                "id": row["id"], "source": "synthetic", "category": "synthetic",
                "instruction": row["instruction"],
            }, ensure_ascii=False) + "\n")
    temporary.replace(catalog)


def report(run: Path) -> dict:
    run = Path(run)
    accepted = _read_rows(run / "accepted.jsonl")
    attempts = _read_rows(run / "attempts.jsonl")
    usage = _read_rows(run / "usage.jsonl")
    kinds_path = run / "kinds.json"
    kinds = json.loads(kinds_path.read_text(encoding="utf-8"))["kinds"] if kinds_path.exists() else []
    similarities = sorted(row["max_similarity"] for row in accepted if row["max_similarity"] is not None)
    high = sorted((row for row in accepted if row["max_similarity"] is not None and 0.85 <= row["max_similarity"] < THRESHOLD), key=lambda row: row["max_similarity"], reverse=True)[:10]
    measured = [row["usage"] for row in usage if row.get("usage")]
    return {
        "accepted": len(accepted), "total_attempts": len(attempts),
        "kinds": len(kinds),
        "top_kinds": [{key: kind[key] for key in ("id", "name", "count")}
                      for kind in sorted(kinds, key=lambda kind: kind["count"], reverse=True)[:10]],
        "attempts_per_accepted": len(attempts) / len(accepted) if accepted else None,
        "rejections": dict(Counter(row["reason"] for row in attempts if row["result"] == "rejected")),
        "max_similarity_median": statistics.median(similarities) if similarities else None,
        "max_similarity_p95": similarities[max(0, math.ceil(0.95 * len(similarities)) - 1)] if similarities else None,
        "highest_borderline": high,
        "sample": random.Random(0).sample(accepted, min(20, len(accepted))),
        "usage": {key: sum(row.get(key, 0) for row in measured) for key in ("input", "output", "cacheRead", "cacheWrite", "totalTokens", "cost")},
        "calls_logged": len(usage), "calls_with_usage": len(measured),
        "usage_complete": bool(usage) and len(measured) == len(usage),
        "usage_note": "Usage is partial/unknown unless every call has measured usage; interrupted calls before logging may be missing.",
    }


def run(agent: Callable, *, wait: Callable | None = None, usage_reader: Callable | None = None,
        run_dir=Path("data/task_synthesis/run-1"), catalog=CATALOG,
        sni_headers=SNI_HEADERS, tasksource_parquet=TASKSOURCE_PARQUET,
        flan_templates=FLAN_TEMPLATES, target=1000, max_attempts=6000,
        freeze_output=True, embedder=None) -> dict:
    """Serial eval orchestration. agent returns a handle; wait(handle) returns parsed JSON.

    Defaults to handle.wait() and automatic transcript lookup for usage. RuntimeError
    from agent/wait becomes a missing result and is logged; ten consecutive missing
    results stop the run. Other agent/wait exceptions propagate. Usage-reader errors
    are logged without stopping the run. No model fallback is permitted. Target is
    the number of new instructions accepted in this run. Set target=5,
    freeze_output=False and a separate run_dir for a pilot; re-enter to resume.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    accepted_path, attempts_path = run_dir / "accepted.jsonl", run_dir / "attempts.jsonl"
    accepted, attempts = _read_rows(accepted_path), _read_rows(attempts_path)
    catalog_rows = [
        {"id": task.source_id, "source": task.source, "category": task.category,
         "instruction": task.instruction}
        for task in load_task_catalog(catalog)
    ]
    run_ids = {row["id"] for row in accepted}
    base = [row for row in catalog_rows if row["id"] not in run_ids]
    numbers = []
    for row in base:
        if row["source"] == "synthetic":
            match = re.fullmatch(r"syn-(\d+)", row["id"])
            if match is None:
                raise ValueError(f"synthetic task id is not syn-<number>: {row['id']}")
            numbers.append(int(match.group(1)))
    offset = max(numbers, default=0)
    # Attempts is the write-ahead log. Recover an accepted attempt whose accepted
    # append was interrupted, without invoking the writer or reviewer again.
    for record in attempts:
        if record["result"] == "accepted" and record["item"] > len(accepted):
            if record["item"] != len(accepted) + 1:
                raise ValueError("noncontiguous accepted item in attempts log")
            recovered = {"id": f"syn-{offset + record['item']:04d}", "instruction": record["instruction"],
                         "max_similarity": record["max_similarity"], "nearest": record["nearest"],
                         "attempts": sum(a["item"] == record["item"] for a in attempts if a["attempt"] <= record["attempt"])}
            _append(accepted_path, recovered)
            accepted.append(recovered)
    if [row["id"] for row in accepted] != [f"syn-{offset + i:04d}" for i in range(1, len(accepted) + 1)]:
        raise ValueError("accepted IDs are not contiguous")
    references = load_references(sni_headers, tasksource_parquet, flan_templates)
    embedder = embedder if embedder is not None else Embedder()
    texts = [row["instruction"] for row in base] + [row["instruction"] for row in accepted]
    index = Index(texts, embedder.encode(texts))
    total_attempts = max((row["attempt"] for row in attempts), default=0)
    consecutive_failures = 0
    if usage_reader is None:
        usage_reader = lambda h: usage_from_transcript(find_transcript(h.id))

    def call(prompt, label, schema):
        nonlocal consecutive_failures
        handle = None
        call_error = None
        try:
            handle = agent(prompt, agent="sonic", label=label, schema=schema)
            result = wait(handle) if wait is not None else handle.wait()
        except RuntimeError as exc:
            call_error = str(exc)
            result = None
        measured, error = None, None
        if usage_reader is not None:
            try:
                measured = usage_reader(handle)
            except Exception as exc:
                error = str(exc)
        _append(run_dir / "usage.jsonl", {"label": label, "handle": str(getattr(handle, "id", "")),
                                         "usage": measured, "usage_error": error, "call_error": call_error})
        consecutive_failures = consecutive_failures + 1 if result is None else 0
        if consecutive_failures >= 10:
            raise RuntimeError(f"10 consecutive sonic calls failed; last error: {call_error}")
        return result
    kinds = sync_kinds(run_dir, index.texts, call)

    while len(accepted) < target and total_attempts < max_attempts:
        n = len(accepted) + 1
        prior = [row for row in attempts if row["item"] == n]
        last = prior[-1] if prior else None
        start_try = last["try"] + 1 if last and last["try"] < 5 else 1
        for trial in range(start_try, 6):
            if total_attempts >= max_attempts:
                break
            total_attempts += 1
            feedback = _feedback([row for row in attempts if row["item"] == n])
            writer = call(WRITER_PROMPT.format(rubric=RUBRIC, kinds=format_kinds(kinds),
                          references=sample_references(references, total_attempts), feedback=feedback),
                          f"write-{n:04d}-{trial}", WRITER_SCHEMA)
            candidate = writer.get("instruction") if isinstance(writer, dict) else None
            nearest, similarity, vec = [], None, None
            reason = "writer_failed" if not isinstance(candidate, str) else format_reason(candidate)
            if reason is None:
                vec = embedder.encode([candidate])[0]
                nearest = index.nearest(vec, k=5)
                similarity = nearest[0][0] if nearest else None
                if similarity is not None and similarity >= THRESHOLD:
                    reason = "too_close"
                else:
                    review = call(REVIEWER_PROMPT.format(rubric=RUBRIC, candidate=candidate,
                                  neighbors="\n".join(f"- {score:.2f} | {text}" for score, text in nearest)),
                                  f"review-{n:04d}-{trial}", REVIEW_SCHEMA)
                    if not isinstance(review, dict):
                        reason = "reviewer_failed"
                    elif review.get("verdict") == "accept" and review.get("reason_code") == "ok":
                        reason = None
                    else:
                        reason = review.get("reason_code") or "reviewer_failed"
                        if reason == "ok":
                            reason = "reviewer_failed"
            record = {"attempt": total_attempts, "item": n, "try": trial, "instruction": candidate,
                      "result": "accepted" if reason is None else "rejected", "reason": reason or "ok",
                      "max_similarity": similarity, "nearest": nearest}
            _append(attempts_path, record)
            attempts.append(record)
            if reason is None:
                row = {"id": f"syn-{offset + n:04d}", "instruction": candidate, "max_similarity": similarity,
                       "nearest": nearest, "attempts": sum(a["item"] == n for a in attempts)}
                _append(accepted_path, row)
                accepted.append(row)
                index.add(candidate, vec)
                kinds = sync_kinds(run_dir, index.texts, call)
                break
    if freeze_output:
        freeze(base, accepted, catalog)
    return report(run_dir)


def _feedback(failures: list[dict]) -> str:
    failures = [row for row in failures
                if row["result"] == "rejected" and isinstance(row["instruction"], str)]
    if not failures:
        return ""
    lines = ["Your previous attempts for this instruction were all rejected. Do not repeat, paraphrase or make a minor variant of any of them:"]
    for number, row in enumerate(failures, 1):
        reason = row["reason"]
        if row["max_similarity"] is not None:
            reason = f"{reason}, max similarity {row['max_similarity']:.2f}"
        line = f"{number}. [{reason}] {row['instruction']}"
        if row["nearest"]:
            line += f" | closest existing: {row['nearest'][0][1]}"
        lines.append(line)
    return "\n".join(lines)
