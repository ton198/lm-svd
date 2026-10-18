from __future__ import annotations

import ast
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, MutableMapping
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .io import write_parquet_atomic


SOURCE_RANK = {"sni": 0, "tasksource": 1, "flan": 2, "synthetic": 3}


@dataclass
class TaskCandidate:
    source: str
    source_id: str
    category: str
    instruction: str
    format_class: str
    format_reason: str | None
    reject_reason: str | None = None
    history_tokens: int | None = None


@dataclass
class Passage:
    url: str
    domain: str
    text: str
    passage_tokens: int


def is_prose(s: str) -> bool:
    return (len(s.split()) >= 5
            and sum(c.isalpha() for c in s) / max(1, len(s)) > 0.7
            and not re.search(r"[\[\]{}=<>]|\d{2,}/\d", s))


FIELD_LABEL = re.compile(r"(?:^|\n)\s*(?:[A-Z][\w \-\(\)]{0,25}|[a-z][\w ]{0,15})\s*:\s", re.M)

STRUCTURE_DEFINITION = re.compile(
    r"\b(two (?:sentences|texts|passages|statements|phrases|words)|sentence ?[12]\b|pair of|answer options"
    r"|(?:four|three|two|following) options|choices|list of (?:numbers|integers|words|lists)|blank|table"
    r"|dialog(?:ue)?s?\b|conversation|premise and|hypothesis|question and (?:an? )?answer"
    r"|given (?:a |an |the )?(?:question|word|number|integer|list|pair|title|date|year|name of)s?\b)", re.I)

STRUCTURED_CATEGORIES = frozenset({
    "Program Execution", "Mathematics", "Number Conversion", "Text to Code", "Code to Text",
    "Word Analogy", "Word Semantics", "Word Relation Classification", "Entity Generation",
    "Data to Text", "Dialogue Generation", "Dialogue Act Recognition", "Dialogue State Tracking",
    "Speaker Identification", "Speaker Relation Classification", "Overlap Extraction",
    "Sentence Ordering", "Question Answering", "Answer Verification", "Answerability Classification",
    "Question Understanding", "Question Rewriting", "Question Decomposition",
    "Wrong Candidate Generation", "Fill in The Blank", "Textual Entailment", "Text Matching",
    "Coreference Resolution", "Commonsense Classification", "Negotiation Strategy Detection",
    "Intent Identification", "Cause Effect Classification", "Explanation", "Fact Verification",
    "Preposition Prediction", "Discourse Connective Identification",
    "Discourse Relation Classification", "Entity Relation Classification", "Sentence Perturbation"})


def normalize_instruction(s: str) -> str:
    return " ".join(re.findall(r"\w+", s.lower()))


def load_sni_tasks(headers_path: Path | str) -> list[TaskCandidate]:
    tasks: list[TaskCandidate] = []
    translations: dict[tuple[str, ...], list[TaskCandidate]] = defaultdict(list)
    with Path(headers_path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            instruction = " ".join(d.strip() for d in record["Definition"]).strip()
            categories = record["Categories"]
            example = record["positive_example_input"]
            if any(category in STRUCTURED_CATEGORIES for category in categories):
                format_reason = "structured_category"
            elif FIELD_LABEL.search(example) or "___" in example:
                format_reason = "multi_field_example"
            elif not is_prose(example):
                format_reason = "non_prose_example"
            elif STRUCTURE_DEFINITION.search(instruction):
                format_reason = "definition_requires_structure"
            else:
                format_reason = None
            task = TaskCandidate(
                source="sni",
                source_id=record["name"],
                category=categories[0],
                instruction=instruction,
                format_class="single_text" if format_reason is None else "structured",
                format_reason=format_reason,
            )
            if (record["Input_language"] != ["English"]
                    or record["Instruction_language"] != ["English"]):
                task.reject_reason = "non_english"
            elif "Translation" in categories:
                translations[tuple(record["Output_language"])].append(task)
            tasks.append(task)
    for group in translations.values():
        for task in sorted(group, key=lambda task: task.source_id)[1:]:
            task.reject_reason = "duplicate_translation_language"
    return tasks


def load_tasksource_tasks(parquet_path: Path | str) -> list[TaskCandidate]:
    table = pq.read_table(parquet_path, columns=["task", "inputs"])
    grouped: dict[str, list[str]] = defaultdict(list)
    for task, inputs in zip(table.column("task").to_pylist(), table.column("inputs").to_pylist()):
        grouped[task].append(inputs)

    tasks: list[TaskCandidate] = []
    for source_id, inputs in grouped.items():
        heads = [text.split("\n", 1)[0] for text in inputs]
        # Counter preserves first appearance, including ties for most_common.
        head = Counter(heads).most_common(1)[0][0]
        body = inputs[0].partition("\n")[2]
        if ("text_A" in inputs[0]
                or "label each line" in head
                or "chose the best option" in head):
            format_reason = "multi_field_example"
        elif not is_prose(body):
            format_reason = "non_prose_example"
        else:
            format_reason = None
        task = TaskCandidate(
            source="tasksource",
            source_id=source_id,
            category="tasksource",
            instruction=head.strip(),
            format_class="single_text" if format_reason is None else "structured",
            format_reason=format_reason,
        )
        label_sets = [frozenset(label for label in re.findall(r'"([^"]*)"', h) if label)
                      for h in heads]
        labels = sorted(set().union(*label_sets), key=lambda label: (label.casefold(), label))
        if any("With no explanation" not in h for h in heads):
            task.reject_reason = "no_instruction_line"
        elif len(set(label_sets)) > 1 and len(labels) > 8:
            task.reject_reason = "label_subsampled"
        elif len(labels) < 2:
            task.reject_reason = "unparseable_labels"
        else:
            markers = ("with either ", "option from ")
            positions = [(head.rfind(marker), marker) for marker in markers]
            position, marker = max(positions, key=lambda item: item[0])
            if position < 0:
                task.reject_reason = "unparseable_labels"
            else:
                quoted = [f'"{label}"' for label in labels]
                rendered_labels = ", ".join(quoted[:-1]) + " or " + quoted[-1]
                task.instruction = head[:position + len(marker)] + rendered_labels + "."
        tasks.append(task)
    return tasks


def load_flan_tasks(templates_path: Path | str) -> list[TaskCandidate]:
    tree = ast.parse(Path(templates_path).read_text(encoding="utf-8"))
    patterns = None
    for statement in tree.body:
        if (isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "PATTERNS"
                        for target in statement.targets)):
            patterns = ast.literal_eval(statement.value)
            break
    if patterns is None:
        raise ValueError(f"PATTERNS assignment not found in {templates_path}")

    allowed_fields = {"text", "context", "sentence", "passage", "article", "document"}
    tasks: list[TaskCandidate] = []
    for dataset, templates in patterns.items():
        for index, (input_pattern, _target_pattern) in enumerate(templates):
            placeholders = list(re.finditer(r"\{(\w+)\}", input_pattern))
            if len(placeholders) != 1 or placeholders[0].group(1) not in allowed_fields:
                continue
            instruction = input_pattern[:placeholders[0].start()].strip()
            if len(instruction.split()) < 3:
                continue
            tasks.append(TaskCandidate(
                source="flan",
                source_id=f"{dataset}#{index}",
                category="flan",
                instruction=instruction,
                format_class="single_text",
                format_reason=None,
            ))
            break
    return tasks


def load_task_catalog(path: Path | str) -> list[TaskCandidate]:
    tasks: list[TaskCandidate] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for name in ("id", "source", "category", "instruction"):
                if name not in row:
                    raise ValueError(f"{path}:{line_number}: missing field {name}")
            source = row["source"]
            if source not in SOURCE_RANK:
                raise ValueError(f"{path}:{line_number}: unknown task source {source}")
            source_id = row["id"]
            if source_id in seen:
                raise ValueError(f"duplicate task id: {source_id}")
            seen.add(source_id)
            tasks.append(TaskCandidate(
                source=row["source"],
                source_id=source_id,
                category=row["category"],
                instruction=row["instruction"],
                format_class="single_text",
                format_reason=None,
            ))
    return tasks


def finalize_tasks(
    candidates: Iterable[TaskCandidate], tokenizer, history_max_tokens: int = 256,
) -> list[TaskCandidate]:
    """Deduplicate and count histories in place; return all candidates in source order."""
    tasks = sorted(candidates, key=lambda task: (SOURCE_RANK[task.source], task.source_id))
    seen: set[str] = set()
    for task in tasks:
        if task.reject_reason is not None:
            continue
        normalized = normalize_instruction(task.instruction)
        if normalized in seen:
            task.reject_reason = "duplicate_instruction"
            continue
        seen.add(normalized)
        task.history_tokens = len(tokenizer.encode(
            f"Instruction: {task.instruction}", add_special_tokens=False,
        ))
        if task.history_tokens > history_max_tokens:
            task.reject_reason = "history_too_long"
    return tasks


def order_by_bucket(tasks: Iterable[TaskCandidate], seed: int) -> list[TaskCandidate]:
    """Shuffle each category independently, then interleave categories round-robin."""
    buckets: dict[str, list[TaskCandidate]] = defaultdict(list)
    for task in tasks:
        buckets[task.category].append(task)
    shuffled: list[list[TaskCandidate]] = []
    for index, category in enumerate(sorted(buckets)):
        bucket = buckets[category]
        order = np.random.default_rng(seed + index).permutation(len(bucket))
        shuffled.append([bucket[int(position)] for position in order])
    ordered: list[TaskCandidate] = []
    round_index = 0
    active = shuffled
    while active:
        remaining = []
        for bucket in active:
            ordered.append(bucket[round_index])
            if len(bucket) > round_index + 1:
                remaining.append(bucket)
        active = remaining
        round_index += 1
    return ordered


def select_single_text_tasks(
    tasks: Iterable[TaskCandidate], seed: int = 0,
) -> list[TaskCandidate]:
    """Return all surviving single-text tasks in balanced category order."""
    return order_by_bucket(
        [task for task in tasks
         if task.reject_reason is None and task.format_class == "single_text"],
        seed,
    )


_PASSAGE_LINE_END = re.compile(r"""[.!?]["')]?$""")
_PASSAGE_FORBIDDEN = re.compile(
    r"http|www\.|@|©|cookie|\{|\}|\||click here|subscribe", re.I,
)
_PASSAGE_REPEATED_PUNCTUATION = re.compile(r"[!?]{2,}")


def select_passages(
    c4_paths: Iterable[Path | str],
    count_tokens: Callable[[str], int],
    input_count: int,
    min_tokens: int = 96,
    max_tokens: int = 160,
    seed: int = 0,
    *,
    rejection_counts: MutableMapping[str, int] | None = None,
) -> list[Passage]:
    """Select seeded, unique-domain C4 passages satisfying the prose rules.

    Each rejected document contributes its first failing rule to
    ``rejection_counts``, when supplied. Counts accumulate into that mapping,
    including when too few passages are accepted and a ValueError is raised.
    Rule order is token bounds, prose filters, text dedup, then domain dedup.
    """
    counts = rejection_counts if rejection_counts is not None else Counter()
    accepted: list[Passage] = []
    seen_texts: set[str] = set()
    seen_domains: set[str] = set()
    for path in c4_paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for record_line in stream:
                record = json.loads(record_line)
                lines = [line.strip() for line in record["text"].splitlines() if line.strip()]
                selected: list[str] = []
                passage_tokens = 0
                text = ""
                for line in lines:
                    selected.append(line)
                    text = "\n".join(selected)
                    passage_tokens = count_tokens(text)
                    if passage_tokens >= min_tokens:
                        break

                reason = None
                if passage_tokens < min_tokens:
                    reason = "short"
                elif passage_tokens > max_tokens:
                    reason = "long"
                elif not (text[0].isupper() or text[0].isdigit()):
                    reason = "fragment_start"
                elif not all(_PASSAGE_LINE_END.search(line) for line in selected):
                    reason = "incomplete_line"
                elif any(len(line.split()) < 4 for line in selected):
                    reason = "short_line"
                elif _PASSAGE_FORBIDDEN.search(text):
                    reason = "forbidden_content"
                elif sum(not char.isascii() for char in text) > 0.01 * len(text):
                    reason = "non_ascii"
                else:
                    letters = sum(char.isalpha() for char in text)
                    if letters < 0.75 * len(text):
                        reason = "low_alphabetic"
                    elif sum(char.isupper() for char in text) > 0.15 * letters:
                        reason = "uppercase"
                    elif _PASSAGE_REPEATED_PUNCTUATION.search(text):
                        reason = "repeated_punctuation"

                if reason is not None:
                    counts[reason] = counts.get(reason, 0) + 1
                    continue

                normalized_text = " ".join(text.split())
                domain = urlparse(record["url"]).netloc.lower().removeprefix("www.")
                if normalized_text in seen_texts:
                    reason = "duplicate_text"
                elif domain in seen_domains:
                    reason = "duplicate_domain"
                if reason is not None:
                    counts[reason] = counts.get(reason, 0) + 1
                    continue

                seen_texts.add(normalized_text)
                seen_domains.add(domain)
                accepted.append(Passage(record["url"], domain, text, passage_tokens))

    if len(accepted) < input_count:
        raise ValueError(
            f"only {len(accepted)} passages accepted; add C4 shards via --c4-paths"
        )
    order = np.random.default_rng(seed).permutation(len(accepted))
    return [accepted[int(index)] for index in order[:input_count]]


_HISTORY_TEMPLATE = "Instruction: {instruction}"
_FUTURE_TEMPLATE = "\n\nDocument: {passage}\n\nResponse:"
_C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"


def build_task_input_tables(
    *,
    task_catalog: Path = Path("data/original/task_sources/single_text.jsonl"),
    c4_paths: Iterable[Path] = (
        Path("data/original/c4/en/c4-validation.00000-of-00008.json.gz"),
    ),
    tokenizer_path: Path = Path("models/gpt2-small"),
    input_count: int = 1309,
    history_max_tokens: int = 256,
    passage_min_tokens: int = 96,
    passage_max_tokens: int = 160,
    seed: int = 0,
    output_dir: Path = Path("data/history_future_samples"),
    overwrite: bool = False,
) -> dict:
    """Build single-text task histories with C4 document futures."""
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)
    output = output_dir / "task-single-text.parquet"
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )
    for name, value in (
        ("input_count", input_count),
        ("history_max_tokens", history_max_tokens),
        ("passage_min_tokens", passage_min_tokens),
        ("passage_max_tokens", passage_max_tokens),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if passage_min_tokens > passage_max_tokens:
        raise ValueError("passage_min_tokens must not exceed passage_max_tokens")
    task_catalog = Path(task_catalog)
    if not task_catalog.is_file():
        raise ValueError(f"task catalog does not exist: {task_catalog}")
    c4_paths = [Path(path) for path in c4_paths]
    if not c4_paths:
        raise ValueError("at least one C4 path is required")
    for path in c4_paths:
        if not path.is_file():
            raise ValueError(
                f"input file does not exist: {path}; run scripts/download_task_sources.py"
            )
    if not tokenizer_path.is_dir():
        raise ValueError(f"tokenizer directory does not exist: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), use_fast=True, local_files_only=True,
        )
    except OSError as error:
        raise ValueError(
            f"cannot load local tokenizer from {tokenizer_path}: {error}"
        ) from error

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    candidates = finalize_tasks(load_task_catalog(task_catalog), tokenizer, history_max_tokens)
    counts = Counter(
        task.reject_reason for task in candidates if task.reject_reason is not None
    )
    if counts:
        raise ValueError(
            f"task catalog contains rejected tasks: {json.dumps(dict(sorted(counts.items())))}"
        )
    single_text = select_single_text_tasks(candidates, seed)
    if not single_text:
        raise ValueError("no single-text tasks survived the harvesting rules")
    passage_reject_counts: Counter = Counter()
    passages = select_passages(
        c4_paths, lambda text: len(encode(text)), input_count,
        passage_min_tokens, passage_max_tokens, seed,
        rejection_counts=passage_reject_counts,
    )
    # Render futures independently of the selected task histories.
    futures = [
        {
            "group": "future",
            "start_token": -1,
            "text": _FUTURE_TEMPLATE.format(passage=passage.text),
            "source": "c4",
            "source_id": passage.url,
            "category": passage.domain,
            "format_class": "document",
        }
        for passage in passages
    ]
    for row in futures:
        row["token_ids"] = encode(row["text"])
        row["token_count"] = len(row["token_ids"])
    histories = [
        {
            "group": "history",
            "start_token": -1,
            "text": _HISTORY_TEMPLATE.format(instruction=task.instruction),
            "source": task.source,
            "source_id": task.source_id,
            "category": task.category,
            "format_class": task.format_class,
        }
        for task in single_text
    ]
    for row in histories:
        row["token_ids"] = encode(row["text"])
        row["token_count"] = len(row["token_ids"])
    max_history = max(row["token_count"] for row in histories)
    max_future = max(row["token_count"] for row in futures)
    if max_history + max_future > tokenizer.model_max_length:
        raise ValueError(
            "single_text: maximum history + future length "
            f"{max_history} + {max_future} exceeds tokenizer model_max_length "
            f"{tokenizer.model_max_length}"
        )
    rng = np.random.default_rng(seed)
    pairs = rng.integers([0, 0], [len(histories), len(futures)], size=(2000, 2))
    for history_index, future_index in pairs:
        history, future = histories[int(history_index)], futures[int(future_index)]
        if history["token_ids"] + future["token_ids"] != encode(
            history["text"] + future["text"]
        ):
            raise ValueError(
                "history/future tokenization is not concatenation-safe for "
                f"{history['source_id']} + {future['source_id']}"
            )
    metadata = {
        "kind": "task_input",
        "dataset": "single_text",
        "tokenizer_path": str(tokenizer_path),
        "seed": seed,
        "history_template": _HISTORY_TEMPLATE,
        "future_template": _FUTURE_TEMPLATE,
        "sources": {
            "allenai/c4": _C4_REVISION,
            str(task_catalog): hashlib.sha256(task_catalog.read_bytes()).hexdigest(),
        },
        "history_count": len(histories),
        "future_count": len(futures),
        "history_counts_by_source_format": dict(sorted(Counter(
            f"{task.source}/{task.format_class}" for task in single_text
        ).items())),
        "passage_reject_counts": dict(sorted(passage_reject_counts.items())),
        "passage_token_bounds": [passage_min_tokens, passage_max_tokens],
        "history_max_tokens": history_max_tokens,
    }
    schema = pa.schema(
        [
            ("sample_id", pa.int32()),
            ("group", pa.string()),
            ("token_count", pa.int32()),
            ("start_token", pa.int64()),
            ("token_ids", pa.list_(pa.int32())),
            ("text", pa.string()),
            ("source", pa.string()),
            ("source_id", pa.string()),
            ("category", pa.string()),
            ("format_class", pa.string()),
        ],
        metadata={b"lm_svd": json.dumps(metadata, ensure_ascii=False).encode("utf-8")},
    )
    rows = [
        {"sample_id": index, **row}
        for index, row in enumerate(histories + futures)
    ]
    table = pa.Table.from_pylist(rows, schema=schema)
    write_parquet_atomic(table, output)
    print(
        "tasks by source/format: "
        + json.dumps(metadata["history_counts_by_source_format"])
    )
    print(f"passage reject counts: {json.dumps(dict(sorted(passage_reject_counts.items())))}")
    print(f"single_text: {len(single_text)} x {len(passages)}")
    print(f"output: {output}")
    return {
        "single_text_output": str(output),
        "single_text_history_count": len(single_text),
        "future_count": len(passages),
    }
