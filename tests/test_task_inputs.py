import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq

from data_pipline.task_inputs import (
    TaskCandidate,
    finalize_tasks,
    load_sni_tasks,
    load_tasksource_tasks,
    order_by_bucket,
    select_passages,
)


class SniTaskTests(unittest.TestCase):
    @staticmethod
    def header(name: str, **overrides) -> dict:
        record = {
            "name": name,
            "Categories": ["Sentiment Analysis"],
            "Definition": ["Classify the given review as positive or negative."],
            "Input_language": ["English"],
            "Instruction_language": ["English"],
            "Output_language": ["English"],
            "positive_example_input": "The delightful performances made this beautiful film truly memorable.",
        }
        record.update(overrides)
        return record

    def load_headers(self, records: list[dict]) -> list[TaskCandidate]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "headers.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            return load_sni_tasks(path)

    def test_classifies_single_text_and_structured_examples(self) -> None:
        records = [
            self.header("review"),
            self.header(
                "sentence_pair",
                positive_example_input=(
                    "Sentence1: The delightful performances made this film memorable.\n"
                    "Sentence2: The wonderful actors made the story enjoyable."
                ),
            ),
            self.header(
                "question_answering",
                Categories=["Question Answering"],
            ),
        ]
        tasks = {task.source_id: task for task in self.load_headers(records)}

        self.assertEqual(set(tasks), {"review", "sentence_pair", "question_answering"})
        self.assertEqual(tasks["review"].format_class, "single_text")
        self.assertIsNone(tasks["review"].format_reason)
        self.assertEqual(tasks["sentence_pair"].format_class, "structured")
        self.assertEqual(tasks["sentence_pair"].format_reason, "multi_field_example")
        self.assertEqual(tasks["question_answering"].format_class, "structured")
        self.assertEqual(tasks["question_answering"].format_reason, "structured_category")
        self.assertTrue(all(task.reject_reason is None for task in tasks.values()))

    def test_translation_collapse_keeps_lexicographically_first_name(self) -> None:
        records = [
            self.header(
                name,
                Categories=["Translation"],
                Definition=["Translate the supplied text into French."],
                Output_language=["French"],
            )
            for name in ("task_b", "task_a")
        ]
        tasks = {task.source_id: task for task in self.load_headers(records)}

        self.assertIsNone(tasks["task_a"].reject_reason)
        self.assertEqual(tasks["task_b"].reject_reason, "duplicate_translation_language")


class TasksourceTaskTests(unittest.TestCase):
    @staticmethod
    def row(task: str, labels: list[str], body: str | None = None) -> dict:
        options = " or ".join(f'"{label}"' for label in labels)
        if body is None:
            body = "The delightful performances made this beautiful film truly memorable."
        return {
            "task": task,
            "inputs": f"With no explanation, classify this text with either {options}.\n{body}",
        }

    def load_rows(self, rows: list[dict]) -> list[TaskCandidate]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            return load_tasksource_tasks(path)

    def test_label_subsets_are_canonicalized_to_sorted_union(self) -> None:
        tasks = self.load_rows([
            self.row("subsets", ["b", "a"]),
            self.row("subsets", ["c", "b"]),
            self.row("reordered", ["c", "a", "b"]),
        ])
        by_id = {task.source_id: task for task in tasks}
        expected = 'With no explanation, classify this text with either "a", "b" or "c".'

        self.assertEqual(set(by_id), {"subsets", "reordered"})
        for task in tasks:
            with self.subTest(task=task.source_id):
                self.assertEqual(task.instruction, expected)
                self.assertIsNone(task.reject_reason)
                self.assertEqual(task.format_class, "single_text")
                self.assertIsNone(task.format_reason)

    def test_nine_label_subsampled_union_is_rejected(self) -> None:
        tasks = self.load_rows([
            self.row("many_labels", ["a", "b", "c", "d", "e"]),
            self.row("many_labels", ["e", "f", "g", "h", "i"]),
        ])

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].reject_reason, "label_subsampled")

    def test_text_a_input_is_structured(self) -> None:
        tasks = self.load_rows([
            self.row(
                "sentence_pair", ["yes", "no"],
                body="text_A: The delightful performances made this beautiful film memorable.",
            ),
        ])

        self.assertEqual(len(tasks), 1)
        self.assertIsNone(tasks[0].reject_reason)
        self.assertEqual(tasks[0].format_class, "structured")
        self.assertEqual(tasks[0].format_reason, "multi_field_example")
        self.assertTrue(tasks[0].instruction.endswith('with either "no" or "yes".'))


class TaskSelectionTests(unittest.TestCase):
    def test_normalized_dedup_prefers_sni_over_tasksource(self) -> None:
        sni = TaskCandidate(
            "sni", "task_z", "Sentiment Analysis",
            "Classify this review!", "single_text", None,
        )
        tasksource = TaskCandidate(
            "tasksource", "task_a", "tasksource",
            "  CLASSIFY   this review. ", "single_text", None,
        )
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text, **kwargs: text.split()

        tasks = finalize_tasks([tasksource, sni], tokenizer, history_max_tokens=256)

        self.assertEqual(tasks, [sni, tasksource])
        self.assertIsNone(sni.reject_reason)
        self.assertEqual(sni.history_tokens, 4)
        self.assertEqual(tasksource.reject_reason, "duplicate_instruction")
        self.assertIsNone(tasksource.history_tokens)
        tokenizer.encode.assert_called_once_with(
            "Instruction: Classify this review!", add_special_tokens=False,
        )

    def test_bucket_order_is_deterministic_and_prefixes_are_balanced(self) -> None:
        tasks = [
            TaskCandidate("sni", f"{category}-{index}", category, "Classify the text.",
                          "single_text", None)
            for category, count in (("zeta", 4), ("alpha", 1), ("middle", 3))
            for index in range(count)
        ]

        first = order_by_bucket(tasks, seed=17)
        second = order_by_bucket(tasks, seed=17)

        self.assertEqual([task.source_id for task in first],
                         [task.source_id for task in second])
        self.assertCountEqual([task.source_id for task in first],
                              [task.source_id for task in tasks])
        for prefix_length in range(1, len(tasks) + 1):
            with self.subTest(prefix_length=prefix_length):
                categories = {task.category for task in first[:prefix_length]}
                self.assertEqual(len(categories), min(prefix_length, 3))


class PassageSelectionTests(unittest.TestCase):
    def test_passage_filters_and_first_per_normalized_domain(self) -> None:
        valid_text = "Patient gardeners carefully nurture beautiful flowers throughout spring."
        records = [
            {"url": "https://fragment.example/page", "text": valid_text.lower()},
            {"url": "https://caps.example/page", "text": valid_text.upper()},
            {"url": "https://url.example/page",
             "text": "Thoughtful readers discover useful information through https://example.com today."},
            {"url": "https://long.example/page",
             "text": "Thoughtful gardeners carefully nurture beautiful flowers throughout every season with patient attention."},
            {"url": "https://www.Gardens.example/first", "text": valid_text},
            {"url": "https://gardens.example/second",
             "text": "Thoughtful teachers patiently explain difficult lessons every morning."},
        ]
        rejection_counts = {}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c4.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
            passages = select_passages(
                [path], count_tokens=lambda text: len(text.split()), input_count=1,
                min_tokens=5, max_tokens=10, seed=7,
                rejection_counts=rejection_counts,
            )

        self.assertEqual(len(passages), 1)
        passage = passages[0]
        self.assertEqual(passage.url, "https://www.Gardens.example/first")
        self.assertEqual(passage.domain, "gardens.example")
        self.assertEqual(passage.text, valid_text)
        self.assertEqual(passage.passage_tokens, len(valid_text.split()))
        self.assertEqual(rejection_counts, {
            "fragment_start": 1,
            "uppercase": 1,
            "forbidden_content": 1,
            "long": 1,
            "duplicate_domain": 1,
        })


if __name__ == "__main__":
    unittest.main()
