import json
import tempfile
import unittest
from pathlib import Path

from dissertation_eval.datasets import load_split
from dissertation_eval.errors import DataIntegrityError


class ReTruthQAAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _row(question="Question?"):
        return {
            "question": question,
            "pos_reasoning_paths_split": [[r"positive step \boxed{\frac{1}{2}}"]],
            "neg_reasoning_paths_split": [["negative step one", "negative step two"]],
        }

    def _manifest(self, file_format: str, rows: list[dict]) -> Path:
        directory = self.root / "R1-7B" / "hallucination"
        directory.mkdir(parents=True, exist_ok=True)
        data = directory / "MATH_task1.jsonl"
        if file_format == "json":
            data.write_text(json.dumps(rows), encoding="utf-8")
        else:
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": "1.0", "dataset_id": "retruthqa",
            "display_name": "ReTruthQA", "modality": "text",
            "source_url": "test", "revision": "test", "license": "test",
            "adapter": "retruthqa",
            "fields": {"id": "question", "prompt": "question"},
            "splits": [{
                "name": "test", "path": str(data.relative_to(self.root)),
                "format": file_format, "count": len(rows),
            }],
        }), encoding="utf-8")
        return manifest

    def test_jsonl_record_is_normalized_with_provenance(self):
        _, examples = load_split(self._manifest("jsonl", [self._row()]), "test")
        example = next(examples)
        self.assertTrue(example.sample_id.startswith("R1-7B/MATH/task1/"))
        self.assertEqual(len(example.group_id), 64)
        self.assertEqual(example.reference_answer, r"\frac{1}{2}")
        self.assertEqual(example.metadata["source_model"], "R1-7B")
        self.assertEqual(example.metadata["negative_step_counts"], [2])
        self.assertEqual(
            example.metadata["reference_answer_source"],
            "last_boxed_answer_in_positive_path",
        )

    def test_json_array_format_is_supported(self):
        _, examples = load_split(self._manifest("json", [self._row("Array question")]), "test")
        self.assertEqual(len(list(examples)), 1)

    def test_empty_reasoning_paths_are_rejected(self):
        row = self._row()
        row["neg_reasoning_paths_split"] = []
        _, examples = load_split(self._manifest("jsonl", [row]), "test")
        with self.assertRaisesRegex(DataIntegrityError, "non-empty list"):
            list(examples)

    def test_positive_path_without_boxed_answer_is_rejected(self):
        row = self._row()
        row["pos_reasoning_paths_split"] = [["answer without expected format"]]
        _, examples = load_split(self._manifest("jsonl", [row]), "test")
        with self.assertRaisesRegex(DataIntegrityError, "has no"):
            list(examples)


if __name__ == "__main__":
    unittest.main()
