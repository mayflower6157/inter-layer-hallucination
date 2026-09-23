import json
import tempfile
import unittest
from pathlib import Path

from dissertation_eval.datasets.prepare import prepare_retruthqa, validate_prepared_retruthqa
from dissertation_eval.errors import DataIntegrityError
from dissertation_eval.io import iter_jsonl, sha256_file


class PrepareReTruthQATests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _source_manifest(self, domain="MATH", positive_answers=None) -> Path:
        positive_answers = positive_answers or [r"\dfrac{1}{2}", r"\frac{1}{2}"]
        directory = self.root / "raw" / "R1-7B" / "hallucination"
        directory.mkdir(parents=True)
        data = directory / f"{domain}_task1.jsonl"
        row = {
            "question": "Prepared question?",
            "pos_reasoning_paths_split": [
                [f"reasoning \\boxed{{{answer}}}"] for answer in positive_answers
            ],
            "neg_reasoning_paths_split": [[r"wrong \boxed{0}"]],
        }
        data.write_text(json.dumps(row) + "\n", encoding="utf-8")
        manifest = self.root / "source-manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": "1.0", "dataset_id": "retruthqa",
            "display_name": "ReTruthQA", "modality": "text",
            "source_url": "test", "revision": "test", "license": "test",
            "adapter": "retruthqa",
            "fields": {"id": "question", "prompt": "question"},
            "splits": [{
                "name": "fixture", "path": str(data.relative_to(self.root)),
                "format": "jsonl", "sha256": sha256_file(data), "count": 1,
            }],
        }), encoding="utf-8")
        return manifest

    def test_prepare_validate_and_reuse(self):
        output = self.root / "processed"
        first = prepare_retruthqa(self._source_manifest(), output)
        self.assertEqual(first["status"], "prepared")
        self.assertEqual(first["total_ambiguous"], 0)
        row = next(iter_jsonl(output / "fixture.jsonl"))
        self.assertEqual(row["reference_answer"], r"\dfrac{1}{2}")
        self.assertEqual(row["reference_answer_normalized"], r"\frac{1}{2}")
        self.assertEqual(row["answer_status"], "consensus")
        self.assertEqual(validate_prepared_retruthqa(output)["total_records"], 1)
        second = prepare_retruthqa(self.root / "source-manifest.json", output)
        self.assertEqual(second["status"], "reused")

    def test_ambiguous_answers_are_materialized_but_rejected_for_experiments(self):
        output = self.root / "processed"
        result = prepare_retruthqa(
            self._source_manifest(domain="Science", positive_answers=["A", "B"]), output
        )
        self.assertEqual(result["total_ambiguous"], 1)
        row = next(iter_jsonl(output / "fixture.jsonl"))
        self.assertIsNone(row["reference_answer"])
        self.assertEqual(row["answer_status"], "ambiguous")
        with self.assertRaisesRegex(DataIntegrityError, "ambiguous"):
            validate_prepared_retruthqa(output)

    def test_strict_majority_selects_reference_and_preserves_votes(self):
        output = self.root / "processed"
        result = prepare_retruthqa(
            self._source_manifest(
                domain="MultiHopQA",
                positive_answers=["Correct Entity", "Wrong Entity", "Correct Entity"],
            ),
            output,
        )
        self.assertEqual(result["total_majority"], 1)
        row = next(iter_jsonl(output / "fixture.jsonl"))
        self.assertEqual(row["reference_answer"], "Correct Entity")
        self.assertEqual(row["answer_status"], "majority")
        self.assertEqual(row["reference_answer_confidence"], 2 / 3)
        self.assertEqual(row["reference_answer_candidate_counts"]["correct entity"], 2)
        self.assertEqual(validate_prepared_retruthqa(output)["ambiguous"], 0)


if __name__ == "__main__":
    unittest.main()
