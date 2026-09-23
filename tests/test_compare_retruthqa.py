import json
import tempfile
import unittest
from pathlib import Path

from dissertation_eval.evaluation.comparison import compare_retruthqa_runs


def generation(sample_id, generation_index, answer, reference="4", **overrides):
    row = {
        "record_id": f"{sample_id}-{generation_index}",
        "run_id": "run",
        "dataset_id": "retruthqa",
        "sample_id": sample_id,
        "generation_index": generation_index,
        "status": "success",
        "finish_reason": "stop",
        "generated_tokens": 10 + generation_index,
        "latency_ms": 1000 + generation_index,
        "peak_memory_bytes": 2 * 1024**3,
        "response_text": rf"FINAL: \\boxed{{{answer}}}",
        "reference_answer": reference,
        "sample_metadata": {"domain": "MATH"},
    }
    row.update(overrides)
    return row


class ReTruthQAComparisonTests(unittest.TestCase):
    def test_comparison_and_self_consistency_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            greedy = root / "greedy.jsonl"
            sampling = root / "sampling.jsonl"
            greedy.write_text(
                json.dumps(generation("a", 0, "4")) + "\n"
                + json.dumps(generation("b", 0, "3")) + "\n",
                encoding="utf-8",
            )
            sample_rows = [
                generation("a", 0, "4"), generation("a", 1, "4"),
                generation("a", 2, "5"), generation("a", 3, "5"),
                generation("a", 4, "4"),
                generation("b", 0, "3"), generation("b", 1, "4"),
                generation("b", 2, "3"), generation("b", 3, "4"),
                generation("b", 4, "3"),
            ]
            sampling.write_text(
                "".join(json.dumps(row) + "\n" for row in sample_rows), encoding="utf-8"
            )

            summary = compare_retruthqa_runs(
                {"greedy": greedy, "sampling5": sampling},
                root / "evaluation",
                self_consistency="sampling5",
            )

            self.assertEqual(summary["strategies"]["greedy"]["unique_samples"], 2)
            self.assertEqual(summary["strategies"]["greedy"]["evaluation"]["exact_matches"], 1)
            self.assertEqual(summary["self_consistency"]["samples"], 2)
            self.assertEqual(summary["self_consistency"]["exact_matches"], 1)
            self.assertEqual(summary["self_consistency"]["generations_per_sample_min"], 5)
            self.assertTrue((root / "evaluation" / "comparison.json").is_file())
            self.assertTrue((root / "evaluation" / "comparison.md").is_file())
            self.assertTrue((root / "evaluation" / "self-consistency-records.jsonl").is_file())

    def test_tie_uses_lowest_generation_index_and_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sampling = root / "sampling.jsonl"
            rows = [generation("a", 0, "5"), generation("a", 1, "4")]
            sampling.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = compare_retruthqa_runs(
                {"sampling": sampling}, root / "out", self_consistency="sampling"
            )
            self.assertEqual(summary["self_consistency"]["tie_count"], 1)
            records = [json.loads(line) for line in (
                root / "out" / "self-consistency-records.jsonl"
            ).read_text().splitlines()]
            self.assertEqual(records[0]["prediction_normalized"], "5")
            self.assertFalse(records[0]["exact_match"])


if __name__ == "__main__":
    unittest.main()
