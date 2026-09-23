import json
import tempfile
import unittest
from pathlib import Path

from dissertation_eval.evaluation.retruthqa import (
    evaluate_record,
    evaluate_retruthqa_generations,
    extract_boxed_answers,
    extract_prediction,
)


def generation(**overrides):
    row = {
        "record_id": "a" * 64,
        "run_id": "run",
        "dataset_id": "retruthqa",
        "sample_id": "sample",
        "generation_index": 0,
        "status": "success",
        "finish_reason": "stop",
        "generated_tokens": 12,
        "response_text": r"Reasoning. Final answer: $\boxed{\frac{1}{2}}$",
        "reference_answer": r"\dfrac{1}{2}",
        "sample_metadata": {"domain": "MATH"},
    }
    row.update(overrides)
    return row


class ReTruthQAEvaluationTests(unittest.TestCase):
    def test_balanced_boxed_extraction_supports_nested_latex(self):
        text = r"first \boxed{1} final \boxed{\frac{1}{2}}"
        self.assertEqual(extract_boxed_answers(text), ["1", r"\frac{1}{2}"])

    def test_incomplete_trailing_box_does_not_erase_complete_answer(self):
        self.assertEqual(extract_boxed_answers(r"\boxed{7} then \boxed{"), ["7"])

    def test_final_marker_variant_is_supported(self):
        self.assertEqual(extract_prediction("work\n### Final Answer:\n42"), ("42", "final_marker"))

    def test_normalized_math_match(self):
        result = evaluate_record(generation())
        self.assertEqual(result["evaluation_status"], "exact_match")
        self.assertTrue(result["exact_match"])

    def test_full_math_text_wrapper_is_ignored(self):
        result = evaluate_record(
            generation(response_text=r"\boxed{Navin}", reference_answer=r"\text{Navin}")
        )
        self.assertEqual(result["evaluation_status"], "exact_match")
        self.assertTrue(result["exact_match"])

    def test_numeric_thousands_separator_is_ignored(self):
        result = evaluate_record(
            generation(response_text=r"\boxed{58500}", reference_answer=r"58,\!500")
        )
        self.assertEqual(result["evaluation_status"], "exact_match")
        self.assertTrue(result["exact_match"])

    def test_truncated_record_is_not_scored(self):
        result = evaluate_record(generation(finish_reason="length"))
        self.assertEqual(result["evaluation_status"], "truncated")
        self.assertIsNone(result["exact_match"])
        self.assertTrue(result["needs_manual_review"])

    def test_mismatch_is_preserved_for_manual_review(self):
        result = evaluate_record(
            generation(
                response_text=r"\boxed{12^{th}\text{ grade}}",
                reference_answer="12",
            )
        )
        self.assertEqual(result["evaluation_status"], "mismatch")
        self.assertFalse(result["exact_match"])
        self.assertTrue(result["needs_manual_review"])

    def test_evaluation_writes_records_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "generations.jsonl"
            source.write_text(
                "\n".join(json.dumps(row) for row in [generation(), generation(
                    record_id="b" * 64, sample_id="truncated", finish_reason="length"
                )]) + "\n",
                encoding="utf-8",
            )
            summary = evaluate_retruthqa_generations(source, root / "evaluation")
            self.assertEqual(summary["records"], 2)
            self.assertEqual(summary["exact_matches"], 1)
            self.assertEqual(summary["status_counts"], {"exact_match": 1, "truncated": 1})
            self.assertTrue((root / "evaluation" / "records.jsonl").is_file())
            self.assertTrue((root / "evaluation" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
