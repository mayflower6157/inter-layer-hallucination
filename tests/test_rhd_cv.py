import json
import tempfile
import unittest
from pathlib import Path

from dissertation_eval.evaluation.heldout import apply_frozen_calibration
from dissertation_eval.evaluation.rhd_cv import (
    MODEL_FEATURES,
    domain_stratified_folds,
    score_frozen_calibration,
    select_training_threshold,
)


class RhdCrossValidationTests(unittest.TestCase):
    def test_serialized_calibration_score_is_reproducible(self):
        model = {
            "features": ["a", "b"], "center": [2.0, 4.0], "scale": [2.0, 4.0],
            "intercept": 0.1, "weights": [0.5, -0.25],
        }
        first = score_frozen_calibration(model, {"a": 4.0, "b": 8.0})
        second = score_frozen_calibration(
            json.loads(json.dumps(model)), {"a": 4.0, "b": 8.0}
        )
        self.assertAlmostEqual(first, second)

    def test_domain_label_strata_are_distributed_without_overlap(self):
        rows = [
            {"record_id": f"{domain}-{label}-{index}", "domain": domain, "label": label}
            for domain in ("MATH", "Science", "MultiHopQA")
            for label in (0, 1)
            for index in range(4)
        ]
        folds = domain_stratified_folds(rows, 4)
        self.assertEqual(sorted(index for fold in folds for index in fold), list(range(24)))
        for fold in folds:
            keys = {(rows[index]["domain"], rows[index]["label"]) for index in fold}
            self.assertEqual(len(keys), 6)

    def test_threshold_is_selected_from_training_data(self):
        threshold = select_training_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.8)

    def test_sampling_generations_for_one_sample_never_cross_folds(self):
        rows = [
            {
                "record_id": f"{domain}-{sample}-{generation}",
                "sample_id": f"{domain}-{sample}",
                "domain": domain,
                "label": generation % 2,
            }
            for domain in ("MATH", "Science", "MultiHopQA")
            for sample in range(8)
            for generation in range(5)
        ]
        folds = domain_stratified_folds(rows, 4)
        assignment = {
            rows[index]["sample_id"]: fold_index
            for fold_index, fold in enumerate(folds)
            for index in fold
        }
        for fold_index, fold in enumerate(folds):
            self.assertTrue(all(assignment[rows[index]["sample_id"]] == fold_index for index in fold))

    def test_heldout_application_uses_frozen_threshold_and_excludes_unscored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation.jsonl"
            diagnostic = root / "diagnostic"
            calibration = root / "calibration.json"
            diagnostic.mkdir()
            manifest = {
                "model_id": "model", "model_revision": "revision",
                "candidate_layers": [1], "anchor_layer": 2,
            }
            (diagnostic / "manifest.json").write_text(json.dumps(manifest))
            evaluations = [
                {"record_id": "a", "sample_id": "q1", "domain": "MATH",
                 "evaluation_status": "exact_match", "exact_match": True},
                {"record_id": "b", "sample_id": "q2", "domain": "MATH",
                 "evaluation_status": "mismatch", "exact_match": False},
                {"record_id": "c", "sample_id": "q3", "domain": "MATH",
                 "evaluation_status": "truncated", "exact_match": None},
            ]
            evaluation.write_text("".join(json.dumps(row) + "\n" for row in evaluations))
            diagnostics = [
                {"record_id": key, "sample_id": f"q{index}", "generation_index": 0,
                 "token_count": index, "step_count": 1,
                 "official_features": {"attention_score": index, "cv_score": 0,
                 "reasoning_score_mean": 0, "pcc": 0}}
                for index, key in enumerate(("a", "b", "c"), 1)
            ]
            (diagnostic / "records.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in diagnostics)
            )
            models = {}
            for name, features in MODEL_FEATURES.items():
                models[name] = {
                    "features": list(features), "center": [0] * len(features),
                    "scale": [1] * len(features), "intercept": 0,
                    "weights": [1] + [0] * (len(features) - 1), "threshold": 0.8,
                }
            calibration.write_text(json.dumps({
                "scope": "development_only_frozen_calibration",
                "held_out_policy": "apply_only_no_refit_no_threshold_selection",
                "diagnostic_manifest": manifest, "models": models,
            }))
            result = apply_frozen_calibration(evaluation, diagnostic, calibration)
            self.assertEqual(result["scored_records"], 2)
            self.assertEqual(result["excluded_records"], 1)
            self.assertEqual(result["models"]["rhd_only"]["overall"]["threshold"], 0.8)
            self.assertEqual(len(result["records"]), 2)


if __name__ == "__main__":
    unittest.main()
