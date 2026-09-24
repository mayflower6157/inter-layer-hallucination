import json
import unittest

from dissertation_eval.evaluation.rhd_cv import (
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


if __name__ == "__main__":
    unittest.main()
