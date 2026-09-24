import math
import unittest

from dissertation_eval.analysis.rhd import (
    official_trace_features,
    resolve_candidate_layers,
    segment_reasoning_steps,
)


class RhdDiagnosticTests(unittest.TestCase):
    def test_relative_layers_map_to_qwen_depths(self):
        self.assertEqual(
            resolve_candidate_layers(28, [0.5, 0.625, 0.75, 0.875]),
            [14, 18, 21, 25],
        )
        self.assertEqual(
            resolve_candidate_layers(36, [0.5, 0.625, 0.75, 0.875]),
            [18, 23, 27, 32],
        )

    def test_duplicate_rounded_layers_are_removed(self):
        self.assertEqual(resolve_candidate_layers(4, [0.5, 0.51, 0.75]), [2, 3])

    def test_invalid_relative_layer_is_rejected(self):
        for value in (0.0, 1.0, -0.5, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_candidate_layers(28, [value])

    def test_paragraph_step_segmentation_is_deterministic(self):
        text = " First step.\n\nSecond step.\n  \nThird step. "
        self.assertEqual(
            [text[start:end] for start, end in segment_reasoning_steps(text)],
            ["First step.", "Second step.", "Third step."],
        )

    def test_official_features_have_released_order_and_finite_values(self):
        features = official_trace_features(
            [1.0, 2.0, 1.0, 3.0, 2.0],
            [0.2, 0.3, 0.25, 0.5, 0.4],
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0.2, 0.8, 0, 0, 0],
                [0.1, 0.3, 0.6, 0, 0],
                [0.1, 0.2, 0.3, 0.4, 0],
            ],
        )
        self.assertEqual(
            list(features),
            ["attention_score", "cv_score", "reasoning_score_mean", "pcc"],
        )
        self.assertAlmostEqual(features["reasoning_score_mean"], 1.8)
        self.assertTrue(all(math.isfinite(value) for value in features.values()))


if __name__ == "__main__":
    unittest.main()
