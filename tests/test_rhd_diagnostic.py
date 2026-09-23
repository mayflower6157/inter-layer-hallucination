import unittest

from dissertation_eval.analysis.rhd import resolve_candidate_layers


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


if __name__ == "__main__":
    unittest.main()
