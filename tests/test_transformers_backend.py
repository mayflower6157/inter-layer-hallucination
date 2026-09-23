import unittest

from dissertation_eval.backends.transformers import infer_finish_reason


class TransformersBackendTests(unittest.TestCase):
    def test_token_limit_is_reported_as_length(self):
        self.assertEqual(infer_finish_reason([10, 11, 12], 3, 99), "length")

    def test_eos_at_limit_is_reported_as_stop(self):
        self.assertEqual(infer_finish_reason([10, 11, 99], 3, 99), "stop")

    def test_multiple_eos_ids_are_supported(self):
        self.assertEqual(infer_finish_reason([10, 101], 8, [99, 101]), "stop")


if __name__ == "__main__":
    unittest.main()
