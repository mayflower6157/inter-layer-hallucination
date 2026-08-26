import unittest

from dissertation_eval.errors import ConfigurationError
from dissertation_eval.schema import DatasetManifest, DecodingConfig


class SchemaTests(unittest.TestCase):
    def test_manifest_rejects_duplicate_splits(self):
        value = {
            "schema_version": "1.0", "dataset_id": "x", "display_name": "X",
            "modality": "text", "source_url": "local", "revision": "1", "license": "test",
            "fields": {"id": "id", "prompt": "question"},
            "splits": [{"name": "test", "path": "a.jsonl"}, {"name": "test", "path": "b.jsonl"}],
        }
        with self.assertRaisesRegex(ConfigurationError, "unique"):
            DatasetManifest.from_dict(value)

    def test_greedy_requires_one_generation(self):
        with self.assertRaisesRegex(ConfigurationError, "num_generations"):
            DecodingConfig.from_dict({"strategy": "greedy", "num_generations": 2})


if __name__ == "__main__":
    unittest.main()
