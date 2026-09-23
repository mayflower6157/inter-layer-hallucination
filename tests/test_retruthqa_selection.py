import json
import tempfile
import unittest
from pathlib import Path

from dissertation_eval.datasets.selection import (
    DEVELOPMENT_QUOTAS,
    PILOT_SPLIT,
    TEST_QUOTAS,
    create_retruthqa_selection,
)


class ReTruthQASelectionTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        splits = []
        names = sorted(set(DEVELOPMENT_QUOTAS) | set(TEST_QUOTAS))
        for name in names:
            count = max(DEVELOPMENT_QUOTAS.get(name, 0), TEST_QUOTAS.get(name, 0), 20)
            parts = name.split("_")
            source_model = "R1-7B" if name.startswith("r1-7b") else "R1-14B"
            domain_key = parts[1]
            domain = {"math": "MATH", "science": "Science", "multihopqa": "MultiHopQA"}[domain_key]
            task = parts[2]
            path = root / f"{name}.jsonl"
            rows = [{
                "sample_id": f"{name}/{index:03d}",
                "prompt": f"Question {index}",
                "reference_answer": str(index),
                "group_id": f"group-{name}-{index}",
                "source_model": source_model,
                "domain": domain,
                "task": task,
            } for index in range(count)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            splits.append({"name": name, "path": path.name, "format": "jsonl", "sha256": digest, "count": count})
        manifest = {
            "schema_version": "1.0", "dataset_id": "retruthqa", "display_name": "Fixture",
            "modality": "text", "source_url": "local", "revision": "fixture", "license": "test",
            "fields": {"id": "sample_id", "prompt": "prompt", "reference_answer": "reference_answer",
                       "group_id": "group_id", "metadata": ["source_model", "domain", "task"]},
            "splits": splits,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path

    def test_selection_is_balanced_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._fixture(root)
            selection = root / "selection.json"
            output = root / "selected"
            first = create_retruthqa_selection(source, selection, output)
            first_text = selection.read_text()
            second = create_retruthqa_selection(source, selection, output)
            self.assertEqual(first, second)
            self.assertEqual(first_text, selection.read_text())

            value = json.loads(first_text)
            development = {row["sample_id"] for row in value["records"] if row["role"] == "development"}
            test = {row["sample_id"] for row in value["records"] if row["role"] == "test"}
            pilot = set(value["observed_pilot"]["sample_ids"])
            self.assertEqual(len(development), 60)
            self.assertEqual(len(test), 180)
            self.assertFalse(development & test)
            self.assertFalse(pilot & test)
            self.assertEqual(value["roles"]["development"]["domain_counts"], {
                "MATH": 20, "MultiHopQA": 20, "Science": 20,
            })
            self.assertEqual(value["roles"]["test"]["domain_counts"], {
                "MATH": 60, "MultiHopQA": 60, "Science": 60,
            })
            self.assertEqual(json.loads((output / "manifest.json").read_text())["splits"][0]["count"], 60)


if __name__ == "__main__":
    unittest.main()
