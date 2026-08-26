import json
import tempfile
import unittest
from pathlib import Path

from dissertation_eval.io import iter_jsonl
from dissertation_eval.pipeline.runner import run_generation


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _manifest(self, rows):
        data = self.root / "data.jsonl"
        data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest = self.root / "manifest.json"
        write_json(manifest, {
            "schema_version": "1.0", "dataset_id": "fixture", "display_name": "Fixture",
            "modality": "text", "source_url": "local", "revision": "test", "license": "test",
            "fields": {"id": "id", "prompt": "question", "reference_answer": "answer", "metadata": ["domain"]},
            "splits": [{"name": "dev", "path": "data.jsonl", "format": "jsonl", "count": len(rows)}],
        })
        return manifest

    def test_mock_pipeline_is_append_only_and_resumable(self):
        self._manifest([
            {"id": "a", "question": "2+2?", "answer": "4", "domain": "math"},
            {"id": "b", "question": "Sky colour?", "answer": "blue", "domain": "science"},
        ])
        config = self.root / "run.json"
        write_json(config, {
            "run_id": "fixture-run", "dataset_manifest": "manifest.json", "split": "dev",
            "backend": "mock", "model_id": "mock", "model_revision": "1",
            "prompt_template": "Q: {prompt}", "output_path": "out.jsonl",
            "decoding": {"strategy": "greedy", "num_generations": 1, "seeds": [7]},
        })
        self.assertEqual(run_generation(config), {"success": 2, "error": 0, "skipped": 0})
        self.assertEqual(run_generation(config), {"success": 0, "error": 0, "skipped": 2})
        rows = list(iter_jsonl(self.root / "out.jsonl"))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "success")
        self.assertEqual(rows[0]["sample_metadata"], {"domain": "math"})
        self.assertNotEqual(rows[0]["record_id"], rows[1]["record_id"])
        self.assertTrue((self.root / "out.jsonl.run.json").exists())

    def test_resume_rejects_changed_config(self):
        self._manifest([{"id": "a", "question": "x"}])
        config = self.root / "run.json"
        base = {"run_id": "r", "dataset_manifest": "manifest.json", "split": "dev", "backend": "mock",
                "model_id": "m", "model_revision": "1", "prompt_template": "{prompt}", "output_path": "out.jsonl"}
        write_json(config, base)
        run_generation(config)
        base["prompt_template"] = "Changed {prompt}"
        write_json(config, base)
        with self.assertRaisesRegex(RuntimeError, "config changed"):
            run_generation(config)


if __name__ == "__main__":
    unittest.main()
