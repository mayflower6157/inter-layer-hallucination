import json
import tempfile
import unittest
from pathlib import Path

from dissertation_eval.datasets import load_split
from dissertation_eval.errors import DataIntegrityError


class HallusionBenchAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "VS" / "chart").mkdir(parents=True)
        (self.root / "VS" / "chart" / "0_1.png").write_bytes(b"fixture")

    def tearDown(self):
        self.temporary.cleanup()

    def _write_fixture(self, filename="./VS/chart/0_1.png") -> Path:
        rows = [
            {
                "category": "VS", "subcategory": "chart", "visual_input": "0",
                "set_id": "0", "figure_id": "0", "question_id": "0",
                "question": "Text control?", "gt_answer": "0", "filename": None,
            },
            {
                "category": "VS", "subcategory": "chart", "visual_input": "1",
                "set_id": "0", "figure_id": "1", "question_id": "0",
                "question": "Image question?", "gt_answer": "1", "filename": filename,
            },
        ]
        data = self.root / "HallusionBench.json"
        data.write_text(json.dumps(rows), encoding="utf-8")
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": "1.0", "dataset_id": "hallusionbench",
            "display_name": "HallusionBench", "modality": "vision_language",
            "source_url": "test", "revision": "test", "license": "test",
            "adapter": "hallusionbench",
            "fields": {
                "id": "question_id", "prompt": "question",
                "reference_answer": "gt_answer", "image": "filename",
                "metadata": ["visual_input", "figure_id"],
            },
            "splits": [{"name": "test", "path": "HallusionBench.json", "format": "json"}],
        }), encoding="utf-8")
        return manifest

    def test_composite_ids_and_optional_images(self):
        _, examples = load_split(self._write_fixture(), "test")
        rows = list(examples)
        self.assertEqual(len({row.sample_id for row in rows}), 2)
        self.assertIsNone(rows[0].image_path)
        self.assertTrue(Path(rows[1].image_path).is_file())
        self.assertEqual(rows[1].reference_answer, "1")
        self.assertEqual(rows[1].metadata["visual_input"], "1")

    def test_missing_image_is_rejected(self):
        _, examples = load_split(self._write_fixture("./VS/chart/missing.png"), "test")
        with self.assertRaisesRegex(DataIntegrityError, "does not exist"):
            list(examples)

    def test_official_uppercase_png_is_resolved(self):
        original = self.root / "VS" / "chart" / "0_1.png"
        uppercase = self.root / "VS" / "chart" / "case_mismatch.PNG"
        original.rename(uppercase)
        _, examples = load_split(
            self._write_fixture("./VS/chart/case_mismatch.png"), "test"
        )
        image_row = list(examples)[1]
        self.assertEqual(Path(image_row.image_path).name, "case_mismatch.PNG")


if __name__ == "__main__":
    unittest.main()
