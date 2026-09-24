import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dissertation_eval.evaluation.sampling_audit import audit_sampling_truncation
from dissertation_eval.io import sha256_file


class SamplingTruncationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _condition(self, label: str, model: str):
        generation = self.root / f"{label}.jsonl"
        evaluation = self.root / f"{label}-eval.jsonl"
        diagnostic = self.root / f"{label}-diag"
        sidecars = diagnostic / "sidecars"
        sidecars.mkdir(parents=True)
        config = self.root / f"{label}.json"
        decoding = {
            "strategy": "sample", "max_new_tokens": 8, "temperature": 0.7,
            "top_p": 0.9, "num_beams": 1, "num_generations": 2, "seeds": [1, 2],
        }
        config.write_text(json.dumps({
            "model_id": model, "prompt_template": "{prompt}", "decoding": decoding,
        }))
        generation_rows, evaluation_rows, diagnostic_rows = [], [], []
        specifications = [
            ("s1", 0, False, True, 3), ("s1", 1, False, False, 5),
            ("s2", 0, True, None, 8), ("s2", 1, False, True, 4),
        ]
        for offset, (sample, index, truncated, exact, tokens) in enumerate(specifications):
            record_id = f"{label}-{sample}-{index}"
            generation_rows.append({
                "record_id": record_id, "sample_id": sample, "generation_index": index,
                "seed": index + 1, "finish_reason": "length" if truncated else "stop",
                "generated_tokens": tokens, "token_ids": list(range(tokens - 1)) + [99],
                "decoding": decoding, "prompt_template": "{prompt}",
                "sample_metadata": {"domain": "MATH"},
            })
            evaluation_rows.append({"record_id": record_id, "exact_match": exact})
            sidecar = sidecars / f"{record_id}.npz"
            np.savez(sidecar, reference_reasoning_score=np.arange(tokens, dtype=float))
            diagnostic_rows.append({
                "record_id": record_id, "sample_id": sample, "generation_index": index,
                "token_count": tokens, "step_count": index + 1,
                "official_features": {
                    "attention_score": offset / 10, "cv_score": offset / 8,
                    "reasoning_score_mean": offset / 6, "pcc": offset / 4,
                },
                "sidecar_path": f"sidecars/{record_id}.npz",
                "sidecar_sha256": sha256_file(sidecar),
            })
        for path, rows in ((generation, generation_rows), (evaluation, evaluation_rows), (diagnostic / "records.jsonl", diagnostic_rows)):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        (diagnostic / "manifest.json").write_text(json.dumps({"model_id": model}))
        return generation, evaluation, diagnostic, config

    def test_audit_keeps_truncation_separate_from_correctness(self):
        conditions = {
            "2B": self._condition("two", "model-2b"),
            "4B": self._condition("four", "model-4b"),
        }
        output = self.root / "output"
        result = audit_sampling_truncation(conditions, output, bootstrap_iterations=20)
        self.assertTrue(result["configuration_sanity"]["passed"])
        self.assertEqual(result["conditions"]["2B"]["truncated_traces"], 1)
        self.assertEqual(result["conditions"]["2B"]["questions_any_truncated"], 1)
        self.assertEqual(result["conditions"]["2B"]["feature_auroc"]["pcc"]["incorrect_completed"]["bootstrap_unit"], "sample_id")
        self.assertEqual(result["paired_comparison"]["pairs"], 4)
        for name in ("audit.json", "report.md", "truncation_rates.svg", "token_step_distributions.svg", "feature_auroc.svg"):
            self.assertTrue((output / name).is_file())


if __name__ == "__main__":
    unittest.main()
