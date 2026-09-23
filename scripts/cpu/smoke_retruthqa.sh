#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_PATH="$PROJECT_ROOT/outputs/smoke-retruthqa-real-mock-v1.jsonl"
RUN_MANIFEST_PATH="${OUTPUT_PATH}.run.json"

if [[ "${RESET_SMOKE_OUTPUT:-0}" == "1" ]]; then
  rm -f "$OUTPUT_PATH" "$RUN_MANIFEST_PATH"
fi

python -m dissertation_eval prepare-retruthqa data/manifests/retruthqa.json \
  --output-dir data/processed/retruthqa/v1
python -m dissertation_eval validate-prepared data/processed/retruthqa/v1
python -m dissertation_eval generate configs/run.retruthqa.mock.json --limit 8 --fail-fast

python - "$OUTPUT_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
if len(rows) != 8:
    raise SystemExit(f"Expected 8 generation records, found {len(rows)}")
if len({row["sample_id"] for row in rows}) != len(rows):
    raise SystemExit("ReTruthQA sample IDs are not unique")
if any(row["status"] != "success" for row in rows):
    raise SystemExit("At least one ReTruthQA mock record did not succeed")
if any(not row["reference_answer"] or row["image_path"] is not None for row in rows):
    raise SystemExit("ReTruthQA answer extraction or image handling is incorrect")
for row in rows:
    metadata = row["sample_metadata"]
    if metadata["source_model"] != "R1-7B" or metadata["domain"] != "MATH":
        raise SystemExit("ReTruthQA provenance metadata is incorrect")
    if metadata["task"] != "task1":
        raise SystemExit("ReTruthQA task metadata is incorrect")
    if metadata["positive_path_count"] < 1 or metadata["negative_path_count"] < 1:
        raise SystemExit("ReTruthQA reasoning-path counts are invalid")
    if metadata["reference_answer_source"] != "last_boxed_answer_in_positive_path":
        raise SystemExit("ReTruthQA reference-answer provenance is missing")
print(f"Validated {len(rows)} real ReTruthQA mock records at {path}")
PY

echo "Re-run without RESET_SMOKE_OUTPUT=1 to verify resume behaviour."
