#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_PATH="$PROJECT_ROOT/outputs/smoke-retruthqa-mock-v1.jsonl"
RUN_MANIFEST_PATH="${OUTPUT_PATH}.run.json"

if [[ "${RESET_SMOKE_OUTPUT:-0}" == "1" ]]; then
  rm -f "$OUTPUT_PATH" "$RUN_MANIFEST_PATH"
fi

python -m dissertation_eval generate configs/run.mock.example.json

python - "$OUTPUT_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
if len(rows) != 3:
    raise SystemExit(f"Expected 3 generation records, found {len(rows)}")
if any(row["status"] != "success" for row in rows):
    raise SystemExit("At least one mock record did not succeed")
required = {"record_id", "run_id", "sample_id", "response_text", "latency_ms", "decoding"}
missing = required.difference(rows[0])
if missing:
    raise SystemExit(f"Generation schema fields missing: {sorted(missing)}")
print(f"Validated {len(rows)} append-only mock generation records at {path}")
PY

echo "Re-run this script to verify resume behaviour (three records should be skipped)."
