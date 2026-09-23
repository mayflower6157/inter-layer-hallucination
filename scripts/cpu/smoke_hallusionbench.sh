#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_PATH="$PROJECT_ROOT/outputs/smoke-hallusionbench-real-mock-v1.jsonl"
RUN_MANIFEST_PATH="${OUTPUT_PATH}.run.json"

if [[ "${RESET_SMOKE_OUTPUT:-0}" == "1" ]]; then
  rm -f "$OUTPUT_PATH" "$RUN_MANIFEST_PATH"
fi

python -m dissertation_eval validate-manifest \
  data/manifests/hallusionbench.json --split test
python -m dissertation_eval generate \
  configs/run.hallusionbench.mock.json --limit 8 --fail-fast

python - "$OUTPUT_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
if len(rows) != 8:
    raise SystemExit(f"Expected 8 generation records, found {len(rows)}")
if len({row["sample_id"] for row in rows}) != len(rows):
    raise SystemExit("HallusionBench sample IDs are not unique")
if any(row["status"] != "success" for row in rows):
    raise SystemExit("At least one HallusionBench mock record did not succeed")
if not any(row["image_path"] is None for row in rows):
    raise SystemExit("Smoke sample did not exercise a text-only control record")
image_rows = [row for row in rows if row["image_path"] is not None]
if not image_rows or any(not Path(row["image_path"]).is_file() for row in image_rows):
    raise SystemExit("Smoke sample image paths were not resolved correctly")
print(f"Validated {len(rows)} real HallusionBench mock records at {path}")
PY

echo "Re-run without RESET_SMOKE_OUTPUT=1 to verify resume behaviour."
