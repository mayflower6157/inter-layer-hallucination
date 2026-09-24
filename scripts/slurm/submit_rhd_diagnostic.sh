#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

GENERATIONS_PATH="${GENERATIONS_PATH:-outputs/qwen3-4b-greedy-development-v1.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-diagnostics/qwen3-4b-greedy-development-rhd-smoke-v1}"
DIAGNOSTIC_LIMIT="${DIAGNOSTIC_LIMIT:-3}"
JOB_NAME="${JOB_NAME:-rhd-q3-4b-smoke}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
RHD_RELATIVE_LAYERS="${RHD_RELATIVE_LAYERS:-}"
RHD_RELATIVE_LAYERS="${RHD_RELATIVE_LAYERS//,/:}"

for path in "$GENERATIONS_PATH" "$GENERATIONS_PATH.run.json"; do
  if [[ ! -f "$path" ]]; then
    echo "Required input not found: $path" >&2
    exit 1
  fi
done

sbatch --chdir="$PROJECT_ROOT" \
  --job-name="$JOB_NAME" \
  --time="$TIME_LIMIT" \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,GENERATIONS_PATH=$GENERATIONS_PATH,OUTPUT_DIR=$OUTPUT_DIR,DIAGNOSTIC_LIMIT=$DIAGNOSTIC_LIMIT,RHD_RELATIVE_LAYERS=$RHD_RELATIVE_LAYERS" \
  scripts/slurm/rhd_diagnostic_qwen.sh
