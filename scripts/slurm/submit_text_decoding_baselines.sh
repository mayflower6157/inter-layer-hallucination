#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

submit() {
  local config="$1"
  local job_name="$2"
  local time_limit="$3"
  CONFIG_PATH="$config" \
    JOB_NAME="$job_name" \
    TIME_LIMIT="$time_limit" \
    SMOKE_LIMIT=44 \
    bash scripts/slurm/submit_baseline.sh
}

submit configs/run.qwen3-4b.greedy-r1-7b-math-task1.json q3-4b-greedy 01:00:00
submit configs/run.qwen3-4b.beam4-r1-7b-math-task1.json q3-4b-beam4 03:00:00
submit configs/run.qwen3-4b.sampling5-r1-7b-math-task1.json q3-4b-sample5 04:00:00
