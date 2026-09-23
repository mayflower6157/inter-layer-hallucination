#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

submit() {
  local config="$1"
  local job_name="$2"
  local time_limit="$3"
  local dependency="${4:-}"
  local output
  output="$(CONFIG_PATH="$config" \
    JOB_NAME="$job_name" \
    TIME_LIMIT="$time_limit" \
    SMOKE_LIMIT=60 \
    DEPENDENCY="$dependency" \
    bash scripts/slurm/submit_baseline.sh)"
  printf '%s\n' "$output" >&2
  printf '%s\n' "$output" | sed -n 's/^Submitted batch job \([0-9][0-9]*\)$/\1/p' | tail -n 1
}

job_2b_greedy="$(submit configs/run.qwen3-2b.greedy-development.json q3-2b-dev-greedy 01:00:00)"
job_2b_beam="$(submit configs/run.qwen3-2b.beam4-development.json q3-2b-dev-beam4 01:30:00 "afterok:$job_2b_greedy")"
job_2b_sample="$(submit configs/run.qwen3-2b.sampling5-development.json q3-2b-dev-sample5 03:00:00 "afterok:$job_2b_beam")"
job_4b_greedy="$(submit configs/run.qwen3-4b.greedy-development.json q3-4b-dev-greedy 01:00:00 "afterok:$job_2b_sample")"
job_4b_beam="$(submit configs/run.qwen3-4b.beam4-development.json q3-4b-dev-beam4 01:30:00 "afterok:$job_4b_greedy")"
job_4b_sample="$(submit configs/run.qwen3-4b.sampling5-development.json q3-4b-dev-sample5 03:00:00 "afterok:$job_4b_beam")"

printf 'Submitted chain: 2B greedy=%s beam=%s sampling=%s; 4B greedy=%s beam=%s sampling=%s\n' \
  "$job_2b_greedy" "$job_2b_beam" "$job_2b_sample" \
  "$job_4b_greedy" "$job_4b_beam" "$job_4b_sample"
