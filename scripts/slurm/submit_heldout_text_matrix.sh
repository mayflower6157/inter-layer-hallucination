#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

submit() {
  local config="$1" job_name="$2" time_limit="$3" dependency="${4:-}"
  local output job_id
  output="$(CONFIG_PATH="$config" JOB_NAME="$job_name" TIME_LIMIT="$time_limit" \
    SMOKE_LIMIT=180 DEPENDENCY="$dependency" bash scripts/slurm/submit_baseline.sh)"
  printf '%s\n' "$output" >&2
  job_id="$(printf '%s\n' "$output" | sed -n 's/^Submitted batch job \([0-9][0-9]*\)$/\1/p' | tail -n 1)"
  [[ -n "$job_id" ]] || { echo "Could not determine job ID for $config" >&2; return 1; }
  printf '%s\n' "$job_id"
}

# Two independent model lanes permit at most two concurrent A40 allocations.
job_2b_greedy="$(submit configs/run.qwen3-2b.greedy-test.json q3-2b-test-greedy 02:00:00)"
job_2b_beam="$(submit configs/run.qwen3-2b.beam4-test.json q3-2b-test-beam4 03:00:00 "afterok:$job_2b_greedy")"
job_2b_sample="$(submit configs/run.qwen3-2b.sampling5-test.json q3-2b-test-sample5 08:00:00 "afterok:$job_2b_beam")"

job_4b_greedy="$(submit configs/run.qwen3-4b.greedy-test.json q3-4b-test-greedy 02:00:00)"
job_4b_beam="$(submit configs/run.qwen3-4b.beam4-test.json q3-4b-test-beam4 03:00:00 "afterok:$job_4b_greedy")"
job_4b_sample="$(submit configs/run.qwen3-4b.sampling5-test.json q3-4b-test-sample5 08:00:00 "afterok:$job_4b_beam")"

printf '2B lane: %s -> %s -> %s\n4B lane: %s -> %s -> %s\n' \
  "$job_2b_greedy" "$job_2b_beam" "$job_2b_sample" \
  "$job_4b_greedy" "$job_4b_beam" "$job_4b_sample"
