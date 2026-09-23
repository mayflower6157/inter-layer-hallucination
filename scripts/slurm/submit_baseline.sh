#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${CONFIG_PATH:-configs/run.qwen3-4b.smoke.json}"
MODULE_NAME="${MODULE_NAME:-Miniforge3/24.11.3-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dissertation-eval}"
SMOKE_LIMIT="${SMOKE_LIMIT:-8}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this command on the cluster login node." >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Run config not found: $CONFIG_PATH" >&2
  exit 1
fi

sbatch --export="ALL,CONFIG_PATH=$CONFIG_PATH,MODULE_NAME=$MODULE_NAME,CONDA_ENV_NAME=$CONDA_ENV_NAME,SMOKE_LIMIT=$SMOKE_LIMIT" \
  scripts/slurm/baseline_qwen.sh
