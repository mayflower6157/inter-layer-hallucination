#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODULE_NAME="${MODULE_NAME:-Miniforge3/24.11.3-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dissertation-eval}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_REVISION="${MODEL_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}"

if ! command -v conda >/dev/null 2>&1; then
  if ! command -v module >/dev/null 2>&1; then
    echo "Neither conda nor the environment-modules command is available." >&2
    exit 1
  fi
  module load "$MODULE_NAME"
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

python - "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, revision = sys.argv[1:]
path = snapshot_download(repo_id=model_id, revision=revision)
print(f"Cached {model_id}@{revision} at {path}")
PY
