#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODULE_NAME="${MODULE_NAME:-Miniforge3/24.11.3-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dissertation-eval}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"
if ! command -v conda >/dev/null 2>&1; then
  if ! command -v module >/dev/null 2>&1; then
    echo "Neither conda nor the environment-modules command is available." >&2
    exit 1
  fi
  module load "$MODULE_NAME"
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

REQUIRE_CUDA="$REQUIRE_CUDA" python - <<'PY'
import os
import sys
import torch
import transformers

print("Python:", sys.version.replace("\n", " "))
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if os.environ["REQUIRE_CUDA"] == "1" and not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this run but PyTorch cannot access a GPU")
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device 0:", torch.cuda.get_device_name(0))
PY

python -m dissertation_eval validate-manifest data/manifests/mock.fixture.json --split dev
echo "Cluster preflight completed. Run this inside an allocated GPU job before the baseline."
