#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dissertation-eval}"
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available. Load the cluster's Conda/Anaconda module first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import sys
import torch
import transformers

print("Python:", sys.version.replace("\n", " "))
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device 0:", torch.cuda.get_device_name(0))
PY

python -m dissertation_eval validate-manifest data/manifests/mock.fixture.json --split dev
echo "Cluster preflight completed. Run this inside an allocated GPU job before the baseline."
