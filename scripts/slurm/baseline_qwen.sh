#!/usr/bin/env bash
#SBATCH --job-name=qwen-baseline-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dissertation-eval}"
CONFIG_PATH="${CONFIG_PATH:-configs/run.qwen3.json}"
SMOKE_LIMIT="${SMOKE_LIMIT:-8}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Run config not found: $CONFIG_PATH" >&2
  echo "Copy configs/run.qwen3.template.json to configs/run.qwen3.json and replace every TODO." >&2
  exit 2
fi
if grep -q 'TODO_' "$CONFIG_PATH"; then
  echo "Run config still contains TODO values: $CONFIG_PATH" >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is unavailable. Add the appropriate cluster module load command for your site." >&2
  exit 2
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: $(hostname)"
echo "Config: $CONFIG_PATH"
echo "Smoke limit: $SMOKE_LIMIT"
nvidia-smi || true

bash scripts/slurm/preflight.sh
python -m dissertation_eval generate "$CONFIG_PATH" --limit "$SMOKE_LIMIT" --fail-fast

echo "Small baseline completed. Inspect JSONL records and Slurm logs before increasing the limit."
