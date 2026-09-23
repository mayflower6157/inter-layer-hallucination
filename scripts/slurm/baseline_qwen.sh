#!/usr/bin/env bash
#SBATCH --job-name=qwen-baseline-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# sbatch copies this script into Slurm's spool directory, so BASH_SOURCE no
# longer points into the repository. Slurm records the submission directory
# explicitly and that is the reproducible project root for this job.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" || ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
  echo "Project root is invalid: ${PROJECT_ROOT:-<unset>}" >&2
  echo "Submit this job through scripts/slurm/submit_baseline.sh from the repository." >&2
  exit 2
fi
cd "$PROJECT_ROOT"

MODULE_NAME="${MODULE_NAME:-Miniforge3/24.11.3-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dissertation-eval}"
CONFIG_PATH="${CONFIG_PATH:-configs/run.qwen3-4b.smoke.json}"
SMOKE_LIMIT="${SMOKE_LIMIT:-8}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Run config not found: $CONFIG_PATH" >&2
  echo "Submit from the repository root or set CONFIG_PATH to an existing run config." >&2
  exit 2
fi
if grep -q 'TODO_' "$CONFIG_PATH"; then
  echo "Run config still contains TODO values: $CONFIG_PATH" >&2
  exit 2
fi
if ! command -v module >/dev/null 2>&1; then
  echo "The environment-modules command is unavailable in this batch shell." >&2
  exit 2
fi
module load "$MODULE_NAME"
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is unavailable after loading $MODULE_NAME." >&2
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

REQUIRE_CUDA=1 bash scripts/slurm/preflight.sh
python -m dissertation_eval generate "$CONFIG_PATH" --limit "$SMOKE_LIMIT" --fail-fast

echo "Small baseline completed. Inspect JSONL records and Slurm logs before increasing the limit."
