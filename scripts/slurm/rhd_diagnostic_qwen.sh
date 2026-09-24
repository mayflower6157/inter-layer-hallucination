#!/usr/bin/env bash
#SBATCH --job-name=rhd-diagnostic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
GENERATIONS_PATH="${GENERATIONS_PATH:?Set GENERATIONS_PATH to a generation JSONL}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for diagnostic artifacts}"
DIAGNOSTIC_LIMIT="${DIAGNOSTIC_LIMIT:-3}"
RHD_RELATIVE_LAYERS="${RHD_RELATIVE_LAYERS:-}"
MODULE_NAME="${MODULE_NAME:-Miniforge3/24.11.3-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dissertation-eval}"

cd "$PROJECT_ROOT"
module load "$MODULE_NAME"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

REQUIRE_CUDA=1 bash scripts/slurm/preflight.sh
args=(
  "$GENERATIONS_PATH"
  --output-dir "$OUTPUT_DIR"
  --limit "$DIAGNOSTIC_LIMIT"
)
if [[ -n "$RHD_RELATIVE_LAYERS" ]]; then
  IFS=':' read -ra depths <<< "$RHD_RELATIVE_LAYERS"
  for depth in "${depths[@]}"; do
    args+=(--relative-layer "$depth")
  done
fi
python -m dissertation_eval diagnose-rhd "${args[@]}"
