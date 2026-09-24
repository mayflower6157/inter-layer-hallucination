#!/usr/bin/env bash
#SBATCH --job-name=rhd-layer-grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
GENERATIONS_PATH="${GENERATIONS_PATH:-outputs/qwen3-4b-greedy-development-v1.jsonl}"
DIAGNOSTIC_LIMIT="${DIAGNOSTIC_LIMIT:-30}"
OUTPUT_ROOT="${OUTPUT_ROOT:-diagnostics/rhd-qwen3-4b-development-layer-grid-v1}"
MODULE_NAME="${MODULE_NAME:-Miniforge3/24.11.3-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dissertation-eval}"

cd "$PROJECT_ROOT"
module load "$MODULE_NAME"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

REQUIRE_CUDA=1 bash scripts/slurm/preflight.sh

run_grid() {
  local label="$1"
  shift
  local args=()
  for depth in "$@"; do
    args+=(--relative-layer "$depth")
  done
  python -m dissertation_eval diagnose-rhd "$GENERATIONS_PATH" \
    --output-dir "$OUTPUT_ROOT/$label" \
    --limit "$DIAGNOSTIC_LIMIT" \
    "${args[@]}"
}

# Our sparse sanity grid, followed by the normalized layer patterns released for
# DeepSeek-R1-Distill-Qwen-7B and -14B. All run on the identical development prefix.
run_grid sparse 0.5 0.625 0.75 0.875
run_grid reference-7b 0.5 0.5714285714 0.6428571429 0.7142857143 0.7857142857 0.8571428571 0.9285714286
run_grid reference-14b 0.6666666667 0.75 0.8333333333 0.875 0.9166666667 0.9583333333

python -m dissertation_eval evaluate-rhd \
  evaluations/qwen3-4b-greedy-development-v1/records.jsonl \
  --diagnostic "sparse=$OUTPUT_ROOT/sparse" \
  --diagnostic "reference-7b=$OUTPUT_ROOT/reference-7b" \
  --diagnostic "reference-14b=$OUTPUT_ROOT/reference-14b" \
  --output "$OUTPUT_ROOT/comparison.json"
