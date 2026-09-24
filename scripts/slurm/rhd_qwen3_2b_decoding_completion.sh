#!/usr/bin/env bash
#SBATCH --job-name=rhd-2b-decode
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
MODULE_NAME="${MODULE_NAME:-Miniforge3/24.11.3-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dissertation-eval}"

cd "$PROJECT_ROOT"
module load "$MODULE_NAME"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

REQUIRE_CUDA=1 bash scripts/slurm/preflight.sh

layers=(
  --relative-layer 0.6666666667
  --relative-layer 0.75
  --relative-layer 0.8333333333
  --relative-layer 0.875
  --relative-layer 0.9166666667
  --relative-layer 0.9583333333
)

run_condition() {
  local label="$1"
  local generations="outputs/qwen3-2b-${label}-development-v1.jsonl"
  local evaluation_dir="evaluations/qwen3-2b-${label}-development-v1"
  local diagnostic_dir="diagnostics/rhd-qwen3-2b-${label}-development-v1"
  local cv_dir="evaluations/rhd-qwen3-2b-${label}-development-cv-v1"

  python -m dissertation_eval evaluate-retruthqa "$generations" \
    --output-dir "$evaluation_dir"
  python -m dissertation_eval diagnose-rhd "$generations" \
    --output-dir "$diagnostic_dir" \
    "${layers[@]}"
  python -m dissertation_eval cross-validate-rhd \
    "$evaluation_dir/records.jsonl" \
    "$diagnostic_dir" \
    --folds 4 \
    --output-dir "$cv_dir"
}

run_condition beam4
run_condition sampling5
