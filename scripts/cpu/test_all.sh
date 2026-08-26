#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/4] Compiling Python sources"
python -m compileall -q src tests

echo "[2/4] Running unit and integration tests"
python -m unittest discover -s tests -v

echo "[3/4] Validating the synthetic dataset manifest"
python -m dissertation_eval validate-manifest data/manifests/mock.fixture.json --split dev

echo "[4/4] Checking the CLI"
python -m dissertation_eval --help >/dev/null

echo "CPU validation completed successfully."
