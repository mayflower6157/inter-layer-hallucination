# Dissertation evaluation pipeline

This directory contains the reproducible input and generation layer for the dissertation experiments. It deliberately separates dataset identity, normalized examples, generation backends, and append-only outputs. Future DoLa, DeCo, RHD, and inter-layer-guided decoders should implement the same `GenerationBackend` interface.

## Layout

```text
code/
├── configs/                 Run configurations
├── data/manifests/          Versioned dataset manifests (raw data are ignored)
├── outputs/                 Append-only JSONL generations (ignored)
├── schemas/                 Language-independent JSON Schema contracts
├── scripts/
│   ├── cpu/                 Dependency-light local validation and mock smoke tests
│   └── slurm/               Cluster preflight, submission, and small GPU baseline
├── src/dissertation_eval/
│   ├── backends/            Mock and optional Hugging Face backends
│   ├── datasets/            Manifest validation and normalized loading
│   ├── pipeline/            Resumable generation runner
│   ├── schema.py            Canonical manifest and generation schemas
│   └── cli.py               Command-line interface
└── tests/                   Unit and end-to-end smoke tests
```

## Conda environment and installation

The recommended setup uses a dedicated Conda environment with Python 3.11. From this directory, create and activate it with:

```bash
conda create --name dissertation-eval python=3.11 pip -y
conda activate dissertation-eval
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

For local Hugging Face text and vision-language generation, install the optional model dependencies instead:

```bash
pip install -e '.[hf,dev]'
```

On a CUDA machine, install the PyTorch build appropriate for the machine's CUDA driver before the editable project installation. Use the command provided by the official PyTorch installation selector rather than hard-coding a CUDA version in this project.

Verify that the environment and command-line entry point are available:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
dissertation-eval --help
```

The `torch` verification command is required only when the `hf` dependencies are installed. Reactivate the environment in a new terminal with:

```bash
conda activate dissertation-eval
```

To remove the environment later:

```bash
conda deactivate
conda env remove --name dissertation-eval
```

## Dataset manifests

A manifest records the dataset source, immutable revision, licence, modality, field mapping, splits, expected row counts, and optional SHA-256 checksums. Paths are relative to the manifest file. Raw datasets belong under `data/raw/` and are intentionally excluded from version control.

Before a real run, copy a template, replace every `TODO`, add the dataset files, and validate them:

```bash
dissertation-eval validate-manifest data/manifests/retruthqa.json
```

Do not run a final experiment with null checksums or mutable model/data revisions.

## Generation records

Every attempted generation produces one JSONL record, including failures. Important fields include:

- stable `record_id`, `run_id`, dataset/split/sample IDs and group ID;
- exact model revision, backend, decoding configuration, prompt and seed;
- image path and reference answer where applicable;
- response, token IDs, finish reason, latency and peak memory;
- dataset metadata and a structured error with traceback.

The adjacent `*.run.json` file fingerprints the run configuration and environment. A run can resume only when that fingerprint is unchanged. This prevents results from two configurations being silently mixed.

## Smoke test and baseline run

The `mock` backend checks the complete data-to-output path without downloading a model:

```bash
bash scripts/cpu/test_all.sh
RESET_SMOKE_OUTPUT=1 bash scripts/cpu/smoke_mock.sh
```

The committed synthetic fixture contains three small examples and is used only to validate infrastructure. Running `smoke_mock.sh` again without `RESET_SMOKE_OUTPUT=1` verifies resume behaviour: all three completed records should be skipped.

After downloading HallusionBench into `data/raw/hallusionbench/`, validate all 1,129
records and run an eight-record real-data smoke test with the mock backend:

```bash
RESET_SMOKE_OUTPUT=1 bash scripts/cpu/smoke_hallusionbench.sh
```

The real-data smoke sample deliberately includes both text-only control questions and
image-conditioned questions. It validates composite sample IDs and local image paths
without loading a model.

The ReTruthQA release can be validated across all 12 provenance-preserving splits and
smoke-tested on eight real questions with:

```bash
RESET_SMOKE_OUTPUT=1 bash scripts/cpu/smoke_retruthqa.sh
```

The adapter records reasoning-path counts and source provenance as metadata, but does
not copy the large positive and negative reasoning traces into generation records.
The release has no top-level answer field, so the adapter extracts the last balanced
`\boxed{...}` value from every positive reasoning path. Preparation selects an answer
when normalized candidates agree or one candidate has a strict majority; ties remain
ambiguous and fail validation. Candidate votes, selection method, and confidence are
retained for audit. Normalization is materialized once under
`data/processed/retruthqa/v1/`.

The smoke script runs these explicit preparation and validation commands before the
mock generation step:

```bash
dissertation-eval prepare-retruthqa data/manifests/retruthqa.json \
  --output-dir data/processed/retruthqa/v1
dissertation-eval validate-prepared data/processed/retruthqa/v1
```

Preparation is content-addressed by the raw manifest checksum and extractor version.
An unchanged prepared directory is reused without scanning the reasoning traces again.
Use `--force` only when intentionally rebuilding the derived files. Generated processed
data remain ignored by Git; the preparation code and raw manifest are versioned.

### Small baseline on a Slurm GPU cluster

Complete CPU/mock validation before submitting a GPU job. On the cluster:

1. Clone the repository and create the Conda environment described above.
2. Install the `hf` dependencies and the cluster-compatible PyTorch build.
3. Copy the run template and replace all `TODO` values:

```bash
cp configs/run.qwen3.template.json configs/run.qwen3.json
```

4. Add and validate the real dataset manifest and raw files. Keep raw data outside Git.
5. Request an interactive GPU allocation once and run the preflight script:

```bash
CONDA_ENV_NAME=dissertation-eval bash scripts/slurm/preflight.sh
```

6. Submit an eight-example baseline smoke test:

```bash
CONFIG_PATH=configs/run.qwen3.json \
CONDA_ENV_NAME=dissertation-eval \
SMOKE_LIMIT=8 \
bash scripts/slurm/submit_baseline.sh
```

The generic Slurm resource requests in `scripts/slurm/baseline_qwen.sh` are intentionally conservative. Add the university-specific `--account`, `--partition`, module-loading, storage, and wall-time settings required by the cluster documentation before submission. Do not increase `SMOKE_LIMIT` until the JSONL records, GPU memory, parser output, and Slurm logs have been inspected.

For a non-Slurm local GPU baseline, use the same run configuration directly:

```bash
dissertation-eval generate configs/run.qwen3.json
```

Running the same command again skips completed `(sample_id, generation_index)` pairs. Use a new `run_id` and output path whenever the model, prompt, data, or decoding parameters change.

## Implementation rules for later decoders

1. Do not read benchmark labels inside a generation backend.
2. Do not change prompts, token limits, parsing, or stopping rules between decoder comparisons.
3. Store method-specific signals in `backend_metrics`; large tensors should use immutable sidecar files referenced by checksum rather than being embedded in JSONL.
4. Record unsuccessful and truncated generations instead of selectively retrying them.
5. Tune layers, thresholds, and intervention strength only on the development split.
