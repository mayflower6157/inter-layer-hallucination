# Dissertation evaluation pipeline

This directory contains the reproducible input and generation layer for the dissertation experiments. It deliberately separates dataset identity, normalized examples, generation backends, and append-only outputs. Future DoLa, DeCo, RHD, and inter-layer-guided decoders should implement the same `GenerationBackend` interface.

## Layout

```text
code/
├── configs/                 Run configurations
├── data/manifests/          Versioned dataset manifests (raw data are ignored)
├── outputs/                 Append-only JSONL generations (ignored)
├── schemas/                 Language-independent JSON Schema contracts
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
PYTHONPATH=src python -m unittest discover -s tests -v
dissertation-eval generate configs/run.mock.example.json --limit 5
```

The example mock command requires fixture data at the path declared by its manifest. For a real baseline, copy `configs/run.qwen3.template.json`, pin the exact model revision, and run:

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
