# Inter-layer reasoning-hallucination evaluation

This repository contains the reproducible generation, Reasoning Hallucination Detection
(RHD), and evaluation pipeline used in the dissertation. The confirmatory study compares
Qwen3-2B and Qwen3-4B under Greedy, Beam-4, and Sampling-5 decoding on a frozen
60-question development split and a sealed 180-question ReTruthQA held-out split. RHD is
applied after generation as a label-blind, teacher-forced diagnostic; it never changes
token selection.

Dataset identity, normalised examples, generation backends, append-only outputs,
diagnostic sidecars, frozen calibrations, and publication artifacts are kept separate so
that every reported aggregate can be traced to an immutable input and protocol version.

Run-level decisions, Slurm job identifiers, timings, failures, and result follow-ups are
recorded in [`docs/experiment-log.md`](docs/experiment-log.md). Update that log before
using a run in the dissertation so reported settings can be traced to a configuration,
code commit, and immutable output.

## Final held-out results

The pre-held-out protocol was frozen at commit [`33d155b`](https://github.com/mayflower6157/inter-layer-hallucination/commit/33d155b).
All transforms, feature directions, logistic weights, and thresholds were fitted on the
development split and then applied without refitting. Confidence intervals use 2,000
domain-stratified bootstrap replicates clustered by `sample_id`, so all five Sampling-5
traces for a question are resampled together.

| Condition | Scored | Length/step AUROC | RHD AUROC [95% CI] | Combined AUROC | Combined − baseline [95% CI] |
|---|---:|---:|---:|---:|---:|
| Qwen3-2B Greedy | 122 | 0.508 | 0.636 [0.535, 0.741] | 0.620 | **0.111 [0.016, 0.208]** |
| Qwen3-2B Beam-4 | 126 | 0.547 | 0.617 [0.516, 0.709] | 0.628 | 0.081 [−0.069, 0.229] |
| Qwen3-2B Sampling-5 | 606 | 0.582 | 0.630 [0.556, 0.698] | 0.617 | 0.036 [−0.033, 0.100] |
| Qwen3-4B Greedy | 169 | 0.556 | 0.642 [0.557, 0.721] | 0.653 | **0.097 [0.027, 0.174]** |
| Qwen3-4B Beam-4 | 169 | 0.530 | 0.574 [0.487, 0.657] | 0.575 | 0.046 [−0.047, 0.139] |
| Qwen3-4B Sampling-5 | 849 | 0.560 | 0.609 [0.550, 0.668] | 0.603 | 0.043 [−0.018, 0.104] |

The central result is conditional: development-fitted RHD features provide modest
held-out discrimination, but incremental value beyond response length and step count is
established only for Greedy decoding. The evidence does not support a decoding-invariant
or universally calibrated detector. The complete statistical statement is frozen in
[`conclusion_freeze.md`](docs/results/rhd-heldout-final-v1/conclusion_freeze.md), and the
full AUROC, AUPRC, prevalence, balanced-accuracy, and confidence-interval table is in
[`heldout_summary.md`](docs/results/rhd-heldout-final-v1/heldout_summary.md).

<p align="center">
  <img src="docs/results/rhd-heldout-final-v1/publication/heldout_auroc_ci.svg" width="92%" alt="Held-out RHD AUROC with clustered confidence intervals">
</p>

### Generation, completion, and self-consistency

| Condition | Completed | Truncated | Accuracy on scored traces | Peak GPU memory (GiB) |
|---|---:|---:|---:|---:|
| Qwen3-2B Greedy | 67.8% | 32.2% | 44.3% | 3.65 |
| Qwen3-2B Beam-4 | 70.0% | 30.0% | 50.8% | 4.06 |
| Qwen3-2B Sampling-5 | 67.3% | 32.7% | 46.2% | 3.65 |
| Qwen3-4B Greedy | 93.9% | 6.1% | 45.6% | 7.70 |
| Qwen3-4B Beam-4 | 94.4% | 5.6% | 45.6% | 8.23 |
| Qwen3-4B Sampling-5 | 94.8% | 5.2% | 44.5% | 7.76 |

Sampling-5 self-consistency produced a valid majority-vote answer for 83.9% of Qwen3-2B
questions and 97.8% of Qwen3-4B questions. Accuracy over all 180 questions was 38.9% and
47.2%, respectively. Detailed publication tables are available as both CSV and LaTeX:

- [generation results](docs/results/rhd-heldout-final-v1/publication/heldout_generation_results.csv)
- [RHD operating results](docs/results/rhd-heldout-final-v1/publication/heldout_operating_results.csv)
- [domain breakdown](docs/results/rhd-heldout-final-v1/publication/heldout_domain_results.csv)
- [Sampling-5 self-consistency](docs/results/rhd-heldout-final-v1/publication/heldout_self_consistency.csv)

<p align="center">
  <img src="docs/results/rhd-heldout-final-v1/publication/heldout_truncation.svg" width="78%" alt="Held-out truncation rate by model and decoding strategy">
</p>

The full set of paper-ready figures, tables, trajectory-selection manifests, and
provenance records is under
[`docs/results/rhd-heldout-final-v1/`](docs/results/rhd-heldout-final-v1/).

## Layout

```text
code/
├── configs/                 Run configurations
├── diagnostics/             RHD extraction outputs and sidecars (large files ignored)
├── evaluations/             Derived evaluation records and summaries (ignored)
├── data/
│   ├── fixtures/            Committed synthetic test data
│   ├── manifests/           Versioned raw dataset manifests
│   ├── raw/                 Downloaded source data (ignored)
│   └── processed/           Reproducible derived data (ignored)
├── outputs/                 Append-only JSONL generations (ignored)
├── docs/results/            Frozen summaries and publication artifacts
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

## Dataset setup and manifests

A manifest records the dataset source, immutable revision, licence, modality, field
mapping, splits, expected row counts, and SHA-256 checksums. Paths are relative to the
manifest file. Raw and processed datasets are intentionally excluded from version
control; never commit downloaded benchmark data or generated outputs.

Expected local layout:

```text
data/raw/
├── hallusionbench/
│   ├── HallusionBench.json
│   ├── VD/
│   └── VS/
└── retruthqa/
    ├── R1-7B/hallucination/*.jsonl
    └── R1-14B/hallucination/*.jsonl
```

Obtain ReTruthQA from the data link in the official
[Reasoning Hallucination repository](https://github.com/Jeryi-Sun/Reasoning_Hallucination).
Obtain `HallusionBench.json` from the official
[HallusionBench repository](https://github.com/tianyi-lab/HallusionBench) and place the
corresponding `VD/` and `VS/` assets beside it. The committed manifests pin the exact
JSON checksums and will reject a different or incomplete release.

Validate the immutable raw inputs:

```bash
dissertation-eval validate-manifest data/manifests/retruthqa.json
dissertation-eval validate-manifest data/manifests/hallusionbench.json
```

The ReTruthQA raw release has an unusual mixed format: R1-7B files are JSONL, while
R1-14B files are JSON arrays despite their `.jsonl` suffix. The manifest records the
correct parser for every split. Do not reformat the raw files because that changes
their checksums.

### ReTruthQA preparation

Generation uses a small, materialized dataset instead of extracting answers from large
reasoning traces on every run:

```bash
dissertation-eval prepare-retruthqa data/manifests/retruthqa.json \
  --output-dir data/processed/retruthqa/v1
dissertation-eval validate-prepared data/processed/retruthqa/v1
```

The extractor reads the final balanced `\boxed{...}` answer from every positive path,
normalizes it by domain, and accepts normalized consensus or a strict majority. Ties
are marked ambiguous and fail validation. Candidate votes, confidence, checksums, and
extractor version remain in the derived records and preparation manifest.

Preparation is cached by source-manifest checksum and extractor version. An unchanged
directory reports `"status": "reused"` without scanning the raw traces again. Use
`--force` only when intentionally rebuilding derived files.

### Frozen dissertation selection

The confirmatory text study uses the committed, deterministic 240-example selection in
`data/selections/retruthqa-240-v1.json`. It contains 60 development examples and 180
held-out test examples, with equal mathematics, science, and multi-hop counts in each
role. The held-out test excludes every example from the previously observed
`r1-7b_math_task1` pilot split.

Materialize and validate it from the prepared ReTruthQA v1 data with:

```bash
dissertation-eval select-retruthqa \
  data/processed/retruthqa/v1/manifest.json \
  --selection-manifest data/selections/retruthqa-240-v1.json \
  --output-dir data/processed/retruthqa/selection-240-v1
dissertation-eval validate-manifest \
  data/processed/retruthqa/selection-240-v1/manifest.json
```

If the prepared source checksum, deterministic policy, seed, or selected IDs differ
from the committed manifest, selection fails instead of silently replacing it. Generated
development/test JSONL files remain ignored because they contain benchmark prompts;
the committed ID manifest is sufficient to reproduce them from the prepared data.

## Generation records

Every attempted generation produces one JSONL record, including failures. Important fields include:

- stable `record_id`, `run_id`, dataset/split/sample IDs and group ID;
- exact model revision, backend, decoding configuration, prompt and seed;
- image path and reference answer where applicable;
- response, token IDs, finish reason, latency and peak memory;
- dataset metadata and a structured error with traceback.

The adjacent `*.run.json` file fingerprints the run configuration and environment. A run can resume only when that fingerprint is unchanged. This prevents results from two configurations being silently mixed.

## Local CPU validation

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

Validate all 12 ReTruthQA splits, prepare/reuse the derived dataset, and smoke-test
eight real questions with:

```bash
RESET_SMOKE_OUTPUT=1 bash scripts/cpu/smoke_retruthqa.sh
```

The smoke scripts write ignored records under `outputs/`. Re-running without
`RESET_SMOKE_OUTPUT=1` must report the records as skipped, confirming resumability.

### Small baseline on a Slurm GPU cluster

Complete CPU/mock validation before submitting a GPU job. On the cluster:

1. Clone the repository and create the Conda environment described above.
2. Install the `hf` dependencies and the cluster-compatible PyTorch build.
3. Use the committed, revision-pinned Qwen3-4B smoke configuration. It limits
   greedy generation to 256 new tokens so the first allocation verifies the pipeline
   without spending unnecessary GPU time:

```bash
sed -n '1,240p' configs/run.qwen3-4b.smoke.json
```

4. Copy the raw dataset directories through the university-approved storage mechanism;
   do not use Git for benchmark data.
5. Validate the raw manifests and prepare ReTruthQA on the cluster:

```bash
dissertation-eval validate-manifest data/manifests/retruthqa.json
dissertation-eval prepare-retruthqa data/manifests/retruthqa.json \
  --output-dir data/processed/retruthqa/v1
dissertation-eval validate-prepared data/processed/retruthqa/v1
```

6. Confirm that the Hugging Face dependencies are installed. The Slurm job will fail
   early if the active PyTorch installation cannot access CUDA. Then cache the pinned
   model revision on the login node so model download time is not charged as GPU time:

```bash
pip install -e '.[hf]'
bash scripts/slurm/cache_qwen_model.sh
```

7. Submit an eight-example baseline smoke test. The supplied script follows the NTU
   Cluster02 requirement to name an exact GPU model and requests one A40:

```bash
CONFIG_PATH=configs/run.qwen3-4b.smoke.json \
MODULE_NAME=Miniforge3/24.11.3-1 \
CONDA_ENV_NAME=dissertation-eval \
SMOKE_LIMIT=8 \
bash scripts/slurm/submit_baseline.sh
```

The batch job loads `Miniforge3/24.11.3-1`, activates `dissertation-eval`, checks
CUDA availability, and then runs generation. Do not increase `SMOKE_LIMIT` until the
JSONL records, GPU memory, parser output, and Slurm logs have been inspected. If NTU
requires an account or partition for your allocation, add the corresponding `#SBATCH`
directive before submission.

If the 256-token smoke run succeeds but reaches the limit on every record, run the
separate three-example 1024-token pilot without modifying or mixing the smoke output:

```bash
CONFIG_PATH=configs/run.qwen3-4b.pilot.json \
SMOKE_LIMIT=3 \
bash scripts/slurm/submit_baseline.sh
```

Generation records report `finish_reason: "length"` when the model reaches
`max_new_tokens`, and `finish_reason: "stop"` when an EOS token ends generation.
If all three pilot records still end with `length`, probe one example at 2048 tokens
before choosing a budget for a larger run:

```bash
CONFIG_PATH=configs/run.qwen3-4b.probe-2048.json \
SMOKE_LIMIT=1 \
bash scripts/slurm/submit_baseline.sh
```

If the same example is still truncated at 2048 tokens, compare Qwen3's non-thinking
mode on that exact example rather than increasing the budget again. This is a separate
experimental condition with its own run ID and output:

```bash
CONFIG_PATH=configs/run.qwen3-4b.no-thinking-probe.json \
SMOKE_LIMIT=1 \
bash scripts/slurm/submit_baseline.sh
```

If that response reaches its 512-token limit, repeat the same controlled condition at
1024 tokens. The prompt and all other settings remain identical:

```bash
CONFIG_PATH=configs/run.qwen3-4b.no-thinking-1024-probe.json \
SMOKE_LIMIT=1 \
bash scripts/slurm/submit_baseline.sh
```

Do not infer dataset-wide behaviour from the first record. The generation CLI supports
a deterministic shuffle before applying the limit. A larger limit with the same seed
preserves the existing sampled prefix. Run the five-example representative pilot with:

```bash
CONFIG_PATH=configs/run.qwen3-4b.representative-pilot.json \
SMOKE_LIMIT=5 \
SAMPLE_SEED=2026 \
bash scripts/slurm/submit_baseline.sh
```

The sample-order strategy and seed are stored in the adjacent run manifest. Resuming
an output with a different seed is rejected.

### ReTruthQA answer extraction and evaluation

Evaluate a generation JSONL on CPU after the run completes:

```bash
dissertation-eval evaluate-retruthqa \
  outputs/retruthqa-qwen3-4b-representative-pilot-v1.jsonl \
  --output-dir evaluations/retruthqa-qwen3-4b-representative-pilot-v1
```

The evaluator uses the last complete balanced `\\boxed{...}` answer first and falls
back to common `FINAL`/`Final answer` markers. It normalizes answers using the prepared
record's domain. Truncated generations and parse failures are not silently scored as
wrong; mismatches remain flagged for manual review. The output directory contains
auditable `records.jsonl` and `summary.json` files with the input checksum and evaluator
version.

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

## Read-only RHD diagnostics

The primary RHD method is a detector, not a decoding intervention. Its frozen v1
definition, layer policy, threshold policy, formulas, and pseudocode are documented in
`docs/rhd-method-v1.md`. Run a three-record, label-blind GPU diagnostic over an existing
generation file with:

```bash
GENERATIONS_PATH=outputs/qwen3-4b-greedy-development-v1.jsonl \
OUTPUT_DIR=diagnostics/qwen3-4b-greedy-development-rhd-smoke-v1 \
DIAGNOSTIC_LIMIT=3 \
bash scripts/slurm/submit_rhd_diagnostic.sh
```

The job reconstructs the exact recorded token sequence and performs a teacher-forced
forward pass. It writes per-record compressed sidecars containing token-level
inter-layer distribution summaries, plus SHA-256 checksums and a run manifest. It does
not call generation, use benchmark labels, or alter any response.
