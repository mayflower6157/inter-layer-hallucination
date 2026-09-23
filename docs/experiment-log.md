# Experiment log

This is the version-controlled audit log for dissertation experiments. Generated JSONL
records and evaluation outputs remain outside Git, but every reported run should be
traceable here to a code commit, configuration, dataset split, model revision, Slurm job,
and evaluation summary. Cluster timestamps may use the server timezone rather than
Asia/Singapore.

## 2026-09-23/24 — Qwen3-4B decoding-strategy validation

### Purpose

Validate an end-to-end comparison of greedy decoding, beam search, and stochastic
sampling before freezing the 240-example ReTruthQA study. This run is an engineering
validation on one complete split, not a final scientific result.

### Frozen inputs and controls

- Dataset: prepared ReTruthQA v1.
- Split: `r1-7b_math_task1`, all 44 examples.
- Model: `Qwen/Qwen3-4B`.
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
- Prompt: `Solve the problem in at most five concise steps. Put the final answer after FINAL:.`
- Thinking mode: disabled.
- Precision: bfloat16.
- Maximum new tokens: 1,024.
- Hardware: one NVIDIA A40 per Slurm job, four CPUs, 40 GiB host memory.
- Cluster software: `Miniforge3/24.11.3-1`; observed Python 3.11 environment,
  PyTorch `2.14.0+cu130`, Transformers `5.17.0`.
- Code/configuration commit introducing the suite: `ab001b9`.
- Submission compatibility fix: `32b931e` (use system `python3` for the login-node
  manifest check before the batch environment is activated).
- Local verification before submission: 27 unit tests passed; all JSON files parsed;
  Slurm scripts passed `bash -n`; `git diff --check` passed.

### Decoding configurations

| Strategy | Configuration | Generations per example | Seeds | Output |
|---|---|---:|---|---|
| Greedy | `configs/run.qwen3-4b.greedy-r1-7b-math-task1.json` | 1 | 42 | `outputs/qwen3-4b-greedy-r1-7b-math-task1-v1.jsonl` |
| Beam search | `configs/run.qwen3-4b.beam4-r1-7b-math-task1.json`; four beams | 1 | 42 | `outputs/qwen3-4b-beam4-r1-7b-math-task1-v1.jsonl` |
| Sampling | `configs/run.qwen3-4b.sampling5-r1-7b-math-task1.json`; temperature 0.7, top-p 0.9 | 5 | 101, 202, 303, 404, 505 | `outputs/qwen3-4b-sampling5-r1-7b-math-task1-v1.jsonl` |

All strategies use the same model revision, examples, prompt, token budget, precision,
and answer evaluator. Sampling should produce 220 records in total. Its five records per
sample will later support both single-sample evaluation and majority-vote
self-consistency.

### Submission and operational notes

The first suite submission attempt pulled commit `ab001b9` successfully but stopped
before `sbatch`: the login shell did not provide a `python` executable until the
Miniforge module was loaded. `/usr/bin/python3` was available, so the manifest precheck
was changed to use configurable `PYTHON_BIN=python3` in commit `32b931e`. No GPU job or
generation output was created by the failed submission attempt.

The corrected submission created the following jobs:

| Job ID | Job name | Requested limit | Last recorded state | Progress/timing |
|---:|---|---:|---|---|
| 168561 | `q3-4b-greedy` | 01:00:00 | COMPLETED, exit 0 | 44/44 successes; 00:14:02 |
| 168562 | `q3-4b-beam4` | 03:00:00 | COMPLETED, exit 0 | 44/44 successes; 00:14:03 |
| 168563 | `q3-4b-sample5` | 04:00:00 | COMPLETED, exit 0 | 220/220 successes; 01:08:50 |

Greedy and beam search ran concurrently on separate A40 nodes. The sampling job was
initially pending with `QOSMaxJobsPerUserLimit` and started as soon as one of the first
two jobs completed. This indicates an effective concurrency limit of two GPU jobs for
the account at that time.

The logs contain a Transformers deprecation warning for `torch_dtype`; it did not cause
a failure or change the recorded configuration. No CUDA out-of-memory or generation
errors were observed in the completed jobs.

Generation is append-only and resumable. Job 168563 completed without requiring a
resume. A future interrupted run can be resubmitted with the same configuration and
will skip completed `(sample_id, generation_index)` pairs rather than overwrite them.

### Evaluation results

Evaluation used `dissertation-eval compare-retruthqa` from commit `61b868f` and
ReTruthQA evaluator version `1.0.2`. The derived files are stored on the cluster under
`evaluations/qwen3-4b-decoding-validation-v1/`. The command writes strategy-specific
records and summaries, `comparison.json`, `comparison.md`, and per-sample
`self-consistency-records.jsonl`.

| Strategy | Records | Samples | Completion | Truncation | Parse coverage | Exact match (scored) | Exact match (all) | Mean tokens | Median latency (s) | Peak GPU (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Greedy | 44 | 44 | 86.4% | 13.6% | 100.0% | 68.4% | 59.1% | 487.0 | 16.1 | 7.68 |
| Beam-4 | 44 | 44 | 93.2% | 6.8% | 100.0% | 53.7% | 50.0% | 463.4 | 18.1 | 8.22 |
| Sampling-5, individual generations | 220 | 44 | 91.4% | 8.6% | 100.0% | 63.2% | 57.7% | 471.5 | 17.2 | 7.68 |
| Sampling-5 self-consistency | — | 44 | — | — | 95.5% vote coverage | 59.5% of predicted | 56.8% | 2,357.4 total | pending refresh | 7.68 |

Raw counts were:

- Greedy: 26 exact matches, 12 mismatches, and six truncations.
- Beam-4: 22 exact matches, 19 mismatches, and three truncations.
- Sampling-5: 127 exact matches, 74 mismatches, and 19 truncations across 220
  individual generations.
- Self-consistency: 25 exact matches across 44 samples; 42 samples had at least one
  valid parsed vote, 35 had a strict majority, and four required a tie break.

Self-consistency selects the most frequent normalized answer among completed, parsed
generations. If multiple answers share the highest count, the answer produced at the
lowest generation index is selected deterministically and the case is marked as a tie.
This rule prevents nondeterministic reporting, but tied cases must be included in a
sensitivity analysis for the confirmatory experiment.

Input SHA-256 values recorded by the evaluator were:

- Greedy: `86419afc2d652b5fbaf62895aa8178f76961ad224aa84fe68193afbc78882b61`.
- Beam-4: `213049063e90b3ddcba6836958e1358eecb4604df448dc6ce9140ebc14c6d372`.
- Sampling-5: `8cdfcab67b680f83fd55be2e5eca53f5bacd461e5dd9b377f290d2e23df6b2ae`.

On this 44-question mathematics validation split, greedy decoding had the highest
all-record and scored exact-match rates. Beam search reduced truncation but did not
improve exact match, and five-sample self-consistency did not exceed greedy despite
approximately 5.3 times the mean end-to-end latency per question. These observations
are pilot findings only: the split is small and covers one domain, no confidence
intervals or paired significance tests have been applied, and evaluator mismatches have
not yet received a blinded error audit.

### Required completion checks

Do not treat this validation run as complete until the following fields are filled from
machine-readable evaluation outputs:

- [x] Sampling job completed with 220/220 successful generation records; no resume was
  required.
- [x] Run `evaluate-retruthqa` separately for greedy, beam, and sampling outputs.
- [x] Record exact-match rate over all records and over scored, non-truncated records.
- [x] Record completion, truncation, parse-failure, and generation-error counts.
- [x] Record generated-token and latency mean/median; retain peak-memory summaries.
- [x] Aggregate sampling predictions by `sample_id` and report five-sample majority-vote
  self-consistency accuracy, including the deterministic tie rule.
- [ ] Inspect disagreements among greedy, beam, and self-consistency before freezing the
  larger study; do not tune prompts or evaluators separately by strategy.

### Planned use in the dissertation

This validation supplies implementation evidence and realistic compute estimates for
the experimental-setup and efficiency sections. Accuracy values from 44 mathematics
examples must be labelled as pilot results. The confirmatory text experiment will use
a version-controlled 240-example selection spanning all 12 ReTruthQA splits: five
development and 15 held-out test examples per split. RHD layers, normalization, and any
thresholds will be selected on the 60-example development portion and frozen before the
180-example text test and text-to-multimodal transfer evaluation.
