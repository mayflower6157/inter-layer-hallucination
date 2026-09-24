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
| Sampling-5 self-consistency | — | 44 | — | — | 95.5% vote coverage | 59.5% of predicted | 56.8% | 2,357.4 total | 88.8 total | 7.68 |

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
the version-controlled 240-example selection documented below. RHD layers,
normalization, and any thresholds will be selected on its 60-example development
portion and frozen before the 180-example text test and text-to-multimodal transfer
evaluation.

## 2026-09-24 — Frozen ReTruthQA 240-example selection

The confirmatory selection is committed as
`data/selections/retruthqa-240-v1.json`. It was generated by selection implementation
version `1.0.0` with seed `20260924` from the prepared ReTruthQA v1 manifest. Selection
within each source split first sorts by sample ID and then uses a SHA-256-derived
per-role, per-split random seed.

The design deliberately treats the full 44-example `r1-7b_math_task1` split as observed
pilot data. Twenty of those examples enter development; none enter the held-out test.
Development examples are selected before test examples, and are removed from every
shared source-split candidate pool before test sampling.

| Role | Total | Mathematics | Science | Multi-hop QA | Pilot samples in test |
|---|---:|---:|---:|---:|---:|
| Development | 60 | 20 | 20 | 20 | — |
| Held-out test | 180 | 60 | 60 | 60 | 0 |

Validation confirmed 60 unique development IDs, 180 unique test IDs, zero overlap
between roles, and zero overlap between all 44 observed pilot IDs and held-out test IDs.
The committed selection-manifest SHA-256 is
`ccdf50c34e7c7215f7e8d19d924edca726855d2db18b3f475533b21d2720fae6`.

The earlier plan of taking five development and 15 test examples from every split was
revised before formal generation because it would have placed already observed pilot
questions into the held-out test. The frozen allocation instead uses explicit
source-split quotas recorded in the selection manifest while preserving equal domain
counts.

## 2026-09-24 — Qwen3 development decoding baselines submitted

The frozen 60-example development split was configured for Qwen3-2B and Qwen3-4B with
identical prompts and decoding settings. Both model revisions are pinned. Greedy and
four-beam decoding produce one generation per question; sampling uses temperature 0.7,
top-p 0.9, and five fixed seeds. All conditions use non-thinking mode and a 1,024-token
generation limit.

The jobs were submitted as a dependency chain so that the 2B conditions finish before
the 4B conditions. Initial Slurm IDs are: 2B greedy `168729`, 2B beam `168730`, 2B
sampling-5 `168731`, 4B greedy `168732`, and 4B beam `168733`. Cluster QOS permits only
five submitted jobs per user, so 4B sampling-5 is deferred until the first slot is
released; it must depend on successful completion of job `168733`.

The first job started successfully on an NVIDIA A40. Preflight confirmed CUDA and the
pinned Qwen3-2B weights loaded correctly. Transformers reported reference-kernel
fallbacks for optional `causal_conv1d` and `flash-linear-attention` packages; these are
performance warnings rather than correctness failures and may increase runtime.

All six jobs subsequently completed with exit code `0:0`. The server-side deferred
submission watcher assigned job `168757` to 4B sampling-5. Output integrity checks found
60 unique and successful questions for each greedy/beam condition and 300 successful
records, comprising all five generation indices for every one of 60 questions, for each
sampling condition.

| Model | Strategy | Slurm job | Elapsed | Records |
|---|---|---:|---:|---:|
| Qwen3-2B | Greedy | `168729` | 00:23:17 | 60 |
| Qwen3-2B | Beam-4 | `168730` | 00:24:22 | 60 |
| Qwen3-2B | Sampling-5 | `168731` | 01:48:23 | 300 |
| Qwen3-4B | Greedy | `168732` | 00:12:00 | 60 |
| Qwen3-4B | Beam-4 | `168733` | 00:12:51 | 60 |
| Qwen3-4B | Sampling-5 | `168757` | 00:53:57 | 300 |

The automatic evaluator produced the consolidated development table in
`docs/results/development-decoding-baseline-v1.md`. Qwen3-4B sampling-5
self-consistency achieved the highest all-question exact match (55.0%). Qwen3-2B
self-consistency reached 50.0%, but only 85.0% of questions had at least one valid
completed vote because the 2B runs had substantially more 1,024-token truncations.
Completed-output parse coverage was 100% for every individual strategy.

The 2B comparison contains completed/exact counts of 47/24 (greedy), 48/23
(beam-4), and 217/122 (300 individual samples); its self-consistency result contains
51 predictions and 30 exact matches across 60 questions. The 4B counts are 57/30,
59/29, and 294/146, with 60 self-consistency predictions and 33 exact matches. The
SHA-256 checksums of the machine-readable `comparison.json` artifacts are
`eb702c658e470f46f2efbffa04bfa8516b89abcd15cd590ca5696b2053518076` (2B) and
`bd04789ac6e3c619c77095ee2c026e2512d014442c488118a2a5168dae128fcc` (4B).

## 2026-09-24 — Read-only RHD inter-layer diagnostic smoke test

RHD v1 was specified as a post-hoc, label-blind detector rather than a decoding
intervention. The frozen method definition is recorded in `docs/rhd-method-v1.md`.
The first diagnostic pass projects hidden states at normalized candidate depths
0.50, 0.625, 0.75, and 0.875 and compares their vocabulary distributions with the
final-layer distribution using Jensen–Shannon divergence. It does not alter prompts,
generated token IDs, or decoding decisions.

Slurm job `168967` ran the diagnostic on the first three records of the Qwen3-4B
greedy development output and completed with exit code `0:0` in 00:01:17. For the
36-block Qwen3-4B model, the candidate layers resolved to `[18, 23, 27, 32]`, with
layer 36 as the final anchor. The output directory is
`diagnostics/qwen3-4b-greedy-development-rhd-smoke-v1` on the server.

All three records produced finite, non-constant token-level scores, all observed JSD
values were within the theoretical interval `[0, ln(2)]`, and all sidecar SHA-256
checksums validated. The source generation file's SHA-256 was unchanged before and
after diagnosis. Across the three records, the mean token-level reasoning score was
0.54656 and the observed record-mean range was 0.14029–0.69306. Candidate-layer mean
JSD decreased toward the final layer in every record; for the first record the means
were 0.6816, 0.6560, 0.5342, and 0.3383 at layers 18, 23, 27, and 32 respectively.
The diagnostic manifest SHA-256 is
`a975682c146ffd61801545060073f7e5b2d3d591a968f58d8b05ee1dea083aac`.

This smoke test establishes extraction integrity and a measurable layer- and
token-dependent signal. It does not establish that the signal discriminates correct
from hallucinated reasoning: that claim requires frozen step segmentation and a
larger development-only comparison containing both correct and incorrect outputs.

## 2026-09-24 — RHD step features and development layer-grid study

The reasoning-step rule was frozen as `paragraph_blank_line_v1`: two or more newlines
with optional intervening whitespace delimit non-empty steps. This rule is label-blind
and requires no manual annotation. Diagnostic version 1.1.1 records the four released
RHD features in the order attention score, early-step coefficient of variation, mean
reference-compatible Reasoning Score, and correlation of first-differenced normalized
Reasoning Score and entropy. Exact whole-sequence decoding and tokenizer offsets were
validated after an initial token-by-token decoding attempt failed the text-identity
audit; the corrected three-record job `168973` achieved 3/3 exact text matches.

Job `168977` used one A40 and completed three 30-record Qwen3-4B Greedy layer grids
sequentially in 00:02:04. Of the prefix, 27 completed/scored traces were usable: 11
correct and 16 incorrect. The best univariate feature for every grid was PCC.

| Grid | Qwen3-4B layers | Attention AUROC | CV AUROC | Mean-R AUROC | PCC AUROC |
|---|---|---:|---:|---:|---:|
| Sparse | 18, 23, 27, 32 | 0.491 | 0.341 | 0.364 | 0.705 |
| Reference-7B relative pattern | 18, 21, 23, 26, 28, 31, 33 | 0.483 | 0.330 | 0.386 | 0.739 |
| Reference-14B relative pattern | 24, 27, 30, 32, 33, 34 | 0.651 | 0.381 | 0.358 | 0.773 |

With 20,000 fixed-seed label permutations, the one-sided PCC p-values were 0.03875,
0.01755, and 0.00875 respectively. Because the same 30-record prefix selected the
grid, these values are exploratory rather than confirmatory.

The reference-14B relative pattern was then frozen and run over all 60 development
records in job `168980` (00:01:02). Evaluation restricted to the previously unused
last 30 records, containing 19 correct and 11 incorrect traces. Confirmation AUROCs
were 0.567 attention, 0.292 CV, 0.297 mean-R, and 0.557 PCC. Thus the apparent prefix
advantage did not replicate clearly above chance. The candidate grid is not promoted
to held-out testing. The next development work must investigate step construction,
score direction/calibration, and the mismatch between pre-segmented reference paths
and free-form Qwen3 generations without inspecting the held-out test split.

Job `168979` is excluded: Slurm's comma-delimited `--export` retained only the first
requested layer. The submission scripts now translate layer lists to colon-delimited
values before export; job `168980` verified all six intended layers.

## 2026-09-24 — RHD segmentation, confound, and trajectory audit

The full 60-record Qwen3-4B Greedy development diagnostic was joined to generation
metadata and automatic exact-match labels. Sidecar checksums were verified before
analysis. The audit contains 30 correct, 27 incorrect, and three unscored truncated
traces and is committed under `docs/results/rhd-qwen3-4b-development-audit-v1`.

Paragraph segmentation was not dominated by degenerate one-step traces: the median was
six steps, the mean 8.72, and only 6.7% of responses contained one or two steps. The
median step length was 26 tokens, although the maximum of 287 tokens shows that a
sentence-level fallback may still be useful for a minority of long paragraphs.

Across all scored traces, AUROC was 0.609 attention, 0.330 CV, 0.626 PCC, and 0.342
mean Reasoning Score. The low direct Mean-R AUROC corresponds to higher Mean-R among
correct traces (mean 2.463) than incorrect traces (mean 2.319), so its direction is
opposite to a naive “larger means more hallucination” assumption. CV is substantially
confounded by step count (Pearson r=0.581) and token count (r=0.482).

The strongest finding is heterogeneity rather than a single stable global detector.
Attention AUROC was 0.817 for MATH, 0.510 for MultiHopQA, and 0.352 for Science; PCC
AUROC was respectively 0.385, 0.610, and 0.769. Attention also varied strongly with
length: 0.385 short, 0.628 medium, and 0.851 long. These results indicate that domain
and length effects can explain the unstable half-split confirmation. Before changing
step rules or entering held-out evaluation, the next development-only analysis should
compare the detector with length/step-count baselines and use domain-grouped or
domain-stratified cross-validation. Any direction changes, feature weights, and
thresholds must be learned inside training folds.

## 2026-09-24 — Domain-stratified RHD cross-validation

A deterministic four-fold cross-validation was run over the 57 scored Qwen3-4B
Greedy development traces. Four folds are the maximum that place examples from every
available domain×correctness stratum across folds because MATH contains only four
incorrect traces. Within each training fold, features were standardized, feature
directions and L2-regularized logistic weights were learned, and an operating threshold
was chosen by balanced accuracy with recall as the tie-breaker. No validation-fold
statistics contributed to preprocessing, fitting, or threshold selection.

| Model | OOF AUROC | OOF AUPRC | OOF balanced accuracy | Fold AUROC mean ± SD |
|---|---:|---:|---:|---:|
| Token/step-count baseline | 0.620 | 0.606 | 0.591 | 0.633 ± 0.060 |
| RHD four features | 0.665 | 0.575 | 0.619 | 0.661 ± 0.098 |
| RHD + token/step count | 0.642 | 0.557 | 0.581 | 0.693 ± 0.123 |

RHD-only improves OOF AUROC by 0.045 and balanced accuracy by 0.028 relative to the
length/step baseline, providing preliminary evidence of signal beyond simple response
length. However, adding the nuisance variables did not improve pooled OOF performance,
and fold variability remains substantial. Learned directions were most consistent for
negative CV and negative mean Reasoning Score; attention and PCC directions varied in
some folds, matching the earlier domain heterogeneity. These development results do not
yet justify a held-out claim. The machine-readable fold assignments, training-only
thresholds, weights, and metrics are committed in
`docs/results/rhd-qwen3-4b-development-cv-v1`.

## 2026-09-24 — Frozen RHD decoding and model-scale comparison

The frozen six-layer relative-depth configuration was applied without retuning to
Qwen3-4B Beam-4, Qwen3-4B Sampling-5, and Qwen3-2B Greedy development generations.
For the 36-layer 4B model this selected layers 24, 27, 30, 32, 33, and 34; for the
24-layer 2B model it selected layers 16, 18, 20, 21, 22, and 23. The same automatic
answer labels, paragraph step rule, four RHD features, and training-fold-only
cross-validation procedure were used throughout.

Job `168995` completed the 4B Beam-4 and Sampling-5 diagnostics, then stopped on the
first 2B trace because the tokenizer's legal decode/re-encode sequence was not
bijective. The incomplete 2B artifact from this job was excluded. Diagnostic version
1.1.2 added a cumulative-prefix offset fallback, preserving exact full-text validation.
Job `169000` then completed the clean 2B diagnostic in 00:01:44 with exit code zero.

| Condition | Scored traces | Length/step AUROC | RHD AUROC | Difference | Combined AUROC | RHD balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B Greedy | 57 | 0.620 | 0.665 | +0.045 | 0.642 | 0.619 |
| Qwen3-4B Beam-4 | 59 | 0.476 | 0.759 | +0.283 | 0.722 | 0.678 |
| Qwen3-4B Sampling-5 | 294 | 0.576 | 0.685 | +0.109 | 0.677 | 0.613 |
| Qwen3-2B Greedy | 47 | 0.641 | 0.665 | +0.024 | 0.683 | 0.660 |

Sampling-5's five traces for each question were kept in the same fold, preventing the
same prompt and reference answer from appearing in both training and validation.
Beam-4 produced the clearest RHD gain beyond token and step count. Sampling-5 retained
a smaller gain, while the 2B Greedy incremental gain was modest and benefited from the
combined nuisance-plus-RHD model. The development evidence therefore indicates that
RHD behavior is sensitive to decoding strategy and model scale; it does not support a
claim of stable universal generalization. Full fold metrics and provenance are archived
under `docs/results/rhd-cross-model-decoding-comparison-v1.md` and its linked condition
directories.

## 2026-09-24 — Completion of the 2B decoding matrix

To complete the model-scale × decoding-strategy development matrix, job `169270` used
one A40 to extract frozen read-only RHD diagnostics for Qwen3-2B Beam-4 (60 traces) and
Sampling-5 (300 traces). Existing generation outputs were reused, so no text generation
was repeated. Answer evaluation, diagnostics, and grouped cross-validation completed in
00:08:54 with exit code zero. The same six relative layer depths mapped to layers 16,
18, 20, 21, 22, and 23.

Beam-4 yielded 48 scored traces (23 correct and 25 incorrect; 12 truncated). Sampling-5
yielded 217 scored traces (122 correct and 95 incorrect; 83 truncated). Sampling traces
from the same question remained in a single fold.

| Qwen3-2B condition | Scored traces | Length/step AUROC | RHD AUROC | Difference | Combined AUROC | RHD balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Beam-4 | 48 | 0.477 | 0.692 | +0.215 | 0.670 | 0.666 |
| Sampling-5 | 217 | 0.472 | 0.458 | -0.014 | 0.438 | 0.565 |

The completed 2×3 matrix strengthens the decoding-sensitivity finding. At both model
scales, Beam-4 exposes a clearer RHD signal than Sampling-5. For Qwen3-2B Sampling-5,
RHD does not improve over token and step count. This is evidence against decoding
invariance and motivates analyzing stochastic trajectories rather than forcing a
universal RHD detector claim.

## 2026-09-24 — Sampling-5 truncation and selection-bias audit

A development-only CPU audit compared all 300 paired Qwen3-2B and Qwen3-4B Sampling-5
traces. The generation sanity check passed: both conditions used the same prompt,
1,024-token limit, temperature 0.7, top-p 0.9, five seeds, shared Transformers backend,
and shared finish-reason implementation. Every length termination occurred at exactly
1,024 generated tokens. The model-specific empirical stop IDs differed (248044 for 2B
and 151645 for 4B), as expected from their tokenizer vocabularies; no record-level
configuration mismatch was found.

| Condition | Truncated traces | Questions with ≥1 truncation | Questions with all 5 truncated |
|---|---:|---:|---:|
| Qwen3-2B Sampling-5 | 83/300 (27.7%) | 27/60 | 9/60 |
| Qwen3-4B Sampling-5 | 6/300 (2.0%) | 2/60 | 0/60 |

Qwen3-2B truncation was strongly domain-dependent: Science 49/100, MATH 32/100, and
MultiHopQA 2/100. It was not concentrated in one seed; seed-specific rates ranged from
25.0% to 31.7%. Qwen3-4B truncated only six MATH traces. Among 300 paired
`(sample_id, generation_index)` outputs, 217 completed in both models, 77 truncated
only for 2B, and six truncated for both; there were no 4B-only truncations. Relative
to 2B, the 4B response used a median 207.5 fewer tokens but two more paragraph steps.

Among completed traces, the strongest individual feature was mean Reasoning Score for
both 2B (descriptive AUROC 0.626, clustered 95% CI 0.497–0.747) and 4B (0.648,
0.547–0.747). These intervals overlap. At the composite cross-validated level,
however, 2B Sampling-5 RHD remained below its length/step baseline (0.458 versus
0.472), whereas 4B RHD exceeded its baseline (0.685 versus 0.576). Thus simple
univariate feature separation does not recover stable 2B composite generalization.

Treating truncation as its own endpoint, rather than assigning it a correctness label,
showed only modest feature association for 2B (maximum descriptive AUROC 0.593 for
PCC). The six 4B truncations were highly separable by CV and mean Reasoning Score, but
that estimate is too sparse for a stable claim. The high 2B truncation rate and nine
entirely missing questions create material selection bias, especially through the
Science and MATH domain mixture. Nevertheless, the modest 2B truncation-endpoint
association indicates that selection alone is unlikely to explain the entire weak
completed-only composite RHD result.

All bootstrap confidence intervals resampled `sample_id`, keeping the five generations
of a question together. Full configuration checks, domain/seed tables, token and step
distributions, ten-bin trajectories, feature AUROCs, figures, and machine-readable
results are archived in `docs/results/rhd-sampling5-truncation-audit-v1`.

## 2026-09-24 — Protocol freeze and held-out preparation

The development phase was formally frozen as `rhd-frozen-protocol-v1`. A reproducible
2×3 AUROC matrix and Sampling-5 truncation comparison figure were generated from the
committed machine-readable development results. The confirmatory questions, generation
settings, layer mapping, segmentation, features, metrics, clustered uncertainty rule,
and truncation policy are fixed in `docs/results/rhd-frozen-protocol-v1/protocol.md`.

One final calibration was fitted on all scored development traces for each model ×
decoding cell. Each artifact contains the feature order, centers, scales, logistic
weights, intercept, and development-selected threshold for the RHD-only, token/step,
and combined models. Serialized reload reproduced development scores with maximum
absolute error between 1.11e-16 and 2.22e-16. The artifacts explicitly require
`apply_only_no_refit_no_threshold_selection` on held-out data.

Six held-out generation configs and a two-lane Slurm submission script were prepared.
Each lane uses one A40 and runs Greedy → Beam-4 → Sampling-5 sequentially for one model,
allowing at most two concurrent A40 allocations. The jobs were not submitted during
this preparation step, so the 180-example held-out split remains unobserved.
## 2026-09-24 — Frozen held-out RHD evaluation

All six held-out generation cells and all six read-only RHD diagnostic cells completed
successfully. The final evaluator loaded the development-only calibrations without refitting
feature direction, centering, scaling, logistic weights, or thresholds. Truncated and unparsed
records were excluded from correctness-based discrimination metrics and retained as separate
generation endpoints.

RHD-only held-out AUROC was 0.636, 0.617, and 0.630 for Qwen3-2B Greedy, Beam-4, and Sampling-5,
respectively, and 0.642, 0.574, and 0.609 for the corresponding Qwen3-4B cells. Combined-model
AUROC gains over the frozen token/step baseline were 0.111, 0.081, 0.036, 0.097, 0.046, and
0.043 in the same order. Domain-level estimates were heterogeneous; for example, Qwen3-2B
Greedy RHD AUROC ranged from 0.472 on Science to 0.713 on MultiHopQA.

Aggregate artifacts are stored under `docs/results/rhd-heldout-final-v1/`. Complete per-trace
scores and diagnostic sidecars remain on the NTU server. Two thousand bootstrap replicates were
stratified by domain and clustered by `sample_id`. Only the Qwen3-2B Greedy gain (0.111, 95% CI
0.016--0.208) and Qwen3-4B Greedy gain (0.097, 0.027--0.174) excluded zero. All Beam-4 and
Sampling-5 incremental-gain intervals crossed zero; the final claim is therefore conditional
rather than decoding-invariant.
