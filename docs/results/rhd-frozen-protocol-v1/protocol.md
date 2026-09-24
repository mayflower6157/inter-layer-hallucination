# Frozen RHD protocol v1

Status: **frozen and execution-ready; held-out test remains sealed**.

## Confirmatory questions

1. Does the incremental RHD signal beyond token/step count vary by decoding strategy?
2. Is Beam-4 more separable than Sampling-5 at both Qwen3 model scales?
3. Does Qwen3-2B Sampling-5 retain a weaker RHD signal than Qwen3-4B Sampling-5 after
   reporting completion and truncation separately?

## Fixed experimental cells

The held-out experiment is a 2 × 3 design: Qwen3-2B and Qwen3-4B crossed with Greedy,
Beam-4, and Sampling-5. It uses the frozen 180-example held-out portion of
`selection-240-v1`; none of its labels may be used for fitting or protocol changes.
Sampling-5 uses five fixed seeds (101, 202, 303, 404, 505); Greedy and Beam use seed 42.

## Fixed generation and diagnostic settings

- Prompt: at most five concise steps, with the final answer after `FINAL:`.
- Maximum new tokens: 1,024 for all cells.
- Beam-4: four beams, deterministic decoding.
- Sampling-5: temperature 0.7, top-p 0.9, one beam, five generations.
- RHD mode: post-hoc, teacher-forced, read-only; it never changes generation.
- Relative layer requests: 2/3, 3/4, 5/6, 7/8, 11/12, and 23/24 of model depth.
- Qwen3-2B layers: 16, 18, 20, 21, 22, 23; anchor layer 24.
- Qwen3-4B layers: 24, 27, 30, 32, 33, 34; anchor layer 36.
- Step segmentation: `paragraph_blank_line_v1`.
- RHD features: attention score, CV score, mean reference-compatible Reasoning Score,
  and PCC, in the released order.

## Fixed fitting and evaluation rules

- Direction, centering, scaling, logistic weights, and operating threshold must be
  fitted once using development data only and serialized before held-out inference.
- The held-out split receives fixed development-fitted transforms. No test-fold CV,
  threshold selection, feature direction changes, or domain-specific retuning is allowed.
- Primary discrimination metric: held-out AUROC for RHD-only score.
- Required comparator: token/step-count baseline; report the AUROC difference.
- Secondary metrics: AUPRC and balanced accuracy at the frozen development threshold.
- Sampling-5 resampling and uncertainty use `sample_id` as the cluster; five generations
  from one question never cross analysis units.
- Correctness is evaluated only for completed, parsed traces. Truncation is reported as
  a separate endpoint at trace and question level; no correctness label is imputed.
- Report overall and domain-stratified results, but domain results are secondary because
  the held-out strata are small.

## Gate before held-out execution

The development diagnostics, AUROC matrix, truncation audit, and protocol are frozen.
One final all-development calibration has been serialized for each of the six cells
under `calibrations/`. Reloaded scores reproduce the original development scores with
maximum absolute error no larger than 2.22e-16. The held-out configurations and a
two-lane submission script are prepared, but submission requires a separate explicit
execution decision because it allocates up to two A40 GPUs and reveals test results.
