# Frozen RHD comparison across decoding strategies and model scales

This development-only comparison uses the frozen six-layer relative-depth pattern
`[0.6667, 0.7500, 0.8333, 0.8750, 0.9167, 0.9583]`, diagnostic schema 1.1.x,
`paragraph_blank_line_v1` step segmentation, and four-fold domain-stratified
cross-validation. Scaling, feature directions, logistic weights, and decision
thresholds are learned using each training fold only.

Sampling-5 contributes five traces per question. All traces sharing a `sample_id`
are assigned to the same fold, preventing prompt/reference leakage between training
and validation folds.

| Model / decoding | Scored traces | Length/step AUROC | RHD AUROC | Standalone contrast | Combined AUROC | Incremental gain | RHD balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B Greedy | 57 | 0.620 | 0.665 | +0.045 | 0.642 | +0.022 | 0.619 |
| Qwen3-4B Beam-4 | 59 | 0.476 | 0.759 | +0.283 | 0.722 | +0.246 | 0.678 |
| Qwen3-4B Sampling-5 | 294 | 0.576 | 0.685 | +0.109 | 0.677 | +0.101 | 0.613 |
| Qwen3-2B Greedy | 47 | 0.641 | 0.665 | +0.024 | 0.683 | +0.042 | 0.660 |
| Qwen3-2B Beam-4 | 48 | 0.477 | 0.692 | +0.216 | 0.670 | +0.193 | 0.666 |
| Qwen3-2B Sampling-5 | 217 | 0.472 | 0.458 | -0.015 | 0.438 | -0.034 | 0.565 |

The standalone contrast is RHD-only AUROC minus length/step AUROC. The incremental
gain is combined AUROC minus length/step AUROC; only the latter estimates the added
discrimination from RHD features after the nuisance variables are included.

## Interpretation

- Qwen3-4B Beam-4 provides the strongest incremental RHD signal: its combined-minus-
  baseline gain is 0.246. Its larger 0.283 RHD-only contrast is descriptive.
- Qwen3-4B Sampling-5 retains signal beyond the nuisance baseline, but its smaller
  increment and lower balanced accuracy than Beam-4 indicate decoding sensitivity.
- Qwen3-2B Greedy reaches an RHD-only AUROC of 0.665, but the gain over its stronger
  length/step baseline is only 0.024. The combined model performs best for this
  condition, with AUROC 0.683.
- Qwen3-2B Beam-4 again shows a substantial incremental RHD gain of 0.193, whereas
  Qwen3-2B Sampling-5 has a negative gain of -0.034. The Beam-versus-Sampling
  contrast therefore appears at both model scales.
- These results support studying how RHD changes with decoding and model scale. They
  do not establish decoding invariance, universal transfer, or held-out performance.

The individual reports and machine-readable fold records are stored in the sibling
result directories for each condition.
