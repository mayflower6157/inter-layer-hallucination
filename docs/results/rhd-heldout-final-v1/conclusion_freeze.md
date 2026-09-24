# Held-out conclusion freeze

## Statistical procedure

Confidence intervals use 2,000 deterministic bootstrap replicates. Resampling is
stratified by ReTruthQA domain and clustered by `sample_id`; consequently, all five
Sampling-5 traces for one question are sampled together. The gain interval is paired
within each replicate and measures combined-model AUROC minus length/step-baseline AUROC.
No held-out transform, feature direction, weight, or threshold is fitted or selected.

## Confirmed conclusions

1. **Incremental signal is supported for Greedy only.** Qwen3-2B Greedy has a paired
   gain of 0.111 (95% CI 0.016 to 0.208), and Qwen3-4B Greedy has a gain of 0.097
   (0.027 to 0.174). These are the only cells whose gain intervals exclude zero.
2. **The evidence is decoding-dependent.** Beam-4 and Sampling-5 have positive point
   estimates, but all four gain intervals include zero. The results therefore reject a
   decoding-invariant interpretation of incremental RHD value.
3. **RHD-only discrimination is modest.** Held-out RHD AUROC ranges from 0.574 to 0.642.
   Its bootstrap interval excludes 0.5 for both Greedy cells and both Sampling-5 cells;
   Beam-4 is less certain, especially for Qwen3-4B (0.487 to 0.657).
4. **Domain heterogeneity remains material.** The widest example is Qwen3-2B Greedy,
   whose RHD AUROC is 0.713 on MultiHopQA but 0.472 on Science. Domain estimates are
   secondary because their effective sample sizes are small.

## Questions requiring qualified language

- **Greedy advantage:** the two Greedy cells independently show non-zero incremental
  gain. This supports Greedy as the most reliable held-out condition, but it is not a
  formal paired test that Greedy exceeds Beam-4 or Sampling-5.
- **Beam-4 development-to-held-out decline:** RHD AUROC changes from 0.692 to 0.617 for
  Qwen3-2B and from 0.759 to 0.574 for Qwen3-4B. The 2B held-out interval includes the
  development estimate. The 4B interval does not, but the 60-question development CV
  itself has high fold variability. The ranking reversal is evidence against selecting
  Beam-4 from development results, not a precise estimate of distribution shift.
- **Sampling truncation:** held-out truncation is 294/900 (32.7%) for Qwen3-2B and
  47/900 (5.2%) for Qwen3-4B. Sampling gain intervals include zero for both scales.
  Because correctness is unobserved for truncated traces, selection effects are a
  plausible contributor but cannot be separated causally from decoding variability.

## Frozen claim for the dissertation

> Development-fitted inter-layer RHD features provide modest held-out discrimination,
> but incremental value beyond response length and step count is established only for
> greedy decoding. Performance varies across decoding strategy, model scale, domain,
> and completion status; the results do not support decoding-invariant or universally
> calibrated reasoning-hallucination detection.
