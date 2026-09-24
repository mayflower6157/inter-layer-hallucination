# Frozen held-out RHD evaluation provenance

- Protocol source commit: `33d155b`
- Held-out evaluator commit: `4ac2490`
- Evaluation scope: 180-question sealed ReTruthQA test split
- Cells: Qwen3-2B and Qwen3-4B crossed with Greedy, Beam-4, and Sampling-5
- Calibration policy: `apply_only_no_refit_no_threshold_selection`
- Positive detection label: incorrect final answer
- Scored population: completed, parsed traces only
- Excluded endpoints: truncation, parse failure, and generation error
- RHD layers, step segmentation, feature definitions, scaling, logistic weights, and
  operating thresholds are loaded unchanged from the six development-only calibration files.

The versioned directory contains aggregate JSON, CSV, Markdown, and LaTeX artifacts only.
Complete generation outputs, token-level sidecars, and per-trace held-out scores remain on the
NTU server and are deliberately excluded from the repository.

Confidence intervals use 2,000 deterministic bootstrap replicates, stratified by domain and
clustered by `sample_id`, so all five Sampling-5 generations for a question remain in the same
resampling unit. The reported incremental-gain interval is paired within each replicate.
