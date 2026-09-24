# Development artifact provenance

- Frozen source commit: `33d155b50cf2b3850a4b8e9444b3449ff23592db`
- Branch: `main`
- Source checkout: `/home/kxu009/mayflower/Disseration` on the NTU EEE cluster
- Local verification date: 2026-09-24 (Asia/Singapore)
- Protocol status: development-fitted and frozen; held-out labels remain sealed
- Evaluator version recorded by the decoding comparisons: `1.0.2`

## Frozen development cells

The frozen design contains Qwen3-2B and Qwen3-4B crossed with Greedy, Beam-4, and Sampling-5. The tracked generation configurations are:

- `configs/run.qwen3-2b.greedy-development.json`
- `configs/run.qwen3-2b.beam4-development.json`
- `configs/run.qwen3-2b.sampling5-development.json`
- `configs/run.qwen3-4b.greedy-development.json`
- `configs/run.qwen3-4b.beam4-development.json`
- `configs/run.qwen3-4b.sampling5-development.json`

## Reproducible report artifacts

- Decoding comparisons and per-strategy summaries: `docs/results/qwen3-*-development-decoding-v1/`
- Cross-validation JSON and Markdown reports: `docs/results/rhd-*-development-cv-v1/`
- Development audits: `docs/results/rhd-qwen3-4b-development-audit-v1/` and `docs/results/rhd-sampling5-truncation-audit-v1/`
- Frozen protocol, AUROC matrix, truncation comparison, and six calibrations: this directory
- Figure generation: `scripts/analysis/render_rhd_frozen_summary.py`
- Statistical and plotting implementation: `src/dissertation_eval/evaluation/{comparison,rhd_audit,rhd_cv,sampling_audit}.py`
- Chronological run notes: `docs/experiment-log.md`

The repository deliberately excludes model weights, complete generated-response records, and hidden-state sidecars. Test configurations may be versioned for reproducibility, but no held-out result is included here or used to fit a direction, feature transform, classifier, threshold, or reporting choice.
