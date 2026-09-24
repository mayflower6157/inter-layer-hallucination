# Frozen held-out RHD evaluation

All transforms, weights, and thresholds are loaded from development-only calibrations. No held-out fitting or threshold selection is performed.

| Condition | Scored | Incorrect | Length/step AUROC | RHD AUROC | Combined AUROC | Gain | RHD AUPRC | RHD balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-2B Greedy | 122 | 68 | 0.508 | 0.636 | 0.620 | 0.111 | 0.668 | 0.549 |
| Qwen3-2B Beam-4 | 126 | 62 | 0.547 | 0.617 | 0.628 | 0.081 | 0.591 | 0.599 |
| Qwen3-2B Sampling-5 | 606 | 326 | 0.582 | 0.630 | 0.617 | 0.036 | 0.637 | 0.605 |
| Qwen3-4B Greedy | 169 | 92 | 0.556 | 0.642 | 0.653 | 0.097 | 0.704 | 0.612 |
| Qwen3-4B Beam-4 | 169 | 92 | 0.530 | 0.574 | 0.575 | 0.046 | 0.655 | 0.533 |
| Qwen3-4B Sampling-5 | 849 | 471 | 0.560 | 0.609 | 0.603 | 0.043 | 0.666 | 0.592 |
