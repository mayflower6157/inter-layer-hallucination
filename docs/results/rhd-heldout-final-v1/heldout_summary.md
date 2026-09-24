# Frozen held-out RHD evaluation

All transforms, weights, and thresholds are loaded from development-only calibrations. No held-out fitting or threshold selection is performed.

| Condition | Scored | Incorrect | Length/step AUROC | RHD AUROC | Combined AUROC | Gain [95% CI] | RHD AUROC [95% CI] | RHD AUPRC [95% CI] | RHD balanced accuracy [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-2B Greedy | 122 | 68 | 0.508 | 0.636 | 0.620 | 0.111 [0.016, 0.208] | 0.636 [0.535, 0.741] | 0.668 [0.545, 0.788] | 0.549 [0.477, 0.623] |
| Qwen3-2B Beam-4 | 126 | 62 | 0.547 | 0.617 | 0.628 | 0.081 [-0.069, 0.229] | 0.617 [0.516, 0.709] | 0.591 [0.479, 0.710] | 0.599 [0.519, 0.670] |
| Qwen3-2B Sampling-5 | 606 | 326 | 0.582 | 0.630 | 0.617 | 0.036 [-0.033, 0.100] | 0.630 [0.556, 0.698] | 0.637 [0.546, 0.728] | 0.605 [0.553, 0.657] |
| Qwen3-4B Greedy | 169 | 92 | 0.556 | 0.642 | 0.653 | 0.097 [0.027, 0.174] | 0.642 [0.557, 0.721] | 0.704 [0.603, 0.794] | 0.612 [0.538, 0.683] |
| Qwen3-4B Beam-4 | 169 | 92 | 0.530 | 0.574 | 0.575 | 0.046 [-0.047, 0.139] | 0.574 [0.487, 0.657] | 0.655 [0.563, 0.750] | 0.533 [0.456, 0.607] |
| Qwen3-4B Sampling-5 | 849 | 471 | 0.560 | 0.609 | 0.603 | 0.043 [-0.018, 0.104] | 0.609 [0.550, 0.668] | 0.666 [0.590, 0.731] | 0.592 [0.544, 0.643] |
