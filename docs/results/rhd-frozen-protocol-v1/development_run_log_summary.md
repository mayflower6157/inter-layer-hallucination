# Development run summary

This is the compact report-facing log summary for the frozen development checkpoint. Detailed chronological notes remain in `docs/experiment-log.md`.

| Model | Decoder | Generated records | Completed/scored | Truncated | RHD CV records |
|---|---:|---:|---:|---:|---:|
| Qwen3-2B | Greedy | 60 | 47 | 13 | 47 |
| Qwen3-2B | Beam-4 | 60 | 48 | 12 | 48 |
| Qwen3-2B | Sampling-5 | 300 | 217 | 83 | 217 |
| Qwen3-4B | Greedy | 60 | 57 | 3 | 57 |
| Qwen3-4B | Beam-4 | 60 | 59 | 1 | 59 |
| Qwen3-4B | Sampling-5 | 300 | 294 | 6 | 294 |

All completed records were parsed in the decoding comparisons. Sampling-5 used five generations for each of 60 development questions. Calibration replay reproduced development scores with maximum absolute error no larger than `2.22e-16`, as recorded in the frozen protocol.

No held-out output, metric, or label is summarized in this file.
