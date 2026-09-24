# RHD development audit

Records: 60 (57 scored). This report uses development data only.

## Step segmentation

Method: `paragraph_blank_line_v1`; one-or-two-step fraction: 0.067.

| Quantity | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Steps per trace | 8.717 | 6.000 | 1 | 31 |
| Tokens per step | 33.323 | 26.000 | 2 | 287 |

## Overall feature discrimination

| Feature | AUROC | Corr. tokens | Corr. steps |
|---|---:|---:|---:|
| attention_score | 0.609 | -0.224 | -0.186 |
| cv_score | 0.330 | 0.482 | 0.581 |
| reasoning_score_mean | 0.342 | -0.054 | -0.023 |
| pcc | 0.626 | -0.321 | -0.365 |

## Stratified AUROC

### domain

| Group | N | Correct | Incorrect | Attention | CV | Mean-R | PCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MATH | 17 | 13 | 4 | 0.817 | 0.615 | 0.404 | 0.385 |
| MultiHopQA | 20 | 10 | 10 | 0.510 | 0.280 | 0.270 | 0.610 |
| Science | 20 | 7 | 13 | 0.352 | 0.363 | 0.462 | 0.769 |

### length_group

| Group | N | Correct | Incorrect | Attention | CV | Mean-R | PCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| long | 18 | 11 | 7 | 0.851 | 0.377 | 0.247 | 0.597 |
| medium | 19 | 9 | 10 | 0.628 | 0.256 | 0.356 | 0.600 |
| short | 20 | 10 | 10 | 0.385 | 0.420 | 0.380 | 0.570 |

## Normalized token-position trajectory

Values are the mean reference-compatible Reasoning Score in ten normalized token-position bins.

| Outcome | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| incorrect | 1.939 | 2.131 | 2.492 | 2.337 | 2.546 | 2.276 | 2.154 | 2.311 | 2.259 | 2.595 |
| correct | 2.247 | 2.264 | 2.529 | 2.423 | 2.685 | 2.685 | 2.502 | 2.480 | 2.378 | 2.564 |
| unscored | 2.191 | 2.309 | 2.342 | 2.222 | 1.961 | 1.962 | 2.083 | 1.932 | 1.910 | 1.903 |
