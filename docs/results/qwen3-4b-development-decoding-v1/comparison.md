| Strategy | Records | Samples | Completion | Truncation | Parse coverage | Exact match (scored) | Exact match (all) | Mean tokens | Median latency (s) | Peak GPU (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 60 | 60 | 95.0% | 5.0% | 100.0% | 52.6% | 50.0% | 290.5 | 7.5 | 7.73 |
| beam4 | 60 | 60 | 98.3% | 1.7% | 100.0% | 49.2% | 48.3% | 279.6 | 9.4 | 8.38 |
| sampling5 | 300 | 60 | 98.0% | 2.0% | 100.0% | 49.7% | 48.7% | 278.4 | 7.5 | 7.73 |
| sampling5 self-consistency | — | 60 | — | — | 100.0% | 55.0% | 55.0% | 1392.2 | 37.7 | 7.73 |
