| Strategy | Records | Samples | Completion | Truncation | Parse coverage | Exact match (scored) | Exact match (all) | Mean tokens | Median latency (s) | Peak GPU (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 60 | 60 | 78.3% | 21.7% | 100.0% | 51.1% | 40.0% | 547.7 | 18.6 | 3.67 |
| beam4 | 60 | 60 | 80.0% | 20.0% | 100.0% | 47.9% | 38.3% | 561.0 | 20.2 | 4.13 |
| sampling5 | 300 | 60 | 72.3% | 27.7% | 100.0% | 56.2% | 40.7% | 576.8 | 20.1 | 3.67 |
| sampling5 self-consistency | — | 60 | — | — | 85.0% | 58.8% | 50.0% | 2884.0 | 105.5 | 3.67 |
