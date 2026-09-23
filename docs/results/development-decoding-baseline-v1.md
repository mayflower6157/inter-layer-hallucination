# ReTruthQA development decoding baseline v1

This table reports the frozen 60-example development split. All conditions use the
same non-thinking prompt and a 1,024-token generation limit. Sampling uses five fixed
seeds with temperature 0.7 and top-p 0.9. “Accuracy (scored)” excludes truncated or
unparsed generations; “accuracy (all)” uses every generation or development question
as the denominator. Parse coverage is measured among completed generations.

| Model | Strategy | Generations | Completion | Truncation | Parse coverage | Accuracy (scored) | Accuracy (all) | Mean tokens | Median latency (s) | Peak GPU (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-2B | Greedy | 60 | 78.3% | 21.7% | 100.0% | 51.1% | 40.0% | 547.7 | 18.6 | 3.67 |
| Qwen3-2B | Beam-4 | 60 | 80.0% | 20.0% | 100.0% | 47.9% | 38.3% | 561.0 | 20.2 | 4.13 |
| Qwen3-2B | Sampling-5 (individual) | 300 | 72.3% | 27.7% | 100.0% | 56.2% | 40.7% | 576.8 | 20.1 | 3.67 |
| Qwen3-2B | Sampling-5 self-consistency | 60 questions | — | — | 85.0% vote coverage | 58.8% | 50.0% | 2,884.0 total/question | 105.5 total/question | 3.67 |
| Qwen3-4B | Greedy | 60 | 95.0% | 5.0% | 100.0% | 52.6% | 50.0% | 290.5 | 7.5 | 7.73 |
| Qwen3-4B | Beam-4 | 60 | 98.3% | 1.7% | 100.0% | 49.2% | 48.3% | 279.6 | 9.4 | 8.38 |
| Qwen3-4B | Sampling-5 (individual) | 300 | 98.0% | 2.0% | 100.0% | 49.7% | 48.7% | 278.4 | 7.5 | 7.73 |
| Qwen3-4B | Sampling-5 self-consistency | 60 questions | — | — | 100.0% vote coverage | 55.0% | 55.0% | 1,392.2 total/question | 37.7 total/question | 7.73 |

## Interpretation

- Qwen3-4B provides the strongest development result: 55.0% all-question exact match
  with Sampling-5 self-consistency.
- Self-consistency improves over the corresponding individual-sampling all-record
  accuracy for both models: 40.7% to 50.0% for 2B and 48.7% to 55.0% for 4B.
- Qwen3-2B reaches its 1,024-token limit much more often, so scored-only accuracy is
  optimistic relative to all-record accuracy. Its self-consistency prediction coverage
  is 85.0% because questions with no valid completed vote cannot receive a prediction.
- Beam-4 does not improve exact match over greedy for either model on this development
  set, despite higher peak memory and/or latency.
- The 2B latency is not a clean model-size efficiency comparison: Transformers reported
  reference-kernel fallbacks for missing optional `causal_conv1d` and
  `flash-linear-attention` implementations. Accuracy and completion remain valid, but
  the observed 2B timing should be labelled environment-dependent.

Machine-readable evaluation artifacts are stored on the experiment server under
`evaluations/qwen3-2b-development-decoding-v1/` and
`evaluations/qwen3-4b-development-decoding-v1/`. Each directory contains the combined
comparison, per-strategy summaries and records, and Sampling-5 self-consistency records.
