# RHD method specification v1

## Scope and claim boundary

Reasoning Hallucination Detection (RHD) is the dissertation's post-hoc analysis and
detection method. It does not alter token selection. Greedy, beam search, sampling,
DoLa, and DeCo are decoding conditions whose traces may be analysed by RHD. The
mitigation component in the reference work is GRPO-R, a training method; it is outside
the present inference-only scope. Consequently, the RHD intervention point and decoding
update are both **none** in the primary experiment.

The extractor is label-blind and teacher-forced over an already generated response. It
must reproduce the exact prompt token count and consume the recorded generated token
IDs. It never calls `generate`, changes a token, or reads the reference answer.

## Token- and step-level quantities

Let a model contain `L` transformer blocks. For generated token `y_i`, let
`h_i^(l)` be the hidden state after block `l`. With the checkpoint's final normalization
`N` and unembedding head `W_U`, define the logit-lens distribution

```text
p_i^(l) = softmax(W_U N(h_i^(l))).
```

The mature anchor is `p_i^(L)`. For candidate late layers `C`, define

```text
d_i,l = JSD(p_i^(l) || p_i^(L))
R_i   = mean_{l in C} d_i,l
R_t   = mean_{i in I_t} R_i,
```

where `I_t` contains the generated-token positions in reasoning step `t`. Diagnostic v1
stores `d_i,l`, L1 distance, the generated token's log probability and rank at every
candidate layer, final-layer entropy, top-token identity, and top-token agreement with
the mature layer. Full vocabulary logits are not persisted.

### Reference-code compatibility

The primary implementation consulted for this specification is commit
`00c1946382301aa9432609ac605f7876c545e545` of
`Jeryi-Sun/Reasoning_Hallucination`. Its released code computes a second quantity that
is not identical to the standard JSD above. For mixture `m=(p+q)/2`, it evaluates

```text
D_ref(p,q) = 0.5 * mean_v[m_v(log m_v - log q_v)
                         + m_v(log m_v - log p_v)]
R_ref      = 1e5 * mean_{token,layer} D_ref.
```

This follows directly from the argument order and `mean(-1)` reduction in the released
`torch.nn.functional.kl_div` calls. Diagnostic version 1.1 stores both the bounded,
standard `JSD` and the separately named `reference_divergence`/`reference_reasoning_score`.
It does not call the latter JSD, preventing an implementation detail from being confused
with the conventional mathematical definition.

The released detector represents every reasoning path by the ordered feature vector

```text
[AttentionScore, CV(R_early), mean(R), PCC(delta R, delta entropy)].
```

It obtains a domain- and model-specific weighted sum of these four entries. Therefore,
an equivalent symbolic form is:

```text
H_C = alpha_1 Avg(R)
    + alpha_2 CV(R_early)
    + alpha_3 AttnScore(C)
    + alpha_4 PCC(R, PPL).
```

`CV(R_early)` is `std/mean` over the first half of steps. `AttnScore` examines the last
20% of reasoning steps and counts, among their top-5 attended earlier steps, those below
the 20th percentile of `R` or above the released code's fixed value 4. `PCC` is Pearson
correlation between first differences of independently min-max-normalized `R` and
normalized entropy. Diagnostic v1.1 records all four unweighted features, plus the
distribution-derived token diagnostics. It does not form a final weighted detector
until weights and an operating threshold are fitted and frozen on development data.

## Frozen reasoning-step segmentation

For generated Qwen3 traces, a reasoning step is a maximal non-empty paragraph.
Boundaries are two or more newline characters with optional intervening whitespace.
Leading/trailing whitespace is excluded, empty paragraphs are discarded, and a
non-empty response with no boundary is one step. This label-blind
`paragraph_blank_line_v1` rule is fixed before detection evaluation and is applied
identically to every model and decoding strategy. It approximates the pre-segmented
reasoning paths consumed by the released RHD code without introducing semantic or
answer-dependent manual annotation.

## Candidate layers

The released implementation hard-codes mature/candidate layers by checkpoint:

```text
DeepSeek-R1-Distill-Qwen-7B:  mature=28, candidates=[14,16,18,20,22,24,26]
DeepSeek-R1-Distill-Qwen-14B: mature=48, candidates=[32,36,40,42,44,46]
```

It does not specify layers for Qwen3-2B or Qwen3-4B. Consequently, copying either list
would not be a faithful application to our architectures. The initial, explicitly
exploratory diagnostic grid uses normalized depths
`{0.50, 0.625, 0.75, 0.875}`, with the final block at depth `1.0` as the anchor. Mapping
uses `floor(depth * L + 0.5)`, is one-based in block space, removes duplicates, and never
uses the final block as a candidate. This yields layers `{14, 18, 21, 25}` for a
28-block model and `{18, 23, 27, 32}` for a 36-block model. Both absolute indices and
normalized depths are stored.

This grid is provisional for label-blind sanity checks, not claimed as the reference
paper's layer configuration. Candidate grids for Qwen3 must be compared using only the
60-example development split under a documented search budget, then frozen before the
180-example held-out test.

## Threshold and fitting policy

Diagnostic extraction has no threshold. For the later binary detector, non-negative
`alpha` coefficients are fitted using development traces only. Operating thresholds
are selected by source-split grouped cross-validation on development data, maximizing
balanced accuracy; ties choose the higher-recall threshold. The median selected
threshold is frozen for held-out text and transfer evaluation. AUROC and AUPRC remain
threshold-free primary ranking metrics.

The reference paper's numerical overthinking threshold is not transferred directly to
Qwen3 because Reasoning Score scale depends on model, layer grid, normalization, and
aggregation. Any threshold used here must be estimated and frozen under this policy.

## Pseudocode

```text
for each completed generation record:
    rebuild chat-templated prompt
    assert rebuilt_prompt_tokens == recorded_prompt_tokens
    sequence = rebuilt_prompt_ids + recorded_generated_token_ids
    hidden_states = base_transformer(sequence, output_hidden_states=True)

    for each generated token position i in bounded chunks:
        q = softmax(unembed(final_hidden[i]))
        store entropy(q), chosen-token logprob/rank, argmax(q)
        for each candidate layer l:
            p = softmax(unembed(final_norm(hidden[l, i])))
            store standard JSD(p, q), reference-code D_ref(p, q), L1(p, q),
                  chosen-token logprob/rank,
                  argmax(p), and argmax agreement
        store R_i = mean_l JSD(p, q)

    write compressed immutable sidecar
    append record metadata, summaries, sidecar path, and SHA-256
```

## Diagnostic acceptance criteria

Before full development extraction:

1. prompt length and generated-token identity checks pass;
2. all JSD values are finite and lie in `[0, ln(2)]` up to numerical tolerance;
3. token scores are non-constant within at least one trace;
4. candidate layers do not all have identical agreement/divergence summaries;
5. rerunning skips checksum-addressed completed records;
6. generation JSONL checksums remain unchanged.

Passing these criteria establishes extraction correctness and the presence of a
measurable inter-layer signal. It does not establish hallucination discrimination;
that requires frozen step construction and development labels.
