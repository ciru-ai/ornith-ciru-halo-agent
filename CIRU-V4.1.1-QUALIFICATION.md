# Ciru v4.1.1 qualification: Ornith

The retained community subset accelerates dense M1 projections, compact head rows and the shared expert while preserving the released numerical storage boundaries. The shared activation uses a 384 KiB reference table derived from the released compiled expression; all 65,280 finite BF16 gate patterns match by bits. The PR's router and small BF16 projection substitutions are excluded after actual GPU replay reproduced arithmetic forks.

Q/K normalization now selects the qualified reduction bodies structurally. Compiler-assigned kernel numbers previously allowed the same input graph to silently switch reduction order. Startup refuses serving when the qualified substitution did not execute. Cache identity includes all executable plugin, native, launcher and reference-data files.

Weights, BF16 vision tensors, trained draft/MTP tensors, pinned engine wheels and model sampling policy are unchanged. Complete scored requests use the full 262,144-token context, native reasoning and no completion cap. These are bounded runtime comparisons, with existing broad benchmark scores retaining their original build labels.

Ornith Q8/Q16 recurrence preserves the original packed single-token arithmetic, and attention uses per-query partitions.

## Complete serving comparisons

| Pool | Workload | Pooled decode gain | Two temporal mirrors | Classification |
| --- | --- | ---: | --- | --- |
| 44 GiB | Complete code | +4.07% | +4.37% / +3.78% | Material win |
| 16 GiB | Complete code | +5.89% | +5.45% / +6.34% | Material win |
| 16 GiB | Complete JSON | +4.76% | +4.98% / +4.53% | Material win |

Prose improves in both mirrors but its magnitude exceeds the preset uncertainty threshold. These compare the retained PR subset around a common original-reference recurrence/attention consistency repair; they do not estimate that repair's speed contribution. The two pool settings are separate experiments. Final structural-selection startup and four complete streams also match the qualified 16 GiB build exactly.

## Reproduced causes and disposition

| Component | Reproduced cause | Disposition |
| --- | --- | --- |
| Shared expert | Fusion removed BF16 storage boundaries; the substituted exponential also lands on the other side of a BF16 midpoint. | Restore boundaries and exact released finite-BF16 activation/sigmoid values. |
| State copy | Negative pointer offsets and an invalid scalar/broadcast fast path. | Wrap valid negative indices; constrain the fast path to qualified one-dimensional indexing. |
| Router | Different FP32 reduction order changes routing weights on identical logits. | Retain the released implementation. |
| Small BF16 projections | Different GEMM reduction changes a BF16 gate input on identical inputs/weights. | Retain released target and draft projections. |
| Ornith speculative recurrence | Pre-existing packed and verification paths used different precision and reduction layouts. | Preserve original packed arithmetic through Q8/Q16 and accepted-state reconstruction. |
| Ornith attention | Pre-existing folded verification used one block partition width for queries with different valid context lengths. | Restore each query's own partition width. |
| Q/K selection | Pre-existing selector depends on generated symbol numbers; recording and fresh baseline compilation can miss it. | Match exact reduction bodies and dependencies, then require successful execution during startup. |

The shared-expression intervention changed only the exponential. For gate −7.0625 and up −9.875, the released product is 0.0596923790872097 and the alternative is 0.0596923865377903, straddling BF16 midpoint 0.0596923828125. Restoring the activation reproduces the entire released captured shared component.

For Q/K normalization, swapping only the exact GPU-replayed sums reproduces all 389,120 query values in each observed path. The differing value is explained by sums 258.10577392578125 versus 258.1057434082031 and the BF16 query midpoint −1.93359375. Structural selection restores the complete original Apodex reference streams.

## Quality scope and remaining limitations

All 14 matched quality cases include complete code/JSON/prose, arithmetic, multilingual output, strict structured output, numeric log probabilities, an actual file write and tool-result continuation, vision, a prompt over 36K tokens, and two requests queued together under a controlled pause/resume. Core token IDs, reasoning and answers match across four independent loads, with fresh compilation and cached restart. Generated code passes its own tests and 145 independent seeded gets plus capacity, eviction, update and recency checks.

Ornith original-reference comparisons contain zero differences in the compared fields and logits. Its recurrence replay checks all 16 captured rows across 30 layers against successive original packed calls, full FP32 states and accepted-state reconstruction. The normalization scheduler produces byte-identical output for 30 already-qualified real graphs and repairs 20 previously missed target graphs after dependency checks.

Released ROCm GEMM still changes rounding with batch shape. A reproduced baseline request uses 54 tokens alone and 59 in a batch; corresponding candidate/control streams match when their batches match. This release does not claim universal batch invariance, arbitrary graph coverage or perfect model correctness. Nonfinite BF16 inputs are excluded from the exhaustive finite-input claim. Cancelled/truncated runs, instrumented timings, invalid recordings and interrupted startup attempts are excluded from qualified results.

## Credit

Community decode optimizations: **Daniele Zannotti**, [PR #2](https://github.com/ciru-ai/ornith-ciru-halo-agent/pull/2) (`7220d47a9428ec75dd0b1506168541b8bd9203b4`). Ciru supplied the numerical repairs, runtime consistency fixes and qualification of the retained subset. Upstream authorship is preserved in Git history.
