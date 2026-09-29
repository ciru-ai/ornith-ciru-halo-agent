# Ciru v4.1 bounded community regression screen

Normal 262144-token context, eight slots, 44 GiB cache, pinned ROCm runtime, native reasoning and full remaining output budget. Model weights, tool component and engine wheels retained. A1/C1/C2/A2 are independent loads. One complete prose, LRU implementation plus generated unittest suite, and JSON request per position; warmup excluded. Throughput includes reasoning tokens. Geometric means of two loads per variant:

| Workload | Control tokens/s | Update tokens/s | Change | Mirrors C1/A1 and C2/A2 |
|---|---:|---:|---:|---:|
| prose | 77.25 | 114.65 | +48.4% | +59.6% / +38.0% |
| code | 105.50 | 117.68 | +11.5% | +16.7% / +6.6% |
| json | 191.32 | 215.49 | +12.6% | +14.3% / +11.0% |

Native unrestricted reasoning causes response-length variation. Prose throughput varies across loads; these two samples do not establish a consistent prose speedup. Ornith target outputs need not be bit-identical because the fused shared activation, router and projection arithmetic differ slightly. No additional prose repetition was added because it would not change the retention decision supported by the code/JSON screen and functional checks.

The two serving quality positions passed exact arithmetic/JSON, multilingual output, structured output, logprobs, a real file write tool call and subsequent tool-result return, image color recognition, over-25K prompt retrieval and two concurrent requests. Every response terminated naturally. This small screen does not establish broad benchmark quality parity; historical full benchmark scores are unchanged.

Approximate target-head pruning is disabled by default. Native tests passed 21 dense/head shape-row cases bit-identically, including the full 248320 head. Additional GPU checks covered BF16 small projections, router expert selection, exact GDN state copy, and a shared expert with all routed slots disabled. The latter required a correction during integration.

Sozo's existing Python runtime contained a separate source edit; Apodex used an isolated copy of pinned Ciru runtime sources over the existing matching runtime binaries. The initial startup rejection and a control JSON grading correction (accepting an existing fenced JSON response) were retained; completed first responses were reused without rerolling.

Whole-request latency (lower is better), including unrestricted reasoning:

| Workload | Control seconds | Update seconds | Change | Completion tokens A1 / C1 / C2 / A2 |
|---|---:|---:|---:|---|
| prose | 49.92 | 56.54 | +13.2% | 4495 / 6183 / 6775 / 3297 |
| code | 16.92 | 22.68 | +34.0% | 1757 / 2637 / 2664 / 1784 |
| json | 11.44 | 10.20 | -10.8% | 2172 / 2167 / 2167 / 2169 |

Throughput is not time to a finished answer. In particular, longer Ornith code reasoning increases total request time despite faster decoding. These results support decode throughput gains for the measured workloads, not a claim that every user request finishes sooner. All raw timing and usage rows are included in the JSON report.

Independent LRU checks passed at all four positions, including capacity validation, update/eviction/recency behavior, capacity one, and 145 seeded reference comparisons per implementation. Generated self-test results are recorded separately.

All four generated unittest suites also passed.

A short-code check produced an identical 253-token answer but slowed by about 0.26 seconds with the block-4 initial policy. The qualified default retains DF15 initially; block-4 start is experimental. Final default-policy checks include that externally tested short function. Final candidate loads took 2.206 and 2.212 seconds versus the final control at 2.952 seconds; response lengths differ (235 vs 253 completion tokens).
