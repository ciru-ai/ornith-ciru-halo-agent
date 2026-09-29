# Apodex v4.1 bounded native-only qualification

The final default retains fused dense M1 and compact head row kernels. Its original shared expert, BF16 projections, router, dense multi-row path, GDN state-copy implementation, native MTP4 and model-specific cache stride are retained. Weights and pinned engine wheels are unchanged.

## Changes held back

The full community port entered a repeating reasoning cycle in the LRU request: over 409 seconds and 124062 reasoning characters without final code. The complete partial stream was preserved and the attempt was stopped after the observed loop, without a token cap.

The first reduced variant retained GDN copy and native kernels. Both candidate loads generated an incorrect capacity-two unittest, while both control loads generated passing suites. The implementations passed independent tests, but the generated suites still represent a repeatable quality regression signal. JSON decode throughput also fell about 4%. This variant is held back. Its original first responses and balanced A1/C1/C2/A2 rows are preserved in APODEX-GDN-SUBSET-SCREEN.json.

Restoring the released GDN state-copy implementation isolates the remaining native changes. This does not prove the root cause of the earlier failures, and those observations are not discarded or described as harmless variation.

## Final qualification

Two independent native-only candidate loads each completed the LRU request, including their generated unittest suites and independent reference checks (145 seeded gets plus capacity, eviction, update and recency edge cases). The first load also completed exact JSON/arithmetic, multilingual output, strict structured output, streamed logprobs, an actual file-write tool call and tool-result return, image color recognition, over-25K prompt retrieval and two concurrent requests. Every scored response terminated naturally.

The retained native operations passed the 21-case dense/head shape-row screen with explicit nonzero M1 inputs and full 248320 vocabulary. The normal routed entry also matched released output exactly for M1/M4/M16. Approximate target-head pruning is disabled.

This final check ran on Ciru with the normal 262144 context, eight slots, 44 GiB cache, pinned runtime and native reasoning. Completion caps were omitted, and endpoint source inspection confirmed remaining-context budgeting. The earlier rejected variants ran on Sozo with an isolated pinned Ciru Python source overlay over matching runtime binaries.

No matched end-to-end speedup is claimed for this final Apodex subset. The original balanced speed comparison exercised a different, rejected variant. Final candidate measurements are shown for transparency:

| Workload | Candidate decode tokens/s | Whole request seconds | Completion tokens |
|---|---:|---:|---:|
| prose | 75.99 | 96.80 | 7301 |
| code | 70.58 | 62.64 | 4402 |
| json | 84.15 | 45.10 | 3784 |

This bounded screen does not establish broad benchmark quality parity. Existing published benchmark scores describe their original builds and runs.
