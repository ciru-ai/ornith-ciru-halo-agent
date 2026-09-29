# Community decode update qualification

PR: ciru-ai/ornith-ciru-halo-agent#2, head 7220d47a9428ec75dd0b1506168541b8bd9203b4.
Controls: current public Ornith 4.0.1 and Apodex 4.0.0, captured by HF revision.
The production runtime, weights, 262144 context, 44 GiB cache, eight slots,
vision configuration and native tool parser are held constant.

Candidate: three rebuilt native libraries; small GEMV; shared expert fusion;
fast routing; GDN copy. Ornith also gets short drafting graphs/policy and
reduced draft candidate vocabulary. Apodex retains its native MTP4 policy.
Approximate target-head pruning is experimental and disabled by default.

Gates: source review; bit-exact native dense/head rows; complete serving screen;
mirrored A1/C1/C2/A2 single-request performance; package hash/provenance checks.
Native tests cover 21 shape/row cases, including the full 248320 vocabulary.
Serving quality covers executable code, exact arithmetic/JSON, multilingual
output, structured output, logprobs, a real write tool action and return,
image input, long-context retrieval, and concurrent requests.

Reasoning stays enabled. Completion caps are omitted; source inspection of
get_max_tokens confirms remaining context is used when defaults contain no cap.
Every scored request must terminate naturally. Requests and first responses
are retained, including failures. This is a bounded regression screen; it does
not establish broad benchmark quality parity.

Primary performance: streamed completion tokens per decode second, plus whole
request elapsed time and first-token latency. Two mirrored positions, three
finite output workloads per position; warmup excluded. Material threshold 2%.
A split direction or a concrete correctness concern permits one focused
confirmation or correction. A substantial short-answer slowdown requires a
policy correction before promotion. No unrelated runtime upgrade is included.

Services were inactive on Ciru and Sozo at preflight. Test processes own their
process groups; normal model service links, enabled state and profile numbers
remain managed by the existing selector. Save rollback copies on deployment.

## Narrowing and one additional bounded screen

The full Apodex port entered a repeated reasoning cycle during its LRU code
request. At the diagnostic capture it had run for over 409 seconds, emitted
124062 reasoning characters, and produced no final answer. The complete partial
stream and server logs are preserved. This attempt was aborted after the
observed loop, not at a completion token cap, and is excluded from scored quality.

One rescue candidate restores all changed Apodex target arithmetic: shared
expert, BF16 projections, router, and dense multi-row path. It retains the
bit-exact fused dense M1, head row kernels and GDN copy, with native MTP4 intact.
The original successful control responses are reused; the corrected candidate
gets fresh first responses and the reversed-order control remains required.

The contributor reported a short-code slowdown, and excluded short warmups
suggested a possible cost. One bounded Ornith short-response screen uses one
load per variant, two warmups, two externally tested functions and short
arithmetic. An uncontextualized JSON prompt elicited a refusal from the control;
that complete response is retained and excluded from quality grading. No valid
first response is rerolled. The short screen is directional, not a new headline
benchmark campaign. No further routine repetitions are planned.

The short has_close_elements response was identical in content and reasoning
(253 tokens), but block-4 start increased latency from 2.696 to 2.960 seconds.
The final Ornith candidate keeps initial DF15 by default, with the new start
behind ORNITH_C1_START_DEPTH=3. This is a new bounded configuration qualification:
C1/C2/A2 loads with the original A1 control retained, three complete performance
workloads, the complete functional screen on C1, and the affected short function
on all three positions. Original policy results remain immutable in separate
artifacts. No scored first response is replaced with a preferred reroll.

## Apodex state-copy isolation

The reduced GDN/native port failed its generated capacity-two unittest on both
candidate loads, while both controls passed. Independent implementation tests
passed, but do not erase a generated-suite failure. JSON throughput declined
about 4%. This variant is rejected for deployment; its balanced rows remain
separate from the final qualification.

A direct implementation investigation verified nonzero M1 inputs in the
21-case native screen and compared the original normal routed entry at M1, M4
and M16; all were exact. The remaining state-copy change is isolated by restoring
Apodex's released GDN source and manifest. Two independent native-only loads
must pass the original LRU generated suites and independent reference checks.
The first load also runs complete JSON, prose and the ten serving quality
requests at full native reasoning budget. No matched throughput claim is made
for this final subset, and no further routine quality rerolls are planned.
