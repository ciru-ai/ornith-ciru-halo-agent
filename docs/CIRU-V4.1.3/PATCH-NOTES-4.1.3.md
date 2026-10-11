# Ciru v4.1.3

**Ciru v4.1.3 — October 11, 2026.** IU4 numerical repairs, exact native sampling, discarded-row RNG restoration, and reversible K0/K3/K7 AUTO at long context. Concurrent requests use K0 with fresh-prefix read isolation. [Patch notes](PATCH-NOTES-4.1.3.md) · [Upgrade](INSTALL.md#upgrading-to-413) · [Validation](evaluation/CIRU-V4.1.3-VALIDATION.json).

| Fixture | Live TG tok/s | 4.1.3 TG tok/s | Change | Live / 4.1.3 complete wall | Live / 4.1.3 draft acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Natural EOS,17 tokens 8k | 123.51 | 76.08 | -38.41% | 3.970s / 4.096s | 53.33% / 80.00% |
| Natural EOS,17 tokens 64k | 88.39 | 64.61 | -26.90% | 42.312s / 42.806s | 53.33% / 80.00% |
| Sampled,512 tokens 8k | 49.08 | 48.38 | -1.43% | 14.370s / 14.561s | 9.09% / 34.84% |
| Sampled,512 tokens 64k | 17.26 | 36.30 | +110.30% | 70.845s / 56.104s | 6.39% / 27.63% |

Cold C8 completed throughput: **23.38 → 20.71 tok/s** (-11.42%). Mean complete cohort wall: 43.792 → 49.437s. Mean first-token latency: 23.925 → 24.558s. Natural draft acceptance: 18.70% (1166/6236) → 66.67% (4/6).

Timing uncertainty was flagged: site-64k: mirror gap exceeds3percentagepoints; ultra-64k: mirror gap exceeds3percentagepoints. Both timing mirrors are retained in the linked comparison.

Four fresh engines on Sozo's Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), in live/candidate/candidate/live order, with the same frozen request contracts. The live comparator is public Ciru 4.1.2. Site fixtures use greedy natural-EOS 17-token serving diagnostics; Ultra fixtures use 512-token sampled serving diagnostics at 8k and 64k. C8 uses eight cold 8k requests with 128 tokens each. These are bounded serving measurements, not unrestricted agent-quality scores. C8 wall and TTFT start at the native resume request; paused HTTP admission wait is excluded and separately retained. TG measures decode, PP measures computed prompt ingestion. Draft acceptance is accepted/proposed tokens and depends on the selected depth; it is separate from correctness. Actual concurrent candidate steps use K0, with a possible sole-request AUTO tail.

The refreshed 1,980-position fixed-prefix teacher panel has mean KL 0.186038106 versus live 0.193916819, a 4.063% reduction, and 1,695 versus 1,666 top-1 matches. This is an aggregate fidelity improvement; individual turns still regress, including TC41 turn 2 top-1 110/115 versus 112/115. It does not establish that every response is better.

The actual assembled production namespace passes exact target p/q checks at proposed K3/K7 positions in both 8k and 64k tests, a 256-position paired cold concurrent K0 replay, and the real pending-draft C1-to-concurrent-K0-to-C1 lifecycle. Cold concurrent raw logits, full sparse probabilities and native token IDs match the paired reference. The 100% acceptance in the coupled numerical oracle is diagnostic and must not be reported as natural draft acceptance.

The failed cached/shared-checkpoint gate remains held. Fresh requests admitted while multiple requests are unfinished bypass prefix reads and retain that isolation for fresh recomputation. A request admitted alone earlier retains its ongoing computed state when another arrives. Qualification covers controlled cold admission; every arrival pattern and cached-prefix numerical equivalence are not established. This workaround does not repair or qualify the underlying shared-checkpoint ownership path. Concurrent K7 and active K15 remain unqualified and disabled. The allocation envelope remains 15. Full unrestricted agent quality has not been rescored. Weights, native binaries and pinned engine wheels are unchanged.


### Controlling speculative draft depth

Default single-request AUTO starts at K3, can use K0/K3/K7 at every context length, and caps active drafting at K7. It uses ordinary eight-observation cost windows. The first steady decode observation latches two startup observations at computed context up to 32,768 and one above it; those observations run normally but are excluded from the startup cost estimate. After 63 generated tokens of observed K0 progress, AUTO can probe K3 again. A successful recovery can probe K7 after two fully accepted blocks when the measured cost improves by at least 10%, subject to retry backoff. Startup does not use the earlier faster upward-probe experiment.

```bash
# Keep adaptive K0/K3/K7 (default):
bash bundle/serve.sh
# Fixed single-request depths:
bash bundle/serve.sh --draft-tokens 3
bash bundle/serve.sh --draft-tokens 7
bash bundle/serve.sh --draft-tokens 0
```

Explicit `--draft-tokens` takes precedence over `ORNITH_C1_POLICY`; otherwise the environment variable is respected and defaults to `auto`. Supported explicit depths are 0, 3 and 7; 15 is rejected. These controls also work with `bundle/serve-vision.sh`. More than one unfinished request, including queued requests, forces actual current and future drafts to K0. Under default AUTO, the final sole-request tail restarts at K3 and adapts; explicit fixed depths remain fixed for C1. K0 retains the drafter and allocation capacity. Startup configuration can therefore report allocation 15 while active candidate depths are only 0/3/7. The guard requires synchronous V1, one concurrent scheduler batch and no KV connector.

