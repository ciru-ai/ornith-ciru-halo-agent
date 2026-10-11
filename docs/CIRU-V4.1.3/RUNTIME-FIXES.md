# Ciru v4.1.3 runtime fixes

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


## Earlier runtime history

## Ciru v4.1.2

Adds the verified synchronization barrier in the first DFlash convolution/residual/RMSNorm kernel used by `ornithtest`. Both packages correct full-context output reservation. [Patch notes](PATCH-NOTES-4.1.2.md) and [upgrade instructions](INSTALL.md#upgrading-to-412).

# Ciru v4.1.1 — September 30, 2026

The retained community subset accelerates dense M1 projections, compact head rows and the shared expert while preserving the released numerical storage boundaries. The shared activation uses a 384 KiB reference table derived from the released compiled expression; all 65,280 finite BF16 gate patterns match by bits. The PR's router and small BF16 projection substitutions are excluded after actual GPU replay reproduced arithmetic forks.

Q/K normalization now selects the qualified reduction bodies structurally. Compiler-assigned kernel numbers previously allowed the same input graph to silently switch reduction order. Startup refuses serving when the qualified substitution did not execute. Cache identity includes all executable plugin, native, launcher and reference-data files.

Weights, BF16 vision tensors, trained draft/MTP tensors, pinned engine wheels and model sampling policy are unchanged. Complete scored requests use the full 262,144-token context, native reasoning and no completion cap. These are bounded runtime comparisons, with existing broad benchmark scores retaining their original build labels.

Ornith Q8/Q16 recurrence preserves the original packed single-token arithmetic, and attention uses per-query partitions.

[Qualified results and limitations](evaluation/CIRU-V4.1.1-QUALIFICATION.md).

Community decode optimizations: **Daniele Zannotti**, [PR #2](https://github.com/ciru-ai/ornith-ciru-halo-agent/pull/2) (`7220d47a9428ec75dd0b1506168541b8bd9203b4`). Ciru supplied the numerical repairs, runtime consistency fixes and qualification of the retained subset. Upstream authorship is preserved in Git history.

# Ciru runtime patch notes

## Ciru v4.0.1 — September 25, 2026

- Set `ORNITH_OPTIMIZED_CACHE` from the selected cache directory before the
  optimized worker starts. A fresh v4.0.0 launch could fail without this value.
- Add `--max-images-per-prompt N` to the launcher. The default remains one;
  vLLM counts images across the full request, including chat history. The
  operator can choose a higher limit for image-heavy agent sessions.
- Keep the v4.0.0 weights, native libraries, pinned vLLM/AITER runtime,
  sampler, tool handling and default serving profile unchanged.

An isolated Strix Halo launch reached `/health`, and a two-image OCR request
returned HTTP 200 with both words. This is a narrow patch check, not a new
multi-image quality, concurrency or performance benchmark. Use the
[4.0.1 update instructions](INSTALL.md#updating-from-400-to-401).

## Ciru v4 (4.0.0) — September 17, 2026

Continues the Ciru v3 model release line. The earlier 1.0.x identifiers below
describe runtime patches; R03/R04 component identity 1.0.3 remains provenance,
while this complete public release is 4.0.0.

- Integrate the retained output32 expert tile and all six qualified GDN
  traversal changes, including the older Apodex campaign work.
- Use Q4480 prefill chunks with the required wide-capacity dense/expert
  libraries and compiler adapters, plus the qualified prefix-cache repair.
- Include the persistent-attention correctness repair, exact Q/K fusion,
  fused attention prefill, draft convolution/residual/RMSNorm repair and
  greedy draft-path fusion. Preserve the default adaptive DFlash policy.
- Integrate strict-grammar scheduling, the native XGrammar finite-repetition
  and shared-memory accounting changes, and continuous usage reporting during
  buffered tool streams. Preserve natural finish reasons and explicit errors
  for malformed output.
- Preserve all target, draft and BF16 vision weights, the native chat template,
  sampler, context/cache limits and previous correctness overlays.
- Make the public launcher select the complete optimized worker and use
  relative installation/model defaults when no paths.env override exists.

Update **both** native libraries and the complete plugin/launchers using
[INSTALL.md](INSTALL.md). The pinned vLLM/AITER wheels are retained.

Final native performance: cold prefill +26.23% at64K and +30.33% near256K;
cached C1 decode +19.74% at63K and +26.40% near256K. Short C2/C4 aggregate
throughput regressed, and near256K C8 first-token latency increased. Numerical
fidelity results and the complete speed comparison are reported on the model card.
All individual historical gains overlap and must not be added.

The final native performance block completed97 measured requests with90/90
coding health checks. Native tool integration passed18 exact live calls and
preserved identical333-token streams across usage modes. See the [Ciru v4 results](README.md#ciru-v4-results).


## Runtime 1.0.2 — September 15, 2026

- Enable native vLLM/XGrammar schema constraints for automatic tool calls when
  the function's `strict` setting is omitted. Required arguments are enforced
  during generation; prompt-visible messages and tool definitions are unchanged.
- Preserve explicit `strict: false`, explicit output formats, and the existing
  required/named tool-choice behavior.
- Align final schema validation with the function's strict setting while
  retaining framing and lossless streaming checks.
- Remove Apodex's inherited 32K output ceiling, matching the retained local
  profile. Request limits and the context window still apply.
- Enable the existing upstream BF16 image encoder/projector by default, with
  one image per request and an explicit `--text-only` option.
- Retain runtime 1.0.1's inference correctness fixes, IU4 tensors, native
  libraries, runtime wheels, and speculation settings.

Validation: **26/26 native schema checks** and **4/4 live serving checks**
passed for this package. The live checks cover plain text, a 3,167-byte tool
payload, image OCR/shapes, and image-to-tool output at the normal C8/262144
serving settings. See [validation receipt](TOOL-SCHEMA-PATCH-1.0.2.json).


**Ciru runtime 1.0.1 — September 14, 2026.** This accuracy patch promotes
BF16 causal-convolution operands to FP32 before multiplication, preventing an
extra rounding step, and restores the accepted GDN recurrent state and
convolution history when a speculative batch returns to ordinary decoding.
Recovery metadata uses persistent buffers so the correction also works with
captured FULL graphs. IU4 weights and native libraries are unchanged.

On Ornith, the matched local speed screen found approximately 7% lower single-request
throughput and roughly unchanged eight-request throughput. This tradeoff is
accepted for this correctness release.

The convolution correction follows [vLLM #52905](https://github.com/vllm-project/vllm/pull/52905);
accepted-state recovery follows [vLLM #55504](https://github.com/vllm-project/vllm/pull/55504),
with Ciru's graph-buffer extension. See [runtime patch validation](ACCURACY-PATCH-1.0.1.json)
for the bounded correctness and speed checks. This patch does not establish that
all long-generation quality issues are resolved.

# Runtime fixes — 13 September 2026

This update prevents a reproduced cache-corruption crash and makes malformed tool output fail explicitly. It updates the serving plugin; the released weights, native libraries, runtime wheels, sampler, context pool and adaptive DFlash2 policy are unchanged.

## Historical September 13 update instructions

Stop your server before replacing its plugin files. Run this from the directory containing `bundle/` and `runtime/`, then restart your usual launcher:

```bash
uvx --from huggingface_hub hf download \
  jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo \
  --include 'bundle/plugin-site/*' 'bundle/serve*.sh' README.md INSTALL.md VISION.md RELEASE.json RUNTIME-FIXES.md TOOL-SCHEMA-PATCH-1.0.2.json \
  --local-dir .
bash bundle/serve.sh --host 127.0.0.1 --port 8000
```

This downloads the small plugin/document update without downloading the weights again. No runtime reinstall or kernel rebuild is required. Deployments that copied the plugin elsewhere must update that copy too. A running process needs a restart to load the changes. Existing vision installations can restart `bundle/serve-vision.sh` instead.

## What changed

**Graph dispatch:** a prompt tail could have the same token width as a captured speculative decode batch while carrying different recurrent-state metadata. Replaying that graph could write into older cache pages, corrupt values and eventually crash the worker. The guard checks the actual request phase and scheduled drafts before allowing that replay. Ordinary single-request and concurrent decode graphs remain enabled.

**Tool parsing:** the bundled Qwen XML adapters validate complete framing, duplicate parameters, arguments against the supplied schema, and required/named tool choices. They reject an ambiguous tool opening inside reasoning followed by a later explicit reasoning terminator. Invalid full responses return HTTP 400 with `ToolParserContractError`; invalid streams terminate with an error instead of a successful tool completion. The native Qwen XML value codec is retained: it removes one leading and one trailing parameter newline as template markup. This is not a byte-for-byte boundary-whitespace guarantee; a model that emits only one final newline for a newline-bearing value can lose that newline. Exact whitespace remains a separate serialization limitation, not a graph-cache repair claim.

Streaming tool deltas are provisional. Clients must wait for successful stream finalization before executing a call and must discard a call when the stream reports an error. This requirement applies to any harness; the server cannot undo a tool already executed by a client.

The graph guard is installed before the adaptive graph-width wrapper. The parser is installed before API startup when tools are enabled. The September 11 explicit C1 draft-depth controls are preserved.

## Evidence and limits

- A causal GPU reproduction of the shared graph fault changed **1,105,793 of 1,146,880 BF16 values** and produced **1,919 non-finite values** without the guard. With the guard, the same check changed **zero values**, produced **zero non-finite values**, and completed.
- The Ornith Hermes profile on Strix Halo completed **21/21 requests**, including ten dangerous prompt-tail transitions, one exact generated write call and **10/10 HE0–9 first-sample canonical checks**. Ordinary Q16/C1 and Q8/C8 FULL decode graphs were observed. This is a deployment screen, not a full HumanEval score.
- **430 CPU guard/dynamic-wrapper checks** passed. After the GPU screen, the final reasoning-boundary parser refinement passed **6/6 ASGI stream/full replay checks**, with no tool dispatch for the rejected capture. These parser checks replayed saved output; they did not generate new GPU responses.
- GPU coverage is the Ornith Hermes text profile. The other release and vision wrappers received source/settings compatibility checks, without another image evaluation. The previously published throughput and broad quality tables describe their original benchmark runs; this patch has no matched new throughput sweep.

These changes fix demonstrated runtime and parser failures. They do not establish BF16 equivalence, guarantee the correctness of generated code, or eliminate every possible model, client or service failure.

Source: [Ciru Ornith runtime](https://github.com/ciru-ai/ornith-ciru-halo-agent). Exact changed-file hashes are recorded in `RELEASE.json` on Hugging Face and `SOURCE-PROVENANCE.json` in the source repository.
