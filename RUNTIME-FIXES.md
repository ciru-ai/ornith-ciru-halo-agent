# Runtime fixes — 13 September 2026

This update prevents a reproduced cache-corruption crash and makes malformed tool output fail explicitly. It updates the serving plugin; the released weights, native libraries, runtime wheels, sampler, context pool and adaptive DFlash2 policy are unchanged.

## Update an existing installation

Stop your server before replacing its plugin files. Run this from the directory containing `bundle/` and `runtime/`, then restart your usual launcher:

```bash
uvx --from huggingface_hub hf download \
  jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo \
  --include 'bundle/plugin-site/*' README.md RELEASE.json RUNTIME-FIXES.md \
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
- **430 CPU guard/dynamic-wrapper checks** passed. After the GPU screen, the final reasoning-boundary parser refinement passed **6/6 ASGI stream/full replay checks** and **3/3 installed Pi client replays**, with no tool dispatch for the rejected capture. These parser checks replayed saved output; they did not generate new GPU responses.
- GPU coverage is the Ornith Hermes text profile. The other release and vision wrappers received source/settings compatibility checks, without another image evaluation. The previously published throughput and broad quality tables describe their original benchmark runs; this patch has no matched new throughput sweep.

These changes fix demonstrated runtime and parser failures. They do not establish BF16 equivalence, guarantee the correctness of generated code, or eliminate every possible model, client or service failure.

Source: [Ciru Ornith runtime](https://github.com/ciru-ai/ornith-ciru-halo-agent). Exact changed-file hashes are recorded in `RELEASE.json` on Hugging Face and `SOURCE-PROVENANCE.json` in the source repository.
