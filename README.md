---
license: mit
library_name: vllm
pipeline_tag: image-text-to-text
base_model:
  - ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
model_name: Ornith1.5 Ciru Halo Agent (vllm strix halo)
tags:
  - ornith
  - ciru
  - amd
  - strix-halo
  - gfx1151
  - rocm
  - vllm
  - agentic
  - tool-use
  - quantized
  - speculative-decoding
  - dflash2
  - vision
  - image-text-to-text
---

**DFlash physical-slot repair (4.1.6):** Multimodal draft-cache slots now use physical sequence positions while rotary positions stay unchanged. Native address/reentry and compiled boundary checks, matched uncapped text/one-image 8K/64K replies, a complete image/tool/artifact continuation, retained concurrency/arrival protections and vision/thinking ON plus text/thinking OFF checks passed on Sozo. Weights, numerical kernels, K0/K3/K7 policy and prefill cadence 16/threshold 8192/chunk 1120 are retained. K15 remains parked and the cached-target hold remains. This is a scoped runtime repair, not a new full-agent score or KLD qualification. See [validation](evaluation/CIRU-V4.1.6-VALIDATION.json).


**Long-prefill service update (4.1.5):** Bounded native scheduling gives existing replies decode service while long prompts arrive. In the matched 8k/128k arrival pilot, the existing reply delivered 12.29 tokens/s versus 0.52, and its worst delivery pause during the new prompt's TTFT fell to 1.54s from 5.63s. The new prompt's TTFT increased to 154.7s from 114.5s (+35.1%) because decode receives service during prefill. Sozo checks passed three fresh 128k arrivals during an established 8k reply, with at least 8 incumbent tokens/s over each complete newcomer TTFT and no delivery pause above 2 seconds within that phase. Uncapped short/8k/64k, cold C8, vision/thinking ON and text/thinking OFF checks also passed. Weights, numerical kernels, K0/cache protections and the sole-prefill path are retained. A fresh mixed K0 diagnostic passed paired logits, full probabilities, emitted IDs and discarded-row RNG checks. This is a decode-service/arrival-latency tradeoff; it is not a general PP speedup. Previous numerical and agent-quality results remain historical; K15 stays parked. See [validation](evaluation/CIRU-V4.1.5-VALIDATION.json).


**Concurrency update (4.1.4):** Fixed concurrent K0 decode falling outside the target Q1 graphs and partitioned attention path. Sozo checks passed exact batched-versus-single attention parity, GPU graph replay, complete uncapped mixed 87k/8k C2 and cold 8k C8 replies, and normal vision/thinking ON and text/thinking OFF checks. Native overlapping generation measured 52.50 tokens/s for C2 and 151.85 tokens/s for C8; these are current-run rates, not a matched full-model speedup comparison. Previous quality results remain historical; K15 stays parked. See [validation](evaluation/CIRU-V4.1.4-VALIDATION.json) and [installation details](INSTALL.md).


**Cache-mode update (4.1.3):** Vision ON and text-only OFF now use separate compiled-cache namespaces selected by the upstream launcher. This prevents an ON compiled graph from being reused for an OFF startup. The numerical implementation and previous scoped results are retained; fresh mode-switch smoke checks are recorded in the validation report. See [installation and cache details](INSTALL.md).


**Ciru v4.1.3 patch notes — October 11, 2026.** Current AUTO starts at K3 and adapts among K0/K3/K7 at every context length. Multiple unfinished requests use guarded K0; fresh concurrent admissions bypass prefix reads. The complete 262,144-token context, 44 GiB KV allocation, eight-request capacity, weights and native binaries remain unchanged.

Fresh matched Sozo measurements versus 4.1.2: sampled 512-token TG changes -1.43% at 8k and +110.30% at 64k; cold C8 completed throughput changes -11.42%. Short natural-EOS TG changes -38.41%/-26.90% (8k/64k). The 1,980-position teacher panel lowers mean KL 4.063% and adds 29 top-1 matches, with individual-turn regressions; unrestricted agent quality remains unscored.

K15 is parked as allocation capacity. Fresh-prefix isolation does not repair shared-checkpoint ownership. Controls below describe historical 4.1.2 and are superseded by this note; retained results keep their original versions. See [detailed results](RESULTS.md), [patch notes](PATCH-NOTES-4.1.3.md), and [4.1.3 installation](INSTALL.md#upgrading-to-413).


**Ciru v4.1.2 — October 5, 2026.** Adds the verified synchronization barrier in the first DFlash convolution/residual/RMSNorm kernel used by `ornithtest`. Explicit 262,144-token output ceilings now use the fully rendered prompt's remaining context. Weights, pinned engine wheels and sampling defaults are unchanged. [Patch notes](PATCH-NOTES-4.1.2.md) · [Upgrade existing installations](INSTALL.md#upgrading-to-412).

OMP clients must advertise `input: [text, image]`, `reasoning: true`, `contextWindow: 262144` and `maxTokens: 262144` to use vision and the full output ceiling.

**Ciru v4.1.1 — September 30, 2026.** Qualified repairs and a retained subset of [community PR #2](https://github.com/ciru-ai/ornith-ciru-halo-agent/pull/2). Ornith Q8/Q16 recurrence preserves the original packed single-token arithmetic, and attention uses per-query partitions.

The recent improvements would not have been possible without **Cinderblade from the toolbox discord**.

Measured complete-output comparisons: code +4.07% at the 44 GiB pool; code +5.89% and JSON +4.76% at 16 GiB. All compared core token IDs, reasoning and answers match. [Results and limits](evaluation/CIRU-V4.1.1-QUALIFICATION.md).

**Ciru v4.0.1 — September 25, 2026.** This maintenance patch supplies the
optimized cache path at startup and lets operators raise the image count per
request with `--max-images-per-prompt N`. The default remains one image; the
limit counts images retained in chat history. Model weights, native libraries
and pinned engine wheels are unchanged. See the [patch notes](RUNTIME-FIXES.md)
and [update instructions](INSTALL.md#updating-from-400-to-401).

The September 17 v4.0.0 release combined prefill, attention, draft-path,
prefix-cache and native tool-runtime improvements. Its measurements below
remain the v4.0.0 results, not new v4.0.1 performance claims.

# Ornith1.5 Ciru Halo Agent (vllm strix halo)

![Ornith1.5 Ciru Halo Agent — local agents on AMD Strix Halo](assets/ciru-halo-agent.png)

**A local AI team, built around AMD Strix Halo.** Ornith1.5 **Ciru Halo Agent** combines a custom quantization of Ornith's 35B-A3B mixture-of-experts model with a purpose-built vLLM/ROCm runtime for fast coding, tool use, and concurrent agents.

The design starts with the hardware: packed four-bit weights, four-bit activation paths, Strix Halo four-bit matrix instructions, specialized kernels, and adaptive DFlash2 speculative decoding. Prefix caching and a shared memory pool let agents return to long working histories.

## Ciru v4 results

**Ciru v4 accelerates long-context ingestion and returning-agent workflows.**
The new measurements below cover the complete combined runtime. Existing
benchmark sections retain their **Ciru v3** version labels and comparison models.

| Workload | Previous published Ciru | Ciru v4 | Change |
| --- | ---: | ---: | ---: |
| Cold prefill, 65,536 input tokens | 1,286.66 tok/s | 1,624.19 tok/s | +26.23% |
| Cold prefill, 253,952 input tokens | 667.61 tok/s | 870.11 tok/s | +30.33% |
| Cached 63K history, C1 mean decode | 123.34 tok/s | 147.69 tok/s | +19.74% |
| Cached near-256K history, C1 mean decode | 94.00 tok/s | 118.81 tok/s | +26.40% |
| Cached near-256K history, C8 mean decode | 11.10 tok/s | 19.86 tok/s | +78.93% |
| Short coding batch, C8 aggregate | 294.89 tok/s | 304.80 tok/s | +3.36% |

These final measurements used Ciru; the accepted published controls used Sozo
and their original runtime. They compare complete products, including those
host/runtime differences. They do not isolate a kernel's causal contribution.
The 97 measured requests ended naturally without API errors; **90/90 coding
outputs passed both native base and extended checks**. Cold-prefill requests
produced a natural 17-token acknowledgement, not sustained reasoning output.

**Tradeoffs:** short C2/C4 aggregate throughput fell 6.22%/8.84%; C4 mean decode
fell 2.98%. Cached near-256K C8 decode rose 78.93%, but mean first-token latency
increased from 18.03 s to 25.29 s. Its complete batch still fell from 53.64 s
to 49.20 s. See the [complete final speed comparison](evaluation/CIRU-V4-PERFORMANCE.md).

A separate same-host C1 HE0–9 test of the combined numerical parent measured
decode gains of 19.81% and 16.15% in two comparisons, pooled +17.97%. Both were
positive, but their spread exceeded the preset uncertainty rule. HF passed
38/40 and final 36/40; all failures were the known HE1 case also seen at BF16.
This is a familiar speed/health panel, not a full HumanEval score.

### BF16 fidelity

Short-prefix probability arrays were byte-identical to the previous HF model
across 3,072 positions. On the reused 32K and 64K panels, each with 1,024 scored
positions, the changes were mixed:

| Metric | 32K: HF → final | 64K: HF → final |
| --- | ---: | ---: |
| Panel perplexity, lower is better | 7.96626 → 8.00191 | 5.12747 → 5.09702 |
| KL distance from BF16, lower is closer | 0.222280 → 0.212698 | 0.116232 → 0.119387 |
| BF16 top-token agreement | 80.9570% → 80.7617% | 88.2813% → 88.0859% |
| BF16 top token in model top five | 97.3633% → 97.8516% | 98.9258% → 99.0234% |
| Top-five overlap | 74.6289% → 74.9609% | 81.5625% → 81.6602% |

At 32K, 38 top tokens were recovered and 40 lost; at 64K, 28 were recovered
and 30 lost. One diagnosed 32K confident error was a real top-token recovery;
another only crossed the probability threshold. These panels were reused during
diagnosis and are not fresh held-out quality evidence. The measurements belong
to the combined numerical parent, whose code and native libraries were preserved
when the final R03/R04 tool changes were added. See [32K fidelity](evaluation/CIRU-V4-FIDELITY32.md),
[64K fidelity](evaluation/CIRU-V4-FIDELITY64.md) and [composition proof](evaluation/CIRU-V4-COMPOSITION.json).

**Historical Ciru v3 measurements on AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151):**

- **178 tok/s single-request decode** on the ten-question coding speed screen, with **166 ms mean time to first token**.
- **295 tok/s aggregate at eight concurrent requests**, completing the ten-question batch in **5.52 seconds**.
- **1,287 tok/s cold prefill at 64K** and **668 tok/s near 256K**—**1.62× and 2.71×** the recorded Q4_K_XL prefill rates, respectively.
- **256K request context capacity**, eight active sequences, and a **44 GiB shared KV/state pool**.
- **123 tok/s cached C1 decode at 63K history**, with all ten return-task health checks passing.

These figures describe specific workloads, rather than a universal generation rate. The benchmark tables below include quality results and the workloads where other builds are faster.

**[Full benchmarks, runner builds, serving settings, and limitations](https://llm.ciru.ai/research/ornith-strix/)** · **[Ciru v4 source snapshot](runtime/ciru-v4-ornith-source.tar.gz)** · **[Source repository](https://github.com/ciru-ai/ornith-ciru-halo-agent)** · **[Credits](CREDITS.md)**

## Earlier September 13 runtime fixes

**Historical plugin-only update; Ciru v4 requires the broader update in [INSTALL.md](INSTALL.md).** This release fixes a reproduced graph-replay cache-corruption crash and rejects malformed or ambiguous tool calls with explicit API errors. The weights, kernels and adaptive DFlash2 settings are unchanged. [Update instructions, root cause and validation scope](RUNTIME-FIXES.md).

## Running Ciru Halo Agent

**Use the accompanying Ciru runtime.** This is a custom packed checkpoint and serving stack; installing stock vLLM and pointing it at the weights does not reproduce this build.

The measured profile targets Linux on AMD Strix Halo, with enough unified memory for the model, drafter, and context pool. Peak whole-host memory during the recorded production campaign was **95.35 GB**. That includes other host processes and is not the model-file size or a minimum-memory guarantee. The hardware used for this work has 128 GB unified memory.

```bash
uvx --from huggingface_hub hf download \
  jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo \
  --local-dir ./ciru-halo-agent
cd ciru-halo-agent
bash runtime/INSTALL-ORNITH-RUNTIME.sh "$PWD/installed-runtime"
bash bundle/serve.sh --host 127.0.0.1 --port 8000
```

Read **[Installation and build instructions](INSTALL.md)** first for Linux prerequisites, the runtime environment, source rebuild commands, and deployment details. The shipped stack was checked in an isolated installation on an existing Strix Halo test host; this is separate from validation on a fresh external machine. Source is available in **[ciru-ai/ornith-ciru-halo-agent](https://github.com/ciru-ai/ornith-ciru-halo-agent)**.

### Vision / image input

Image input remains enabled by default in Ciru v4, as introduced in runtime 1.0.2. The explicit image launcher is also available:

```bash
bash bundle/serve-vision.sh --host 127.0.0.1 --port 8000
```

**The matching vision encoder and projector are already included.** This vLLM checkpoint uses native BF16 tensors, not a separate llama.cpp `mmproj` GGUF. All **333 `model.visual.*` tensors**, including the six `model.visual.merger.*` projector tensors, are in [protected-00.safetensors](bundle/models/target/protected-00.safetensors), indexed by [model.safetensors.index.json](bundle/models/target/model.safetensors.index.json). The image processor configuration is included alongside them. No additional projector download or `--mmproj` argument is needed; a GGUF projector cannot be substituted into this runtime.

The default profile accepts **one image across the complete request**, including image-bearing chat history, with up to **eight active requests** and a **1,048,576-pixel preprocessing budget per image**. To raise the request limit, pass `--max-images-per-prompt 4` to `bundle/serve.sh` and restart. Higher counts use more encoder and context capacity; the retained vision checks used one image per request. Video is disabled. The profile uses `TRITON_ATTN` for the BF16 image encoder and retains the 262,144-token request capacity, 44 GiB shared KV/state pool, A4/IU4 target paths, prefix caching, and adaptive DFlash2. Pass `--text-only` to disable image input.

Send an image through the OpenAI-compatible chat API; this example embeds a local file without exposing a local-file server:

```bash
python3 - ./image.png <<'PYIMAGE'
import base64, json, mimetypes, pathlib, sys, urllib.request
image = pathlib.Path(sys.argv[1])
mime = mimetypes.guess_type(image.name)[0] or "image/png"
data_url = f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode()
payload = {
    "model": "ciru-halo-agent",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": data_url}}
    ]}],
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": False}
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=180) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
PYIMAGE
```

Earlier retained image-profile checks passed **8/8 API image requests**, **8/8 native Hermes image clients**, and **20/20 post-vision HumanEval base and extended checks**, with **96.91 GB peak whole-host memory**. A mixed batch scored **7/8**: all four images were correct, but one text-only request refused. These are narrow, partly image-cache-warm checks of the earlier packaged image profile, not a new evaluation of this publication or broad vision-quality evidence. The published weights and launcher settings were checked separately. See [vision checks and limits](VISION.md).

### Controlling speculative draft depth

The default remains adaptive C1 drafting (15/7/off through 32,768 computed tokens, then 15), with 7 draft tokens for 2–8 concurrent requests. Explicit `--draft-tokens` now selects a fixed **single-request** depth at every supported context length:

```bash
# Lower C1 drafting for a low-acceptance workload, including at 64K:
bash bundle/serve.sh --cache-gib 22 --draft-tokens 7

# Disable C1 proposals while retaining the drafter/cache for concurrent requests:
bash bundle/serve.sh --cache-gib 22 --draft-tokens 0

# Fixed C1 depth 15:
bash bundle/serve.sh --draft-tokens 15
```

Omit `--draft-tokens` for the default policy. An explicit flag takes precedence over `ORNITH_C1_POLICY`; otherwise that environment variable is respected, defaulting to `auto`. These options also work with `bundle/serve-vision.sh`. C2–8 always retains depth 7, including when C1 depth is 0. A depth of 0 does not unload the drafter or reduce its allocated memory. Unsupported depths are rejected explicitly.

The startup JSON reports the effective `ORNITH_C1_POLICY` (`auto`, `k0`, `k7`, or `k15`). Its speculative configuration still reports the maximum 15-token graph envelope; the scheduler selects the actual C1 depth within it. This update was checked without loading the model. Fixed-depth long-context speed and transition correctness have not been newly GPU-validated; lower acceptance suggests trying shorter drafting, not a guaranteed speedup.

### Serving behavior

| Setting | Measured profile |
| --- | --- |
| Request context limit | 262,144 tokens, including prompt and output |
| Active requests | Up to 8; additional requests queue |
| Shared KV/recurrent-state pool | 44 GiB |
| Prefix caching | Enabled, including recurrent-state reuse |
| Speculative drafter | Trained Ornith DFlash2 by jzinno |
| Single-request speculation | Adaptive 15/7/off below 32,768 computed tokens; 15-token drafting on longer contexts |
| Concurrent speculation | 7-token drafting for 2–8 active requests |
| Agent integration | OpenAI-compatible API, tool calling, preserved Ornith chat/reasoning behavior |
| Default release evidence | Text profile |

The pool is shared. **Eight active requests does not mean eight unrelated, fully populated 256K histories fit at once.** Output must fit in the remaining request context; there is no separate promise of 256K output after a 256K input. Preserve the supplied chat template and use the client’s intended thinking setting.

## Historical v3 coding speed and concurrency

HumanEval 0–9, thinking off, greedy sampling, natural end-of-sequence, cold prompt salts. Each row runs all ten questions. These are **speed and health checks**, not evidence of general coding quality.

| Concurrent requests | Ten-task completion | Mean request decode | Mean first-token latency | Aggregate throughput |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.65 s | **178.17 tok/s** | **0.166 s** | 148.69 tok/s |
| 2 | 8.47 s | 111.59 tok/s | 0.259 s | 187.07 tok/s |
| 4 | 7.12 s | 73.54 tok/s | 0.340 s | 222.74 tok/s |
| 6 | 5.99 s | 60.81 tok/s | 0.524 s | 264.84 tok/s |
| 8 | **5.52 s** | 55.84 tok/s | 0.709 s | **294.89 tok/s** |
| 10 offered / 8 active | 7.40 s | 55.38 tok/s | 1.297 s | 249.20 tok/s |

**60/60 base and extended health checks passed.** Mean request decode measures generation after the first token. Aggregate throughput divides all output tokens by complete batch wall time, including scheduling and prefill. They are different metrics, and output length varies between configurations.

For comparison, the fresh **Q4_K_XL + DFlash2 (7-token)** C1 run at 256K capacity completed the same ten-question batch in **17.32 s**, at **131.65 tok/s mean request decode** and **0.351 s mean first-token latency**, with 10/10 health. Ciru Halo Agent completed that batch **1.63× faster**. The measured ROCmFP4 C1 MTP4 mode took **19.56 s**, at **113.31 tok/s**. Complete concurrency sweeps, alternative modes, and their health outcomes are on the research page.

## Historical v3 prompt ingestion

Cold input, without a prefix-cache hit. These are backend prefill-counter rates; each request then generated a natural 17-token acknowledgement. The short acknowledgement is not a sustained decode or reasoning benchmark.

| Input tokens | Ciru Halo Agent | Q4_K_XL | ROCmFP4 |
| ---: | ---: | ---: | ---: |
| 1,024 | **1,868 tok/s** | 1,056 tok/s | 1,058 tok/s |
| 8,192 | **1,715 tok/s** | 1,174 tok/s | 1,046 tok/s |
| 32,768 | **1,509 tok/s** | 998 tok/s | 842 tok/s |
| 65,536 | **1,287 tok/s** | 794 tok/s | 629 tok/s |
| 131,072 | **983 tok/s** | 552 tok/s | 414 tok/s |
| 253,952 | **668 tok/s** | 246 tok/s | 252 tok/s |

The complete near-256K cold request took **380.83 s**, versus **1,032.06 s for Q4_K_XL** and **1,007.04 s for ROCmFP4**. ROCmFP4’s lower-depth rows come from its retained author-profile sweep; the near-256K point is the fresh cold capture. The research page identifies each source run.

## Historical v3 returning agents and large histories

All rows below use **all ten coding questions**, populated histories, and confirmed prefix-cache reuse. Timing includes queueing, suffix prefill, and generation; preparing the shared history is separate.

| Shared history | Concurrency | Ciru mean decode | Ciru mean first-token latency | Ciru ten-task completion |
| ---: | ---: | ---: | ---: | ---: |
| 63,000 tokens | 1 | **123.34 tok/s** | 0.848 s | 21.49 s |
| 63,000 tokens | 8 | **32.08 tok/s** | 3.968 s | 13.69 s |
| 253,952 tokens | 1 | **94.00 tok/s** | 4.619 s | 66.37 s |
| 253,952 tokens | 8 | **11.10 tok/s** | 18.033 s | 53.64 s |

**40/40 base and extended health checks passed.** At 63K/C1, Q4_K_XL and ROCmFP4 completed the ten returns in **52.24 s** and **33.45 s**. Near 256K/C1, they completed in **75.93 s** and **61.06 s**: ROCmFP4 finished that batch sooner despite Ciru’s higher generation rate. First-token latency and output length also matter.

The tested Q4_K_XL and ROCmFP4 configurations did not share the warmed prefix successfully across eight request slots. Their **cached C8 comparison remains unavailable**; this does not establish that the runners cannot support it.

## Historical v3 quality and actual agent work

The release was evaluated separately from the short coding speed screen. Full EvalScope runs use the native **541 IFEval, 1,319 GSM8K, and 164 HumanEval** tasks. Tool and Hermes scores use their own native graders.

| Quality measure | Ciru Halo Agent | Q4_K_XL | ROCmFP4 |
| --- | ---: | ---: | ---: |
| Full GSM8K | **1,254/1,319 · 95.07%** | 1,252/1,319 · 94.92% | 1,228/1,319 · 93.10% |
| Full HumanEval | 144/164 · 87.80% | **151/164 · 92.07%** | 147/164 · 89.63% |
| Full IFEval, prompt strict | 390/541 · 72.09% | 413/541 · 76.34% | **421/541 · 77.82%** |
| Difficult HumanEval subset | **4/6** | 2/6 | 0/6 |
| Tool suite, 69 core + 15 hard | 140/168 points | 138/168 points | **143/168 points** |
| BF16 top-token agreement | 45/56 | **47/56** | 41/56 |

Full EvalScope used C8, thinking off, greedy sampling, and a 32,768-token response allowance. Ciru and ROCmFP4 used 256K capacity; Q4’s retained full-suite run used 65K capacity and its native chat protocol. Ciru had two IFEval length stops; ROCmFP4 had two GSM8K length stops. All other requests in those two runs completed naturally. Their complete evaluation wall times, including orchestration and grading, were **4,006 s** and **5,024 s**, respectively.

The difficult subset and tool suite are single blocks, not repeated confidence estimates. BF16 agreement uses seven held-out documents with eight adjacent prefix positions each: these 56 correlated anchors measure short-prefix fidelity, not task accuracy. The original BF16 source was fixed; the comparison quants’ exact source-weight ancestry is not established.

### Historical v3 Hermes: three complete passes per model and concurrency

Twenty native agent scenarios per pass, three seeds, 256K capacity, up to 64 turns, and output bounded by remaining context. Scores below are native scores, not percentages of tasks passed.

| Model | C1 mean score | C1 mean workflow time | C8 mean score | C8 mean workflow time |
| --- | ---: | ---: | ---: | ---: |
| **Ciru Halo Agent** | 90.33 | 820.00 s | 92.00 | 369.66 s |
| Q4_K_XL | 94.00 | 753.18 s | 94.00 | 358.95 s |
| ROCmFP4 | 93.67 | 681.84 s | 91.00 | 550.07 s |

Ciru’s C8 mean workflow time was **1.49× faster than ROCmFP4**, while Q4 was slightly faster than Ciru in this suite. One interrupted Q4 transport run was retained separately and replaced once; it is excluded from the three complete passes. The research page includes every pass, score range, token timing, and failure count.

## Where the historical v3 speed advantage changes

Ciru Halo Agent’s strongest results are speculative coding, prompt ingestion, and shared-history workloads. **It is not the fastest model on every task.** Four ordinary prose tasks with automatic speculation measured **49.90–63.27 tok/s**, compared with approximately **59.5–59.8 tok/s for Q4** and **73.9–74.1 tok/s for ROCmFP4** in the retained no-speculation prose captures. Ciru’s target-only diagnostic measured approximately **56–58 tok/s**. The diagnostic skips draft work; its allocations still include the drafter.

256K capacity and the successful return tests do not establish uniformly strong reasoning across every long document. The main benchmark tables measure the text profile. The optional vision profile and its separate, earlier validation limits are described above. Hardware, software builds, memory pools, prompt mix, and speculation acceptance affect results.

## Credits and sponsorship

**Ciru Halo Agent is built by Ciru / Crown ([jcbtc](https://huggingface.co/jcbtc)). AMD provided me with a Ryzen AI Halo, and I am sponsored by AMD.**

- **[Ornith Team](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)** created the base Ornith model. Ciru’s work is its hardware-specific quantization, kernels, serving integration, and evaluation; we do not claim authorship of the base model’s training.
- **[jzinno](https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2)** trained the Ornith DFlash2 drafter, initialized from **[z-lab’s Qwen3.5 DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash)**. Credit also goes to the **[DFlash](https://arxiv.org/abs/2602.06036)** and **[DFlash 2](https://inco.ai/blog/dflash2/)** authors and the NVIDIA Nemotron dataset contributors whose work supports that drafter.
- **[XGrammar](https://github.com/mlc-ai/xgrammar)** supplies the native grammar implementation; its license and pinned Ciru patch are included in the v4 source snapshot.
- **[Qwen](https://github.com/QwenLM)** supplied the underlying Qwen model architecture used by this Ornith checkpoint.
- **[vLLM](https://github.com/vllm-project/vllm)**, **[AMD ROCm](https://github.com/ROCm)**, **[AITER](https://github.com/ROCm/aiter)**, **[Composable Kernel](https://github.com/ROCm/composable_kernel)**, **[PyTorch](https://github.com/pytorch/pytorch)**, and **[Triton](https://github.com/triton-lang/triton)** provide the runtime and compiler foundations.
- Thanks to **[peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll/Unsloth-Ornith-1.5-35B-A3B)**, **[Daniel Han Chen](https://github.com/danielhanchen/llama.cpp)**, **[julianmb](https://huggingface.co/julianmb/Ornith-1.5-35B-A3B-ROCmFP4-GGUF)**, **[HaloFPX](https://github.com/julianmb/halofpx)**, and the llama.cpp/ROCmFP4 community for the comparison builds. Their artifacts are baselines, not components of Ciru’s target weights. The Q4 quant is a community Unsloth-style release, not an official Unsloth upload.

See **[CREDITS.md](CREDITS.md)** for pinned provenance and license details. The target model follows Ornith’s MIT declaration; the DFlash2 companion is Apache-2.0 and runtime components retain their own licenses.
