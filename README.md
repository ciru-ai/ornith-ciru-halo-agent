# Ornith1.5 Ciru Halo Agent

## Ciru v4.1.1 — September 30, 2026

The retained community subset accelerates dense M1 projections, compact head rows and the shared expert while preserving the released numerical storage boundaries. The shared activation uses a 384 KiB reference table derived from the released compiled expression; all 65,280 finite BF16 gate patterns match by bits. The PR's router and small BF16 projection substitutions are excluded after actual GPU replay reproduced arithmetic forks.

Q/K normalization now selects the qualified reduction bodies structurally. Compiler-assigned kernel numbers previously allowed the same input graph to silently switch reduction order. Startup refuses serving when the qualified substitution did not execute. Cache identity includes all executable plugin, native, launcher and reference-data files.

Weights, BF16 vision tensors, trained draft/MTP tensors, pinned engine wheels and model sampling policy are unchanged. Complete scored requests use the full 262,144-token context, native reasoning and no completion cap. These are bounded runtime comparisons, with existing broad benchmark scores retaining their original build labels.

Community decode optimizations: **Daniele Zannotti**, [PR #2](https://github.com/ciru-ai/ornith-ciru-halo-agent/pull/2) (`7220d47a9428ec75dd0b1506168541b8bd9203b4`). Ciru supplied the numerical repairs, runtime consistency fixes and qualification of the retained subset. Upstream authorship is preserved in Git history.

[Qualified results and limits](CIRU-V4.1.1-QUALIFICATION.md).

## Ciru v4.0.1 — September 25, 2026

This launcher patch supplies `ORNITH_OPTIMIZED_CACHE` and adds
`--max-images-per-prompt N` for image-bearing agent histories. The default is
one image across the complete request. Weights, native libraries and pinned
engine wheels are unchanged. See [patch notes](RUNTIME-FIXES.md) and the
[published update instructions](https://huggingface.co/jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo/blob/main/INSTALL.md).

## Runtime 1.0.2 — September 15, 2026

Automatic tool calls now use native vLLM/XGrammar schema constraints, enforcing
required arguments without changing prompt-visible messages or tool definitions.
The Ornith and Apodex packages enable their existing BF16 image encoder/projector
by default; `--text-only` disables image input. Apodex's inherited 32K output cap
is removed. IU4 weights, inference kernels and speculation settings are unchanged.
See [patch notes](RUNTIME-FIXES.md).

**Fast local agents on AMD Ryzen AI Max+ Strix Halo.** This repository contains the custom vLLM plugin, adaptive speculative decoding integration, and eight native GPU kernels used by the Ciru Halo Agent release.

- [Download the model and pinned runtime](https://huggingface.co/jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo)
- [Install and run](https://huggingface.co/jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo/blob/main/INSTALL.md)
- [Interactive benchmark comparisons](https://llm.ciru.ai/research/ornith-strix/)
- [Build the native kernels](BUILD.md)
- [Credits and licenses](CREDITS.md)

The measured production profile provides 262,144-token context capacity, up to eight active sequences, a 44 GiB shared KV/state pool, prefix caching, and adaptive DFlash2 speculation. Input and output share the context budget. Eight completely full 256K histories are not guaranteed.

Recorded short HumanEval 0–9 coding bursts reached **178.17 tokens/s mean request decode at C1** and **294.89 tokens/s aggregate at C8**, completing the ten-task C8 batch in **5.52 seconds**. HumanEval 0–9 is a speed/health screen; these are workload-specific measurements, not prose or sustained-load guarantees. Quality and baseline limitations are reported on the benchmark page.

## September 13 runtime fixes

**Existing installations should update the serving plugin and restart.** This release fixes a reproduced graph-replay cache-corruption crash and rejects malformed or ambiguous tool calls with explicit API errors. The weights, kernels and adaptive DFlash2 settings are unchanged. [Update instructions, root cause and validation scope](RUNTIME-FIXES.md).

## Install

```bash
uvx --from huggingface_hub hf download \
  jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo \
  --local-dir ./ciru-halo-agent
cd ciru-halo-agent
# Install OS prerequisites from INSTALL.md before this step.
bash runtime/INSTALL-ORNITH-RUNTIME.sh "$PWD/installed-runtime"
bash bundle/serve.sh --host 127.0.0.1 --port 8000
```

Use the supplied runtime; stock pip vLLM does not implement this custom model format. The API model name is `ciru-halo-agent`. Validated on NixOS Strix Halo with 128 GB memory; a mainstream Ubuntu 26.04 setup recipe is provided with its validation scope stated explicitly.

## Source layout

`plugin-site/` contains the serving plugin; `kernels/` contains source for all eight native libraries; `scripts/build-native.sh` rebuilds them using the pinned runtime's HIP toolchain. Engine source archives and matching wheels are distributed alongside the model, including vLLM/AITER upstream licenses and release overlay documentation. Model weights are hosted on Hugging Face rather than GitHub.

Stable code/module identifiers in the implementation are compatibility interfaces; the public model name is **Ornith1.5 Ciru Halo Agent**.

**Sponsorship disclosure: Ciru is sponsored by AMD.**
