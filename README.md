# Ornith1.5 Ciru Halo Agent

**Fast local agents on AMD Ryzen AI Max+ Strix Halo.** This repository contains the custom vLLM plugin, adaptive speculative decoding integration, and eight native GPU kernels used by the Ciru Halo Agent release.

- [Download the model and pinned runtime](https://huggingface.co/jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo)
- [Install and run](https://huggingface.co/jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo/blob/main/INSTALL.md)
- [Interactive benchmark comparisons](https://llm.ciru.ai/research/ornith-strix/)
- [Build the native kernels](BUILD.md)
- [Credits and licenses](CREDITS.md)

The measured production profile provides 262,144-token context capacity, up to eight active sequences, a 44 GiB shared KV/state pool, prefix caching, and adaptive DFlash2 speculation. Input and output share the context budget. Eight completely full 256K histories are not guaranteed.

Recorded short HumanEval 0–9 coding bursts reached **178.17 tokens/s mean request decode at C1** and **294.89 tokens/s aggregate at C8**, completing the ten-task C8 batch in **5.52 seconds**. HumanEval 0–9 is a speed/health screen; these are workload-specific measurements, not prose or sustained-load guarantees. Quality and baseline limitations are reported on the benchmark page.

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
