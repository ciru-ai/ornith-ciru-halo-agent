# Ornith1.5 Ciru Halo Agent — credits and provenance

Ciru / Crown ([jcbtc](https://huggingface.co/jcbtc)) developed the hardware-specific quantization, native kernels, vLLM integration, adaptive serving policy, and benchmark campaign for this release. **Crown is sponsored by AMD.**

## Target model

The target derives directly from **[Ornith Team’s Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/tree/10fbf86fed7ecee4a061f8b499a618f46001cac1)**, revision `10fbf86fed7ecee4a061f8b499a618f46001cac1`. The original BF16 weights were the quantizer input. Ciru did not train the base Ornith model.

The pinned upstream card declares **MIT**. On September 9, 2026, the Hugging Face file inventory had no separate LICENSE or NOTICE file at either that pinned revision or `main`; both resolved to the same revision. The included [upstream model card](LICENSES/ornith-upstream-model-card.md) preserves that declaration; this release does not invent an upstream copyright statement. See the downloaded source inventory in [LICENSES/sources.json](LICENSES/sources.json).

The checkpoint uses the Qwen3.5 MoE architecture. Credit to the **[Qwen team](https://github.com/QwenLM)** and the upstream contributors acknowledged by the [Ornith model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/10fbf86fed7ecee4a061f8b499a618f46001cac1/README.md) for their foundational work. Ciru’s measurements should not be substituted for the upstream authors’ original model evaluations.

## Speculative drafter

The companion is **[jzinno/Ornith-1.5-35B-A3B-DFlash2](https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2/tree/9b4852c05fd00b672b7434b1bb105bc03c8682b0)**, revision `9b4852c05fd00b672b7434b1bb105bc03c8682b0`, declared **Apache-2.0**. Credit for training this drafter belongs to **jzinno**, not Ciru.

Its upstream card records initialization from **[z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash)**, also Apache-2.0. It attributes its training conversations to **NVIDIA Nemotron Post-Training Dataset v2**, under **CC-BY-4.0**, and releases the generated data as **[jzinno/Ornith-1.5-35B-A3B-Nemotron-v2-100M](https://huggingface.co/datasets/jzinno/Ornith-1.5-35B-A3B-Nemotron-v2-100M)**. These datasets are not redistributed as part of the target checkpoint.

Credit to the authors of **[DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)**, **[DFlash 2: Keep Drafting Parallel](https://inco.ai/blog/dflash2/)**, and the **[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)** speculative-drafting implementation referenced by the drafter authors. Ciru’s contribution is deployment and optimization of this trained companion on Strix Halo.

## Runtime foundations

- **[vLLM](https://github.com/vllm-project/vllm)** — Apache-2.0. The retained runtime wheel identifies itself as `0.1.0rc2.dev9+g9255fd9fb9.rocm100`, full commit `9255fd9fb9fedf4b29d574a8d8bb21d93892cc98`, with documented Ciru cache/serving changes. This is a custom runtime, not a claim that an unmodified stock wheel reproduces the results.
- **[AMD ROCm](https://github.com/ROCm)** — the GPU platform, compiler, and SDK. Individual components retain their own licenses and notices.
- **[AITER](https://github.com/ROCm/aiter)** and **[Composable Kernel](https://github.com/ROCm/composable_kernel)** — MIT-licensed kernel/runtime foundations; preserve their bundled notices.
- **[PyTorch](https://github.com/pytorch/pytorch)**, **[Triton](https://github.com/triton-lang/triton)**, **[Hugging Face Transformers](https://github.com/huggingface/transformers)**, tokenizers, and safetensors — preserve their component licenses and dependencies’ notices.
- **[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)** and its contributors — underlying linear-attention work retained within the vLLM source distribution and its license.

Use the packaged runtime source and notice files for the exact implementation provenance. A top-level model license does not replace the licenses of the companion drafter or software components.

## Comparison builds

These authors supplied independently developed artifacts used as comparison baselines. Their weights and serving engines are not the Ciru target model or its vLLM runner.

| Artifact / runner | Attribution and recorded identity |
| --- | --- |
| Community Q4_K_XL | [peculiar-ragdoll/Unsloth-Ornith-1.5-35B-A3B](https://huggingface.co/peculiar-ragdoll/Unsloth-Ornith-1.5-35B-A3B), revision `a015e1ea842854d64a1b31d4448419f6229d7a26`; community Unsloth-style quant, not an official Unsloth upload |
| Q4 Vulkan runner | [Daniel Han Chen’s llama.cpp fork](https://github.com/danielhanchen/llama.cpp/tree/d1a92352cbd417fd840b4e765c0b82f5fe3d1d89), revision `d1a92352cbd417fd840b4e765c0b82f5fe3d1d89`; also credit [llama.cpp](https://github.com/ggml-org/llama.cpp) and contributors |
| ROCmFP4 model | [julianmb/Ornith-1.5-35B-A3B-ROCmFP4-GGUF](https://huggingface.co/julianmb/Ornith-1.5-35B-A3B-ROCmFP4-GGUF), revision `a0db6d02a557324a04cecd3feb0ab3f08565521e` |
| Recommended ROCmFP4 runner | [julianmb/HaloFPX](https://github.com/julianmb/halofpx); its supplied engine reported ROCmFPX build 213 / `e87d53e`, acquired through installer revision `16bd8a020c23e20dd844de913c4ee7486f438000` |

The final ROCmFP4 comparisons use Julian’s recommended engine. They should not be described as results from the separately investigated Laurent runner. Broader thanks to the ROCmFP4 and Strix Halo inference community for their public work.

## Evaluation

Credit to **[EvalScope](https://github.com/modelscope/evalscope)** and the creators of **HumanEval, EvalPlus, GSM8K, IFEval**, and the **Hermes agent ecosystem**. Ciru used full public suites and native agent/tool scenarios alongside a small difficult subset and fixed BF16-prefix probes. Native benchmark licenses and attribution remain applicable to any redistributed fixtures.

The public **[benchmark report](https://llm.ciru.ai/research/ornith-strix/)** records the model-specific scores, timing definitions, source runs, runner builds, serving settings, and limitations. No endorsement by any upstream project or comparison author is implied.
