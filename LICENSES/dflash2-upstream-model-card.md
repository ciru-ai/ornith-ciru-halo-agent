---
license: apache-2.0
library_name: sglang
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
- z-lab/Qwen3.5-35B-A3B-DFlash
tags:
- speculative-decoding
- speculative-decoding-draft
- dflash2
- block-diffusion
- draft-model
- sglang
- ornith
- qwen
- qwen3
- qwen3.5
---

# Ornith 1.5 35B A3B DFlash2

This repository contains a DFlash2 draft model for
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).
It is not a standalone language model: a compatible Ornith target checkpoint is
required at inference time, and the target verifies every proposed token.

The checkpoint was trained on projected teacher features from 100,014,884
generated assistant completion tokens (108,729 conversations). The full
generation dataset is released separately as
[`jzinno/Ornith-1.5-35B-A3B-Nemotron-v2-100M`](https://huggingface.co/datasets/jzinno/Ornith-1.5-35B-A3B-Nemotron-v2-100M).

## Training method

Target responses were generated first. A separate offline teacher-forcing pass then
loaded the revision-pinned BF16 Ornith teacher, ran each completed sequence once, and
selected hidden layers 1, 6, 11, 16, 22, 27, 32, and 37. Feature capture used
8 NVIDIA H200 141GB HBM3e GPUs in independent contiguous partitions; the selected drafter was
trained on 1 NVIDIA H200 141GB HBM3e GPU.

The inherited 2048-wide projection from
[`z-lab/Qwen3.5-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash)
was applied during that pass and kept frozen. Only the projected BF16 features, token
IDs, and loss masks were cached. Drafter training therefore did not keep the full
teacher resident; it loaded only Ornith's frozen embedding and language-model head with
the warm-started draft network.

The Qwen3.5 checkpoint is a plain DFlash drafter. Its compatible parameters initialize
the shared draft network before adaptation to the continued-pretrained Ornith target.
The DFlash2-only grouped causal convolutions and path selector start as exact no-ops:
each convolution has a unit self-tap with zero predecessor/dynamic correction, and the
selector's successor codebook is zero so its scores reduce to the inherited DFlash
logits. The upgraded network is therefore numerically equivalent to the plain DFlash
warm start before optimization, rather than introducing a random functional change.

The final recipe used:

- six draft layers, a 16-token training block, and a 4096-token sliding window;
- 128 anchors, selector rank 256, and selector top-k 16;
- microbatch 1 and effective batch 32;
- AdamW with learning rate 0.0001 and a warmup-plus-cosine schedule;
- 3,228 optimizer steps.

## Held-out serving evaluation

The serving evaluation uses 40 prompts held outside the complete training-source
reservoir: 10 each from chat, code, math, and STEM. Both hardware matrices use
concurrency 1, 4, and 8, greedy sampling, and up to 512 output tokens per request. The
target is the official Ornith NVFP4 checkpoint, KV cache is FP8 E4M3, Marlin runs the
MoE, and FlashInfer runs attention. The four matched configurations are autoregressive
decoding, Ornith's built-in NEXTN head, the unadapted Qwen3.5 DFlash warm start, and
this Ornith-adapted DFlash2 drafter. Both external drafts propose up to 10 tokens.

Ornith's NEXTN/MTP head is inherited from Qwen3.5 and was not trained during Ornith's
continued pretraining, so it is not expected to be a strong speculative baseline. Its
result is included as a reference for the head shipped with the target checkpoint.

![Held-out throughput and accepted-token comparison for ornith-ai/Ornith-1.5-35B-A3B-NVFP4 autoregressive decoding, z-lab/Qwen3.5-35B-A3B-DFlash, and jzinno/Ornith-1.5-35B-A3B-DFlash2](assets/serving-benchmark.png)

### NVIDIA DGX Spark (GB10)

The first matrix was measured on one NVIDIA DGX Spark.

| Configuration | Concurrency | Overall tok/s | Per-user tok/s | Median TTFT | Median TPOT | Accepted length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Autoregressive | 1 | 79.3 | 80.7 | 118.0 ms | 12.39 ms | n/a |
| Autoregressive | 4 | 163.1 | 46.0 | 246.1 ms | 21.74 ms | n/a |
| Autoregressive | 8 | 202.5 | 31.8 | 436.1 ms | 31.43 ms | n/a |
| NEXTN | 1 | 72.1 | 74.8 | 160.3 ms | 13.37 ms | 1.92 |
| NEXTN | 4 | 138.2 | 39.0 | 255.4 ms | 25.63 ms | 1.99 |
| NEXTN | 8 | 172.6 | 26.5 | 536.5 ms | 37.67 ms | 2.01 |
| Qwen3.5 DFlash warm start | 1 | 103.6 | 104.3 | 144.7 ms | 9.59 ms | 3.48 |
| Qwen3.5 DFlash warm start | 4 | 174.6 | 50.8 | 255.1 ms | 19.69 ms | 3.64 |
| Qwen3.5 DFlash warm start | 8 | 218.4 | 34.4 | 523.6 ms | 29.03 ms | 3.69 |
| Ornith DFlash2 | 1 | 114.2 | 115.4 | 145.1 ms | 8.67 ms | 4.02 |
| Ornith DFlash2 | 4 | 193.9 | 57.9 | 259.6 ms | 17.27 ms | 4.16 |
| Ornith DFlash2 | 8 | 236.5 | 37.2 | 539.0 ms | 26.92 ms | 4.18 |

### NVIDIA H200

The same matrix was measured on one NVIDIA H200 141GB HBM3e GPU.

| Configuration | Concurrency | Overall tok/s | Per-user tok/s | Median TTFT | Median TPOT | Accepted length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Autoregressive | 1 | 253.6 | 267.2 | 92.0 ms | 3.74 ms | n/a |
| Autoregressive | 4 | 712.6 | 225.0 | 181.8 ms | 4.45 ms | n/a |
| Autoregressive | 8 | 989.2 | 188.6 | 233.0 ms | 5.30 ms | n/a |
| NEXTN | 1 | 281.1 | 304.0 | 101.5 ms | 3.29 ms | 1.93 |
| NEXTN | 4 | 708.9 | 221.3 | 115.5 ms | 4.52 ms | 1.95 |
| NEXTN | 8 | 982.6 | 171.8 | 250.2 ms | 5.82 ms | 1.95 |
| Qwen3.5 DFlash warm start | 1 | 437.1 | 482.2 | 97.2 ms | 2.07 ms | 3.51 |
| Qwen3.5 DFlash warm start | 4 | 892.7 | 266.3 | 113.3 ms | 3.76 ms | 3.55 |
| Qwen3.5 DFlash warm start | 8 | 1190.0 | 217.3 | 250.1 ms | 4.60 ms | 3.56 |
| Ornith DFlash2 | 1 | 472.9 | 511.2 | 99.0 ms | 1.96 ms | 3.98 |
| Ornith DFlash2 | 4 | 981.0 | 312.1 | 112.2 ms | 3.20 ms | 3.90 |
| Ornith DFlash2 | 8 | 1351.6 | 244.3 | 237.0 ms | 4.09 ms | 3.97 |

On DGX Spark at concurrency 1, the draft configuration produced 114.2 output tokens/s,
44.0% above target-only serving, with mean accepted length
4.02 at concurrency 1. These measurements are specific to the recorded
software, hardware, prompt mix, quantization, and serving settings; they are not
general quality or speed claims. Machine-readable results are in
`spark-evaluation.json` and `h200-evaluation.json`.

## SGLang usage

Use an SGLang build with DFlash2 support. The release evaluation used SGLang commit
`710267dc4c817b38d9965346390cd56b59b54eda`.

```console
sglang serve \
  --trust-remote-code \
  --model-path ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \
  --revision 0f0b1b59b879ccde1353e6ebd0fb10c204d4c544 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend flashinfer \
  --moe-runner-backend marlin \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path jzinno/Ornith-1.5-35B-A3B-DFlash2 \
  --speculative-num-draft-tokens 10 \
  --speculative-draft-attention-backend flashinfer \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-ssm-dtype float32
```

Memory limits, context length, and concurrency should be set for the deployment
hardware rather than copied blindly from the evaluation setup.

## License and attribution

The checkpoint is released under Apache-2.0. It was initialized from the Apache-2.0
Qwen3.5 DFlash checkpoint and trained against the MIT-licensed Ornith target. The
training conversations derive from NVIDIA Nemotron Post-Training Dataset v2 under
CC-BY-4.0. See the linked repositories for their notices and model cards.

## References

- [DFlash 2: Keep Drafting Parallel](https://inco.ai/blog/dflash2/)
- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- [NeMo AutoModel speculative-drafting implementation](https://github.com/NVIDIA-NeMo/Automodel)
