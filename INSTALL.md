# Ciru v4.1.4 concurrency update

Reuse the installed pinned engine and weights. Use this release's runtime/UPGRADE-CIRU.py for the 4.1.4 distribution. The preceding release and caches remain available for rollback. Numerical and serving qualification is scoped to the new validation report.

# Install and run Ornith1.5 Ciru Halo Agent

## Upgrading to 4.1.3

Stop this model's server and run from the existing download directory:

```bash
uvx --from huggingface_hub hf download jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo \
  runtime/UPGRADE-CIRU.py --local-dir .
uv run --with huggingface_hub python runtime/UPGRADE-CIRU.py \
  --model ornith --directory .
```

The updater verifies every file selected by the control-file manifest before swapping code. It preserves model weights and caches and saves previous code under `.ciru-upgrade-backups/`. Existing v4 installations reuse the pinned engine. For installations older than v4, add `--install-runtime "$PWD/.runtime-4.1.3"` and set `ORNITH_RUNTIME_ROOT` to that new directory. Restart the usual launcher. The complete 262,144-token shared prompt/output context, native reasoning and existing thinking/vision controls remain available. Do not set a short quality-test output cap merely because a harness supplies one.

The default is now AUTO K0/K3/K7 and guarded concurrent K0. Fresh guarded admissions can incur fresh-prefill work because prefix-cache reads are bypassed. See [results and limits](README.md). Earlier upgrade sections below are retained as version history; their version-specific removal and update commands are not instructions for the current 4.1.3 installation.


## Upgrading to 4.1.2

Stop this model's server, then run these commands from your existing download directory. The updater downloads code and release files without downloading model weights. It verifies checksums, replaces the complete serving bundle, removes superseded package metadata, preserves `bundle/models` and `bundle/cache`, and saves the previous code in `.ciru-upgrade-backups/`.

```bash
uvx --from huggingface_hub hf download jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo \
  runtime/UPGRADE-CIRU.py --local-dir .
uv run --with huggingface_hub python runtime/UPGRADE-CIRU.py \
  --model ornith --directory .
```

Existing Ciru v4 installations reuse their pinned engine: restart your usual `bundle/serve.sh` command afterward. The executable fingerprint selects a fresh compile namespace automatically.

For versions **older than Ciru v4**, or an installation without version metadata, add `--install-runtime "$PWD/.runtime-4.1.2"` to the updater command. This installs the supplied engine into a new directory and preserves the old runtime. Then set `ORNITH_RUNTIME_ROOT="$PWD/.runtime-4.1.2"` for the usual launcher. The Linux prerequisites below still apply.

For a **new installation**, use the complete download and runtime installer below; the latest download already includes this patch. No requantization is needed.

For **OMP**, mark the model's `models.yml` entry with `input: [text, image]`, `reasoning: true`, `contextWindow: 262144` and `maxTokens: 262144`. Reload by exiting and resuming OMP from the same directory. Input and output share the 262,144-token context. The server's default is one image across the request; use the existing `--max-images-per-prompt N` option for larger image-bearing histories.

## Updating to runtime 4.1.1

Download the complete plugin, native libraries (including both `shared_math_reference` files), packaging scripts and launchers together. Remove older `ciru_ornith_g256-*.dist-info/` directories once the 4.1.1 metadata is present. Restart the model after updating. The executable fingerprint creates a fresh compile cache automatically. Retain the complete previous bundle for rollback.

The public launcher keeps its 44 GiB shared-pool default. The qualified lower-memory Ciru profile uses 16 GiB:

```bash
bash bundle/serve.sh --cache-gib 16 --context 262144 --max-seqs 8
```

This reduces aggregate cached-history capacity while retaining a full 262,144-token request and native reasoning. Pools up to 16 GiB use 0.35 GPU utilization for startup admission; model, cache and transient allocations still require sufficient free memory. Omit short output limits for reasoning and agent work.

## Updating to Ciru v4 (4.0.0)

Stop the model process before updating. Download the complete plugin, native
libraries and launchers together; a plugin-only update is insufficient. Existing
target, DFlash2, vision and source-prefill weights and pinned engine wheels are
unchanged. Keep a copy of the previous bundle for rollback.

```bash
uvx --from huggingface_hub hf download \
  jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo \
  --include 'bundle/plugin-site/*' 'bundle/native/*' 'bundle/serve*.sh' \
    'bundle/packaging/*' README.md INSTALL.md RUNTIME-FIXES.md RELEASE.json \
    'evaluation/*' 'runtime/ciru-v4-ornith-source.tar.gz' \
  --local-dir .
```

Once `bundle/plugin-site/ciru_ornith_g256-4.0.0.dist-info/` is present, remove
older `ciru_ornith_g256-*.dist-info/` directories so only the 4.0.0 metadata
remains, then restart the usual launcher. Also update any separately copied
plugin/native directories. Existing `bundle/paths.env` overrides remain valid;
a fresh download defaults to `installed-runtime` and `bundle/models/`.

Strict automatic tool schemas, the original sampler, 262,144-token context,
eight active requests, 44 GiB shared cache, native vision and adaptive DFlash2
policy are preserved. Explicit function `strict: false` retains its opt-out.
For thinking/agent work, omit a short completion cap and leave room in the
shared prompt/output context. Rollback requires restoring the previous complete
plugin/native/launcher set and restarting, not mixing release files.

This release includes the target model, trained DFlash2 drafter, native kernels, custom vLLM plugin, and exact vLLM/AITER runtime wheels and source archives. **Use this runtime; stock `pip install vllm` does not provide the custom quantization or serving path.**

## Hardware and platform

- AMD Ryzen AI Max+ Strix Halo, gfx1151, with 128 GB unified memory.
- Linux x86-64 with a working AMD GPU driver, readable/writable `/dev/kfd` and render nodes.
- Budget roughly 100 GB available system memory for the loaded production profile. The recorded whole-host peak was 95.35 GB; other applications also use that memory.
- Allow at least 60 GB free disk for the 24.3 GB model assets, runtime installation and caches; source rebuilds need additional space.
- Validated platform: NixOS, Linux 7.2.2, glibc 2.42. A clean isolated runtime installation was tested on Strix Halo. The Ubuntu recipe below is provided for deployment and has not been independently validated on Ubuntu. Do not replace your system glibc to run this model.

## Download

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git, then:

```bash
uvx --from huggingface_hub hf download \
  jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo \
  --local-dir ./ciru-halo-agent
cd ciru-halo-agent
```

## Ubuntu 26.04 LTS prerequisites

Ubuntu 26.04 supplies [glibc 2.43](https://packages.ubuntu.com/resolute/libc6). This is the mainstream distro recipe; the validated host remains NixOS.

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake ninja-build pkg-config xxd \
  curl ca-certificates tar libnuma-dev libdrm-dev libelf-dev libssl-dev \
  zlib1g-dev libvulkan-dev
```

Ensure your user can access `/dev/kfd` and `/dev/dri/renderD*`; GPU access must work before launching. The installer obtains the pinned ROCm SDK and Python packages; a stock distro vLLM package is not needed.

```bash
bash runtime/INSTALL-ORNITH-RUNTIME.sh "$PWD/installed-runtime"
bash bundle/serve.sh --dry-run
bash bundle/serve.sh --host 127.0.0.1 --port 8000
```

The runtime installer refuses to overwrite an existing installation. `uv` must be on PATH. Installation downloads pinned dependencies from the AMD wheel index and Python package index.

## NixOS

Enable the standard dynamic loader with `programs.nix-ld.enable = true;` and working AMD GPU device access. The provided shell supplies the compiler and host libraries:

```bash
nix-shell runtime/shell.nix --run \
  'bash runtime/INSTALL-ORNITH-RUNTIME.sh "$PWD/installed-runtime"'
nix-shell runtime/shell.nix --run \
  'bash bundle/serve.sh --host 127.0.0.1 --port 8000'
```

## API and agent clients

The server exposes an OpenAI-compatible API at `http://127.0.0.1:8000/v1`, model ID **`ciru-halo-agent`**. Point Hermes or another compatible agent client to that URL. Use `--host 0.0.0.0` only when you intend to expose it on your network; the launcher provides no authentication by default.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ciru-halo-agent","messages":[{"role":"user","content":"Write a Python function that merges overlapping intervals."}],"temperature":0.6,"top_p":0.95,"max_tokens":4096,"chat_template_kwargs":{"enable_thinking":false}}'
```

The default profile supplies **262,144 tokens of per-request context capacity**, **eight active sequences**, **44 GiB shared KV/state pool**, prefix caching and adaptive speculation. Input and output share the context window; your agent client must reserve output space and compact history before filling it. Eight independent, fully populated 256K histories are not promised. The default server profile enables images. To explicitly select the image launcher, run `bash bundle/serve-vision.sh --host 127.0.0.1 --port 8000` after installation. The matching native BF16 vision encoder and projector are already included in `bundle/models/target/protected-00.safetensors`; no separate GGUF mmproj is required. This enables one image per request at a 1,048,576-pixel budget with up to eight active requests. See [the model card](README.md#vision--image-input) and [vision validation limits](VISION.md).

First startup compiles/loads GPU kernels and creates caches. Wait for `/health` before sending work. Keep `bundle/cache` writable. Change runtime/model locations with `ORNITH_RUNTIME_ROOT`, `ORNITH_MODEL`, and `ORNITH_DRAFT`. Ordinary users do not need to change quantization or draft-policy settings.

## Build from source

Use the [Ciru v4 source snapshot](runtime/ciru-v4-ornith-source.tar.gz) for the exact combined plugin, all eight native kernel sources and the build script. The [source repository](https://github.com/ciru-ai/ornith-ciru-halo-agent) retains the development history. See its `BUILD.md` for the native rebuild command. Runtime source archives are provided in this Hugging Face repository under `runtime/`; they include the matching vLLM/AITER source, licenses and release overlay notes. The binary installation above is the tested way to assemble the pinned engine; native source rebuilding is separate from retraining or requantizing the model.

Pinned runtime: vLLM `0.1.0rc2.dev9+g9255fd9fb9.rocm100` (base `9255fd9fb9fedf4b29d574a8d8bb21d93892cc98` plus supplied cache overlay), AITER `0.1.0rc1`, Python 3.14.3, PyTorch `2.13.0+rocm10.0.0`, ROCm SDK 10.0.0 and Transformers 5.16.1. Preserve included third-party licenses when redistributing.


### 4.1.3 upstream cache-mode repair

The optimized launcher derives its compiled-cache namespace from the fully resolved image configuration before starting vLLM. Vision ON and text-only OFF use separate directories under the executable build cache. Use the normal ON/OFF profile choices; do not copy compiled graphs between modes. Old cache directories and the preceding release remain available for rollback. This changes startup cache selection; weights, native libraries, pinned engine wheels, numerical helpers and draft policy are preserved. The validation report distinguishes retained numerical/performance evidence from fresh ON/OFF startup and uncapped smoke checks.
