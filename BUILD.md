# Build Ornith1.5 Ciru Halo Agent

The supplied native libraries and the source here belong to the measured AMD
Strix Halo (`gfx1151`) release. The model uses its accompanying vLLM runtime and
plugin; this is not a generic drop-in GPTQ/AWQ checkpoint.

## Prerequisites

Use the release runtime installer first. Its pinned runtime contains Python
3.14, ROCm 10, PyTorch, Triton, vLLM and AITER. A C++20 host development toolchain
and ROCm HIP compiler with `gfx1151` support are required for native compilation.
The build uses CPU compilation; it does not start a model or submit inference.
Use the release launch instructions for inference after building.

## Build the eight native libraries

```bash
git clone https://github.com/ciru-ai/ornith-ciru-halo-agent.git
cd ornith-ciru-halo-agent

# Set this to the directory produced by the release runtime installer.
RUNTIME="$HOME/ornith-runtime"

# Optional fast dependency/source check, without compilation:
bash scripts/build-native.sh --runtime "$RUNTIME" --out "$PWD/build/native" --check

# Compile all eight libraries into a new directory:
bash scripts/build-native.sh --runtime "$RUNTIME" --out "$PWD/build/native"
```

The script discovers HIP from the runtime's ROCm SDK wheels. `ROCM_PATH` and
`HIPCC` can select an equivalent standalone ROCm SDK. Additional host-toolchain
arguments can be supplied after `--`; for example, Clang's `--gcc-toolchain`
option when the GCC installation is outside the system compiler search paths.
NixOS users should enter the release development shell before compiling.
Do not add fast-math flags: the rounding and contraction settings are deliberate.

Compilation preserves `-O2 -ffp-contract=off` and
`-fhip-fp32-correctly-rounded-divide-sqrt`, targets `gfx1151`, and emits shared
PIC libraries. Host-specific Nix store paths and experimental directory RPATHs
are not built into this portable recipe. Use the installed runtime environment
when loading the output so its HIP and C++ libraries can be resolved.

| Source | Output | Role |
| --- | --- | --- |
| `kernels/dense/dense_g256_consumer.cpp` | `libornith_dense_g256.so` | Dense transformed projections |
| `kernels/dense_n32/dense_g256_consumer.cpp` | `libornith_dense_g256_n32.so` | Single-token dense projection path |
| `kernels/routed/routed_direct_consumer.cpp` | `libornith_routed_direct.so` | Routed expert decode and A4 prefill |
| `kernels/routed_n32/routed_n32_consumer.cpp` | `libornith_routed_n32.so` | Routed expert N32 dispatch |
| `kernels/routed_storage/routed_storage_n32_consumer.cpp` | `libornith_routed_storage_n32.so` | Sparse routed expert storage path |
| `kernels/head/head_i8_tile_consumer.cpp` | `libornith_head_i8_tile.so` | Quantized output head |
| `kernels/attention/prefill.cpp` | `libornith_attention_iu4.so` | Long-context IU4 attention prefill |
| `kernels/attention/persistent.cpp` | `libornith_persistent_iu4.so` | Persistent long-context IU4 attention |

Keep component header directories separate; similarly named headers belong to
different measured implementations. A library build records the compiler version
and the exact command for every output. Existing output libraries are never
overwritten. To try rebuilt libraries, use a separate copy of the downloaded
release bundle and place the eight `.so` files in its `native/` directory.
Keep the distributed libraries available until the rebuilt bundle passes the
same startup and workload checks on your host.

## Python plugin

`plugin-site/ornith_g256/` is the current plugin, including adaptive speculation
and the current single-token dense kernel binding. Module and library names are
stable runtime interfaces, not alternate public model names.

The release bundle supplies plugin metadata and loads this directory directly.
For development, a wheel can also be built without importing the GPU runtime:

```bash
uv build --python "$RUNTIME/venv/bin/python" --wheel --out-dir dist
```

The plugin intentionally does not install or upgrade its runtime dependencies.
Use the pinned release runtime; a stock vLLM upgrade is not a compatible-runtime
promise. When editing plugin code, retain the release bundle layout because the
dense kernel binding resolves `native/` relative to `plugin-site/`.

## Reproduction scope

`SOURCE-PROVENANCE.json` records the current source files and the corresponding
measured native library digests. The sources were recovered from the compilation
locations recorded in those binaries and the retained build commands. The
portable build recipe does not claim bit-for-bit equality across compilers,
operating systems or linker versions. External-machine performance and startup
must be checked on that machine; the published benchmark results describe the
recorded release runtime.

## Rebuild vLLM and AITER from the supplied sources

The download also includes `runtime/ciru-halo-agent-vllm-source.tar.gz` and
`runtime/ciru-halo-agent-aiter-source.tar.gz`. These contain the release source
and overlays, with upstream licenses; use these archives rather than a stock
upstream checkout. In particular, preserve the vLLM cache overlay documented in
`DEV9_CACHE_PATCH.md` within the source package. The archive's older embedded
installation notes describe its original application, not this model's profile.

The following developer route builds replacement wheels using the installed
release runtime's pinned PyTorch/ROCm. Run it in a separate runtime installation,
not one currently serving requests. Engine wheel rebuilding is more expensive
than compiling the eight model libraries and has **not been rerun for this
publication**. The published runtime uses the retained, measured engine wheels.

```bash
# BUNDLE is the downloaded model bundle; RUNTIME is a separate installed runtime.
BUNDLE="$HOME/Ornith1.5-Ciru-Halo-Agent"
RUNTIME="$HOME/ornith-runtime-build"
mkdir -p engine-source/vllm engine-source/aiter engine-wheels

tar -xzf "$BUNDLE/runtime/ciru-halo-agent-vllm-source.tar.gz" \
  -C engine-source/vllm --strip-components=1
tar -xzf "$BUNDLE/runtime/ciru-halo-agent-aiter-source.tar.gz" \
  -C engine-source/aiter --strip-components=1

export VLLM_VENV="$RUNTIME/venv"
export VLLM_SOURCE="$PWD/engine-source/vllm"
export AITER_SOURCE="$PWD/engine-source/aiter"
source "$RUNTIME/runtime-env.sh"
export PYTORCH_ROCM_ARCH=gfx1151 GPU_ARCHS=gfx1151 VLLM_TARGET_DEVICE=rocm
export BUILD_TARGET=rocm ENABLE_CK=1
export MAX_JOBS=8

# Build tools only: do not install the upstream ROCm requirement file, which
# contains PyTorch/ROCm pins different from this release.
uv pip install --python "$VLLM_VENV/bin/python" \
  pip 'setuptools>=77.0.3,<80' 'setuptools-scm>=8' 'setuptools-rust>=1.9' \
  wheel 'cmake>=3.26.1,<4' ninja packaging pybind11 psutil pandas jinja2

# The source archives already include required vendored submodule content.
# No git submodule command is needed for these archives.
"$VLLM_VENV/bin/python" -m pip wheel --no-build-isolation --no-deps \
  "$AITER_SOURCE" -w "$PWD/engine-wheels"
"$VLLM_VENV/bin/python" -m pip wheel --no-build-isolation --no-deps \
  "$VLLM_SOURCE" -w "$PWD/engine-wheels"
```

NixOS builds also require the supplied development shell and the source's CMake
integration helpers before the wheel commands:

```bash
export CMAKE_ARGS="${CMAKE_ARGS:-} \
-DHIP_HIPCC_CMAKE_LINKER_HELPER=$VLLM_SOURCE/ciru-release/nixos/hipcc_cmake_linker_helper \
-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=$VLLM_SOURCE/ciru-release/nixos/rocm10_numa_target.cmake"
```

Install rebuilt wheels only into an isolated runtime and check model startup,
tool calls, prefix-cache reuse and your context/concurrency profile before
switching to them. Engine build dependencies and system linkers vary by host;
this source route is provided for development, not an assertion that a full
engine rebuild has been qualified on every Linux distribution.

## Release source checks

All eight native libraries were successfully compiled from this source tree with
the clean installed release runtime on the development Strix Halo host. The
compiler was AMD clang 23 / ROCm 10. The check exercised the portable script,
including automatic discovery of NixOS GCC/glibc paths. It did not run inference
with the newly compiled libraries or replace any running model's libraries.
All plugin Python files passed syntax parsing. Engine wheel rebuilding was not
repeated; the shipped engine wheels retain their existing runtime evidence.
