# Native tools: maintained 1.0.3 integration

This source integrates qualified R04 XGrammar and R03 continuous streaming usage.
For the user's complete model, compose it with the **complete promoted model
bundle** using `scripts/assemble-unified.py`. Installing this source checkout's
base plugin over that bundle would discard its newer model optimizations.

## Source boundaries

- XGrammar 0.2.3, commit `557becfb64c503ae9c04344b0047661f43f44320`,
  with `patches/xgrammar/r04.patch`. Source digests are in
  `patches/xgrammar/SOURCE.json`; exact compiler commands and output digests are
  written into each native build receipt. No speculative state machine rewrite.
- R01 is unchanged, SHA256
  `e9c3608438712521980d7fcd60f58862d0b0a2876620183f304c64a03817edea`.
  A parent that already registers it keeps its existing location and initializer.
- R03 uses the owner's exact qualified chat source, SHA256
  `5720371f5bcb351e340a3e8d9a0235ec01ff27ed93e4f35542e31b83a055ef45`.
  Ciru's receiving source differs from the owner's baseline by two finish-reason
  guards. `patches/receiving_stream_usage.py` reproduces the qualified source
  from that exact receiving hash, preserving `length` instead of reporting a
  truncated output as a successful tool call. Both baseline files are retained
  under tests. The original R03 transformation remains unchanged.
- The four existing vLLM correctness overlays and strict parser restrictions
  remain unchanged. The new chat overlay is a fifth manifest entry. Installed
  vLLM and XGrammar files are never overwritten.

## Build and compose

Activate the pinned runtime environment before importing its Python libraries.
On Ciru, use `/opt/ciru/glm53-iu4/runtime-env.sh` with `VLLM_SOURCE`, `VLLM_VENV`
and `AITER_SOURCE` pointing to that installation. Use CMake, Ninja and GCC with
the pinned Python. On NixOS enable `nix-command flakes` for `nix shell`.

```bash
python scripts/build-xgrammar.py --python "$VLLM_VENV/bin/python" \
  --out /new/path/native-build --control --jobs 2
python scripts/assemble-unified.py --parent /path/to/complete-promoted/bundle \
  --native-build /new/path/native-build --out /new/path/unified/bundle
```

Assembly requires the reviewed promoted parent's BUILD digest, verifies every
parent file, preserves its distribution/entry-point metadata, and privately
copies mutable compilation caches. It adds the native payload and streaming
overlay, changing only the parent initializer and correctness manifest. Weights,
optimized modules, native model kernels and launch settings stay intact.
Each output has a BUILD manifest and TOOL-INTEGRATION receipt. A successor from
the numerical/kernel task needs source review and an explicit updated parent
identity before assembly. An assembled artifact is not a completed model.

## Qualification

```bash
python tests/native_tools/test_packaging.py
python tests/native_tools/upstream/test_stream_usage.py
python tests/native_tools/run_receiving.py --native-build /new/path/native-build \
  --model /path/to/target --out /new/path/native-checks
python tests/native_tools/check_bundle.py --bundle /new/path/unified/bundle \
  --out /new/path/import-check.json
python tests/native_tools/run_r01.py --bundle /new/path/unified/bundle \
  --out /new/path/r01-checks
```

Native equivalence checks cover 122 cases and 461 masks per build, including
bounded/unbounded grammar cases, Unicode, strict enums, and DF4/DF7/DF15
rollback/commit. R01 checks cover real DF7/DF15 windows at C1/C2/C8, including
uneven drafts. Import checks hash the package and assert the actual native
library mapping. Some runtime imports open `/dev/kfd` even without loading a
model; coordinate these checks with the shared GPU reservation.

Run `tests/native_tools/live_tools.py` against an isolated instance of the
assembled bundle with the original model settings. Require exact, complete
arguments and strict framing errors; save all raw responses/SSE. This is an
explicit non-thinking tool replay, not a reasoning/agent quality benchmark.
One-token preparation requests are intentionally truncated and earn no quality
credit. Full model/agent evaluations use the user's full reasoning allowance.
Check `/proc/<pid>/maps` for the selected R04 library in every relevant API and
engine process. A Python override by itself does not prove the library selection.

The owner's measured tool gains apply to its stated workloads. Receiving
compatibility does not establish a receiving or combined-model speed gain.
R04 increases cold grammar compilation time; record preparation separately from
cached generation. R03 reports progress accurately and does not change tokens.

## Packaging

The unified bundle is the complete runtime artifact. A component development
wheel can be built only after staging `_xgrammar_native/` into this checkout's
`plugin-site/ornith_g256/`. `setup.py` refuses a wheel without the native payload
and emits a platform-specific tag, not an `any` tag. That component wheel is
not a replacement for the optimized bundle. Preserve the bundle's native model
libraries, weights, optimized plugin modules and entry point during installation.

## Rollback

Keep the complete prior bundle immutable. Stop only the instance launched for
this integration, then launch the prior bundle with its original command and
settings. R03's source finder and R04's loaded native library are process-local:
a full process restart is required. Do not overwrite a loaded `.so`, edit the
installed XGrammar wheel, or try to switch selection inside a running process.
For qualification, the shared service guard restores the original model service,
active state and unit link. Production selector/profile changes and publication
are separate from assembling this artifact.

Completion requires the numerical/kernel task's final dispositions, all promoted
changes in the same manifest, receiving checks on that exact artifact, and the
remaining final-model comparisons. Existing combined-v3 results retain their
original build identity; do not silently relabel them as unified-v4 results.
