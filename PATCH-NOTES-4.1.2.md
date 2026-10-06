# Ciru v4.1.2 — October 5, 2026

Adds the verified synchronization barrier in the first DFlash convolution/residual/RMSNorm kernel used by `ornithtest`.

## Shared output handling

The pinned vLLM renderer reserved the entire requested output ceiling before rendering the prompt. With a 262,144-token ceiling, a nonempty prompt was rejected because no input budget remained. The correction lets the renderer validate the complete prompt, then leaves the existing sampler to use the remaining context. With 41 rendered prompt tokens, the effective output ceiling is 262,103. Omitted limits keep the existing remainder behavior; smaller explicit limits retain their semantics. Both model packages receive this correction.

## Model-specific scope

The first DFlash convolution/residual/RMSNorm kernel stored an FP32 residual and reloaded its scratch buffer without synchronization. Component tests and real inference captured stale pre-write values; poisoned scratch reproduced downstream invalid draft/head values. A single barrier between store and reload fixes the reproduced operation. This package uses the same barrier source and worker hook as the tested ornithtest profile. The hook verifies the original kernel AST and refuses an unrecognized implementation.

## Installation and clients

[Upgrade instructions](INSTALL.md#upgrading-to-412) include a checksum-verified updater for existing directories and a fresh pinned-runtime option for pre-v4 engines. Existing weights and compile caches are preserved, with a separate namespace for the updated executable bundle. The previous code is retained for rollback.

Vision was already enabled in both serving packages. OMP must also advertise image input in its model entry. A real OMP→Ciru `read` test identified a white background, green circle, purple triangle and orange square after correcting the client's text-only flag. Keep native reasoning enabled and the 262,144 context/output ceiling.

## Verification

The actual pinned renderer was compared before and after the correction in each package. Only the output reservation field changes, and only when the explicitly requested ceiling equals the full context. Both portable launchers pass `--inspect-only` at 262,144 context, eight slots and 44 GiB KV. The Ornith barrier files match the tested private profile by SHA256; its live worker intercepts the identified generated kernel, and graph-replay preflight covers rows 1, 8, 16, 24, 32, 40, 48, 56 and 64. [Patch validation](CIRU-V4.1.2-VALIDATION.json).

The patch changes no weights, pinned engine wheels or sampling defaults. Existing benchmark results retain their original build labels. This is a scoped repair, with no new speed or universal model-quality claim.

The recent improvements would not have been possible without **Cinderblade from the toolbox discord**.
