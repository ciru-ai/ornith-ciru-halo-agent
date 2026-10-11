"""Restore actual pre-sample generator offsets for discarded IU4 target rows.

The pinned v4.1.2 synchronous GPU runner subtracts a fixed four in bookkeeping.
Speculative sampling can advance by eight; snapshotting the actual starting
offset handles both paths without altering probabilities, noise or samples.
This adapter is deliberately scoped to synchronous scheduling.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import MethodType


def install(runner, output=None):
    """Install on the actual runner after its sampler/bookkeeping are ready."""
    audit_path = None if output is None else Path(output)
    if getattr(runner, "_ornith_exact_discard_rng_restored", False):
        runner._ornith_discard_rng_audit_path = audit_path
        return
    if runner.use_async_scheduling:
        raise RuntimeError("Exact discard RNG restoration requires synchronous scheduling")

    sample_original = runner._sample
    bookkeeping_original = runner._bookkeeping_sync
    pending = []
    runner._ornith_discard_rng_audit_path = audit_path

    def sample(instance, *args, **kwargs):
        if pending:
            raise RuntimeError("Previous discarded-row RNG checkpoint was not consumed")
        batch = instance.input_batch
        for index in range(batch.num_reqs):
            if instance.discard_request_mask.np[index]:
                generator = batch.generators.get(index)
                if generator is not None:
                    pending.append((batch.req_ids[index], generator, generator.get_offset()))
        try:
            return sample_original(*args, **kwargs)
        except BaseException:
            # A failed sample has no output to commit. Preserve seeded request
            # state and avoid carrying a stale checkpoint into an error retry.
            for _, generator, before in pending:
                generator.set_offset(before)
            pending.clear()
            raise

    def bookkeeping(instance, *args, **kwargs):
        value = bookkeeping_original(*args, **kwargs)
        if pending:
            rows = []
            for request_id, generator, before in pending:
                upstream_after = generator.get_offset()
                generator.set_offset(before)
                rows.append({
                    "req_id": request_id,
                    "seed": generator.initial_seed(),
                    "before_sample": before,
                    "after_upstream_bookkeeping": upstream_after,
                    "after_exact_restore": generator.get_offset(),
                })
            pending.clear()
            audit = instance._ornith_discard_rng_audit_path
            if audit is not None:
                with audit.open("a") as stream:
                    stream.write(json.dumps({
                        "scope": "Actual discarded seeded target rows; exact offset restore",
                        "rows": rows,
                    }) + "\n")
        return value

    runner._sample = MethodType(sample, runner)
    runner._bookkeeping_sync = MethodType(bookkeeping, runner)
    runner._ornith_exact_discard_rng_restored = True

