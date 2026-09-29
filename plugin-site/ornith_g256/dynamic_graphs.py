"""C1 adaptive fallback: exact Q1/C1 target graph; retain Q16/Q8 keys.

Copy into the experimental ornith_g256 package and call install() before
GPUModelRunner construction. Installed vLLM files and drafter hooks stay intact.
"""
# Copyright 2026 Ciru.
from contextlib import contextmanager
from functools import wraps


_INSTALLED = False
_CONFIG_HOOK_INSTALLED = False


def _requested(config):
    return config.additional_config.get("ornith_g256", {}).get(
        "dynamic_spec_profile") is True


def _profile_errors(config):
    """Report actual values instead of silently opting an explicit profile out."""
    spec = config.speculative_config
    parallel = config.parallel_config
    cache = config.cache_config
    schedule = getattr(spec, "num_speculative_tokens_per_batch_size", None)
    values = {
        "speculative.method": (getattr(spec, "method", None), "dflash"),
        "speculative.num_speculative_tokens": (
            getattr(spec, "num_speculative_tokens", None), 15),
        "speculative.schedule": (
            tuple(tuple(row) for row in schedule) if schedule else None,
            ((1, 1, 15), (2, 8, 7))),
        "max_num_seqs": (config.scheduler_config.max_num_seqs, 8),
        "async_scheduling": (config.scheduler_config.async_scheduling, False),
        "quantization": (config.model_config.quantization, "ornith_g256"),
        "enable_prefix_caching": (cache.enable_prefix_caching, True),
        "lora_config": (config.lora_config, None),
        "use_v2_model_runner": (config.use_v2_model_runner, False),
        "use_ubatching": (parallel.use_ubatching, False),
    }
    values.update((field, (getattr(parallel, field), 1)) for field in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"))
    errors = [f"{name}={actual!r} (expected {expected!r})"
              for name, (actual, expected) in values.items() if actual != expected]
    # Retention changes CPU checkpoint eligibility, not page geometry or graph buffers.
    # This private repair profile permits frontier-only retention with the same DF15/7 graphs.
    if cache.prefix_cache_retention_interval not in (0, 1120):
        errors.append(f"prefix_cache_retention_interval={cache.prefix_cache_retention_interval!r} (expected 0 or 1120)")
    if not 65536 <= config.model_config.max_model_len <= 262144:
        errors.append("max_model_len outside 65536..262144")
    # EngineCore normalizes the shared config to the smallest participating
    # group before worker KV initialization: draft560, target/Mamba1120.
    # Physical block geometry remains unchanged for both supported retention policies.
    if cache.block_size not in (560, 1120):
        errors.append(f"cache.block_size={cache.block_size!r} (expected 560 or 1120)")
    return errors


def _reject(errors):
    from vllm.logger import init_logger

    message = "Dynamic target graph profile rejected: " + "; ".join(errors)
    init_logger("vllm.ornith_dynamic_graphs").error(message)
    raise ValueError(message)


def install_config_hook():
    """Call before AsyncEngineArgs builds its VllmConfig, and in the worker."""
    global _CONFIG_HOOK_INSTALLED
    if _CONFIG_HOOK_INSTALLED:
        return
    from vllm.config import CUDAGraphMode, VllmConfig

    original = VllmConfig._maybe_override_dynamic_sd_cudagraph_mode

    @wraps(original)
    def keep_supported_full(config):
        if not _requested(config):
            return original(config)
        # Only suppress the installed automatic FULL->PIECEWISE rewrite for
        # our exact two-shape target profile. Other modes retain their rules.
        if config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
            return original(config)
        errors = _profile_errors(config)
        sizes = config.compilation_config.cudagraph_capture_sizes
        if sizes != [16, 32, 48, 64]:
            errors.append(f"cudagraph_capture_sizes={sizes!r} (expected [16, 32, 48, 64])")
        if errors:
            _reject(errors)
        from vllm.logger import init_logger

        init_logger("vllm.ornith_dynamic_graphs").info(
            "Dynamic target config retains FULL_DECODE_ONLY for C1/Q16 and Q8/C1..8")

    VllmConfig._maybe_override_dynamic_sd_cudagraph_mode = keep_supported_full
    _CONFIG_HOOK_INSTALLED = True


def _enabled(runner):
    config = runner.vllm_config
    if not _requested(config):
        return False
    cached = getattr(runner, "_ornith_dynamic_graphs_enabled", None)
    if cached is not None:
        return cached
    errors = _profile_errors(config)
    if errors:
        _reject(errors)
    import torch

    props = torch.cuda.get_device_properties(runner.device)
    arch = getattr(props, "gcnArchName", "")
    if not arch.startswith("gfx1151"):
        _reject([f"gcnArchName={arch!r} (expected gfx1151)"])
    runner._ornith_dynamic_graphs_enabled = True
    return True


def _supported_query_len(num_tokens, num_reqs, max_query_len):
    if num_tokens != num_reqs * max_query_len:
        return None
    if max_query_len == 1 and num_reqs == 1:
        return 1
    if max_query_len == 16 and num_reqs == 1:
        return 16
    if max_query_len == 8 and 1 <= num_reqs <= 8:
        return 8
    return None


@contextmanager
def _query_len(runner, query_len):
    # This profile is synchronous TP1/PP1/DP1 without microbatching. Nested
    # capture -> dummy -> dispatch scopes restore their caller's values.
    dispatcher = runner.cudagraph_dispatcher
    runner_previous = runner.uniform_decode_query_len
    dispatcher_previous = dispatcher.uniform_decode_query_len
    runner.uniform_decode_query_len = query_len
    dispatcher.uniform_decode_query_len = query_len
    try:
        yield
    finally:
        runner.uniform_decode_query_len = runner_previous
        dispatcher.uniform_decode_query_len = dispatcher_previous


def _replace_full_keys(runner):
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    dispatcher = runner.cudagraph_dispatcher
    if dispatcher.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        _reject([f"resolved cudagraph_mode={dispatcher.cudagraph_mode.name}; "
                 "expected FULL_DECODE_ONLY (install_config_hook must run before config creation)"])
    # Reuse vLLM's resolved capture-size lookup, including its Q16 rounding.
    # Thus [16,32,48,64] yields Q8 captures at C2/C4/C6/C8; odd C pad up.
    lookup = dispatcher._bs_to_padded_graph_size
    if len(lookup) <= 64 or lookup[16] != 16 or lookup[64] != 64:
        raise ValueError("Dynamic target graphs require capture sizes 16 and 64")
    # Do not change global capture sizes: those also configure DFlash.
    # Add only an exact target Q1/C1 key after upstream's Q16 initialization.
    lookup[1] = 1
    lookup[8] = 8
    keys = {BatchDescriptor(num_tokens=1, num_reqs=1, uniform=True),
            BatchDescriptor(num_tokens=16, num_reqs=1, uniform=True)}
    for num_reqs in range(1, 9):
        padded = lookup[num_reqs * 8]
        if padded % 8 or not 8 <= padded <= 64:
            raise ValueError("Dynamic Q8 graph padding must stay within C1..8")
        keys.add(BatchDescriptor(
            num_tokens=padded, num_reqs=padded // 8, uniform=True))
    dispatcher.cudagraph_keys[CUDAGraphMode.FULL].clear()
    dispatcher.cudagraph_keys[CUDAGraphMode.FULL].update(keys)
    from vllm.logger import init_logger

    init_logger("vllm.ornith_dynamic_graphs").info(
        "Dynamic target FULL graph shapes (tokens, requests): %s; "
        "cache_config.block_size=%s, KV group block sizes=%s",
        sorted((key.num_tokens, key.num_reqs) for key in keys),
        runner.vllm_config.cache_config.block_size,
        [group.kv_cache_spec.block_size for group in runner.kv_cache_config.kv_cache_groups],
    )


def install():
    """Install process-local, config- and hardware-scoped target wrappers."""
    global _INSTALLED
    if _INSTALLED:
        return
    # The worker also creates replaced VllmConfigs while loading DFlash;
    # dataclasses.replace reruns __post_init__ and would downgrade them again.
    install_config_hook()
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_resolve = GPUModelRunner._check_and_update_cudagraph_mode
    original_dispatch = GPUModelRunner._determine_batch_execution_and_padding
    original_capture = GPUModelRunner._warmup_and_capture

    @wraps(original_resolve)
    def resolve(runner, *args, **kwargs):
        result = original_resolve(runner, *args, **kwargs)
        if _enabled(runner):
            _replace_full_keys(runner)
        return result

    @wraps(original_dispatch)
    def dispatch(runner, num_tokens, num_reqs, num_scheduled_tokens_np,
                 max_num_scheduled_tokens, *args, **kwargs):
        query_len = _supported_query_len(
            num_tokens, num_reqs, max_num_scheduled_tokens)
        if query_len == 1 and kwargs.get('force_uniform_decode') is None:
            # A one-token prompt/tail is not proof of decode. CPU metadata is
            # already available; no GPU read or synchronization is added.
            batch = runner.input_batch
            if not (batch.num_reqs == 1 and
                    batch.num_computed_tokens_cpu[0] >= batch.num_prompt_tokens[0]):
                query_len = None
        if query_len is None or not _enabled(runner):
            return original_dispatch(
                runner, num_tokens, num_reqs, num_scheduled_tokens_np,
                max_num_scheduled_tokens, *args, **kwargs)
        with _query_len(runner, query_len):
            result = original_dispatch(
                runner, num_tokens, num_reqs, num_scheduled_tokens_np,
                max_num_scheduled_tokens, *args, **kwargs)
        if kwargs.get('force_uniform_decode') is None:
            seen = getattr(runner, '_ornith_dynamic_dispatch_seen', set())
            key = (query_len, num_reqs, result[0].name)
            if key not in seen:
                from vllm.logger import init_logger
                init_logger("vllm.ornith_dynamic_graphs").info('Dynamic target dispatch Q%d/C%d: %s', *key)
                seen.add(key)
                runner._ornith_dynamic_dispatch_seen = seen
        return result

    @wraps(original_capture)
    def capture(runner, desc, *args, **kwargs):
        query_len = None
        if desc.uniform and desc.num_reqs:
            query_len = _supported_query_len(
                desc.num_tokens, desc.num_reqs, desc.num_tokens // desc.num_reqs)
        if query_len is None or not _enabled(runner):
            return original_capture(runner, desc, *args, **kwargs)
        if query_len == 1:
            # Only target Q1 is captured. The K0 diagnostic never uses the
            # drafter, whose trained convolution cannot warm up a one-row input.
            # Retain all its ordinary Q8/Q16 warmup and construction unchanged.
            drafter = runner.drafter
            had_override = 'dummy_run' in drafter.__dict__
            previous_dummy = drafter.__dict__.get('dummy_run')
            drafter.dummy_run = lambda *a, **k: None
            try:
                with _query_len(runner, query_len):
                    return original_capture(runner, desc, *args, **kwargs)
            finally:
                if had_override:
                    drafter.dummy_run = previous_dummy
                else:
                    del drafter.dummy_run
        with _query_len(runner, query_len):
            return original_capture(runner, desc, *args, **kwargs)

    GPUModelRunner._check_and_update_cudagraph_mode = resolve
    GPUModelRunner._determine_batch_execution_and_padding = dispatch
    GPUModelRunner._warmup_and_capture = capture
    _INSTALLED = True
