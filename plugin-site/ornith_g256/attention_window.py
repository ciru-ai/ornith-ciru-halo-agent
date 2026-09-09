# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ciru. Derived at runtime from the installed vLLM kernel.
"""Skip empty query tiles and context outside the drafter's sliding window.

Masking probabilities alone is insufficient when an evicted cache entry points
to the null page: zero times a nonfinite V still contaminates the accumulator.
The remaining per-row attention mask preserves the installed window semantics.
"""
import inspect
import linecache


def _skip_empty_queries(source):
    marker = '    block_start_loc = BLOCK_M * start_m\n'
    if source.count(marker) != 1:
        raise RuntimeError('Unsupported installed prefix kernel: query tile structure')
    return source.replace(marker, marker +
        '    if block_start_loc >= cur_batch_query_len:\n'
        '        return\n')


def _compile_kernel(prefix_prefill, source, variant):
    # Triton obtains source through inspect; keep generated sources available
    # under distinct project-owned filenames without modifying installed vLLM.
    filename = __file__ + '.' + variant + '.generated'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(vars(prefix_prefill))
    exec(compile(source, filename, 'exec'), namespace)
    return namespace['_fwd_kernel']


def build_query_kernel(prefix_prefill):
    """Preserve full attention arithmetic, omitting tiles with no output rows."""
    source = _skip_empty_queries(inspect.getsource(prefix_prefill._fwd_kernel.fn))
    return _compile_kernel(prefix_prefill, source, 'query')


def build_window_kernel(prefix_prefill):
    source = _skip_empty_queries(inspect.getsource(prefix_prefill._fwd_kernel.fn))
    old_loop = '''    # compute query against context (no causal mask here)
    for start_n in tl.range(
        0, cur_batch_ctx_len, BLOCK_SIZE, loop_unroll_factor=num_unroll_cache
    ):'''
    new_loop = '''    # No row in this query tile can attend earlier context. Align the loop
    # to the cache tile and mask its boundary loads as well as probabilities.
    first_context_token = 0
    if SLIDING_WINDOW > 0:
        first_context_token = tl.maximum(
            0, cur_batch_ctx_len + block_start_loc - SLIDING_WINDOW + 1
        )
    first_context_tile = (first_context_token // BLOCK_SIZE) * BLOCK_SIZE
    for start_n in tl.range(
        first_context_tile, cur_batch_ctx_len, BLOCK_SIZE,
        loop_unroll_factor=num_unroll_cache
    ):'''
    replacements = [(old_loop, new_loop)]
    # These clauses occur once each for K and V in the context loop only.
    old_condition = '''            start_n + BLOCK_SIZE > cur_batch_ctx_len
            or BLOCK_DMODEL != BLOCK_DMODEL_PADDED'''
    if source.count(old_condition) != 2:
        raise RuntimeError('Unsupported installed prefix kernel: context load conditions')
    source = source.replace(old_condition, '''            start_n < first_context_token
            or start_n + BLOCK_SIZE > cur_batch_ctx_len
            or BLOCK_DMODEL != BLOCK_DMODEL_PADDED''')
    for indices in ('offs_bs_n[None, :]', 'offs_bs_n[:, None]'):
        old = f'& ((start_n + {indices}) < cur_batch_ctx_len),'
        new = (f'& ((start_n + {indices}) < cur_batch_ctx_len)\n'
               f'                & ((start_n + {indices}) >= first_context_token),')
        replacements.append((old, new))
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('Unsupported installed prefix kernel: context window structure')
        source = source.replace(old, new)
    return _compile_kernel(prefix_prefill, source, 'window')
