"""Isolated mixed-target dispatch using existing kernels and CPU phase bounds.

The backend writes current K/V once before this helper. This helper only creates
views and rebases query starts on the device; it neither changes cache ownership
nor reads GPU metadata on the host. Not installed in any measured bundle.
"""


_diagnostic_signatures = set()

def try_forward(dispatch, query, key, value, output, kv_cache_dtype,
                key_cache, value_cache, block_table, query_start_loc, seq_lens,
                max_seq_len, max_query_len, k_scale, v_scale,
                alibi_slopes, sliding_window, sm_scale, output_scale, sinks,
                is_block_table_ptr, causal):
    from . import native
    requests = getattr(native, '_ATTENTION_PHASE_REQUESTS', ())
    # Subcalls have fewer requests, preventing recursion without global state.
    # Dummy capture, draft attention, and unmatched metadata use the old route.
    n = len(requests)
    if (n < 2 or n != seq_lens.numel() or query_start_loc.numel() != n + 1
            or block_table.shape[0] < n or output.shape != query.shape
            or requests[0][0] != 0 or requests[-1][1] > query.shape[0]
            or max(end-start for start, end, _, _ in requests) != max_query_len
            or not all(end > start for start, end, _, _ in requests)
            or not any(r[2] for r in requests) or all(r[2] for r in requests)):
        return False
    groups = []
    for req, (_, _, prefill, _) in enumerate(requests):
        if not prefill and groups and not groups[-1][2]:
            first, _, phase = groups[-1]
            groups[-1] = (first, req + 1, phase)
        else:
            # Keep each prefill C1, enabling its existing Q1120 IU4 path.
            groups.append((req, req + 1, prefill))
    # Folded decode keeps its current <=128-row contract. Do not partially
    # dispatch before deciding whether the entire mixed call is supported.
    if any(not phase and requests[last-1][1]-requests[first][0] > 128
           for first, last, phase in groups):
        return False
    logical_end = requests[-1][1]
    if logical_end < output.shape[0]:
        output[logical_end:].zero_()
    diagnostic = []
    for first, last, prefill in groups:
        start, end = requests[first][0], requests[last-1][1]
        # GPU subtraction rebases cu_q to the sliced query, without .item(),
        # .cpu(), or tensor reconstruction from GPU values.
        starts = query_start_loc[first:last+1] - start
        max_q = max(r[1]-r[0] for r in requests[first:last])
        max_k = max(r[3] for r in requests[first:last])
        dispatch(query[start:end], None if key is None else key[start:end],
                 None if value is None else value[start:end], output[start:end],
                 kv_cache_dtype, key_cache, value_cache,
                 block_table[first:last], starts, seq_lens[first:last],
                 max_k, max_q, k_scale, v_scale, alibi_slopes, sliding_window,
                 sm_scale, output_scale, sinks, is_block_table_ptr, causal)
        if len(_diagnostic_signatures) < 4:
            from . import attention_compact, attention_iu4_persistent
            iu4_prefill = (prefill and last-first == 1 and max_q == 1120
                           and max_k >= 4096 and max_k % 32 == 0
                           and attention_compact._arena['iu4'] is not None)
            diagnostic.append({'rows': (start, end), 'prefill': prefill,
                'max_q': max_q, 'max_k': max_k,
                'iu4_prefill_guard': iu4_prefill,
                'persistent_decode_installed': attention_iu4_persistent._installed})
    # Successful subdispatch receipts only; CPU metadata, no GPU read or sync.
    # At most four distinct query/path signatures, so repeated layers stay quiet.
    if diagnostic:
        signature = tuple((d['max_q'], d['prefill'], d['iu4_prefill_guard'])
                          for d in diagnostic)
        if signature not in _diagnostic_signatures:
            _diagnostic_signatures.add(signature)
            print('ORNITH_MIXED_ATTENTION_SUCCESS ' + str(diagnostic), flush=True)
    return True
