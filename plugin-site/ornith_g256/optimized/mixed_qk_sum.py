import triton
import triton.language as tl

@triton.jit
def mixed_qk_sum(in_ptr0, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr, K_XBLOCK: tl.constexpr, K_RBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, K_XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = xindex % 16
        x1 = xindex // 16
        _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages=2):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + 512 * x0 + 9216 * x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tmp1 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tl.store(out_ptr0 + x3, tmp4, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = K_RBLOCK
        xoffset = pid_offset * K_XBLOCK
        xindex = xoffset + tl.arange(0, K_XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, K_RBLOCK)[None, :]
        rbase = r0_base
        x4 = xindex % 2
        x5 = xindex // 2
        _tmp10 = tl.full([K_XBLOCK, K_RBLOCK], 0, tl.float32)
        x7 = xindex
        for r0_offset in tl.range(0, r0_numel, K_RBLOCK, num_stages=2):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp6 = tl.load(in_ptr0 + (8192 + r0_6 + 256 * x4 + 9216 * x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tmp7 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [K_XBLOCK, K_RBLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tl.store(out_ptr1 + x7, tmp10, xmask)
    else:
        pass
