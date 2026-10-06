
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*i32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=20, cc='gfx1151', major=11, regs_per_multiprocessor=196608, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16], ['tt.pointer_range', 32]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16], ['tt.pointer_range', 32]], (3,): [['tt.divisibility', 16], ['tt.pointer_range', 32]], (4,): [['tt.divisibility', 16], ['tt.pointer_range', 32]], (5,): [['tt.divisibility', 16], ['tt.pointer_range', 32]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16], ['tt.pointer_range', 32]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 4861952}, 'kernel_num_gb': 0.00270746, 'kernel_flop': 0, 'backend_hash': 'F5627C075C721B9CDEB734769815D32C6AF2509F0681D2D2AB848A58ADE437E3', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'is_hip': True}
)
@triton.jit
def triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [1, 1])
    _tmp31 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (4096 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (512*x0 + (r0_1 // 16)), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr0 + (6144 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (128 + 512*x0 + (r0_1 // 16)), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = ((-1) + x0).to(tl.int32)
        tmp9 = tl.full([1, 1], 0, tl.int64)
        tmp10 = tmp8 >= tmp9
        tmp11 = tmp10.to(tl.int1)
        tmp12 = tl.load(in_ptr2 + ((-2048) + r0_1 + 2048*x0), r0_mask & tmp11 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tmp7 * tmp12
        tmp16 = tmp15.to(tl.int64)
        tmp17 = (x0).to(tl.int64)
        tmp18 = (tmp17).to(tl.int64)
        tmp19 = tmp18 & tmp16
        tmp20 = tl.full([1, 1], 1, tl.int64)
        tmp21 = tmp19 >= tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp13 * tmp22
        tmp24 = tmp4 + tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 + tmp27
        tmp29 = tmp28 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(r0_mask & xmask, tmp32, _tmp31)
        tl.store(out_ptr0 + (r0_1 + 2048*x0), tmp28, r0_mask & xmask)
    tl.debug_barrier()
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp33 = tl.load(out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp41 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tl.full([1, 1], 2048.0, tl.float32)
        tmp35 = (tmp31 / tmp34)
        tmp36 = tl.full([1, 1], 1e-06, tl.float32)
        tmp37 = tmp35 + tmp36
        tmp38 = tl.rsqrt(tmp37)
        tmp39 = tmp33 * tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tl.store(out_ptr2 + (r0_1 + 2048*x0), tmp42, r0_mask & xmask)
