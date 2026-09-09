"""Current-stream native launches with explicit scratch/output mutations."""
# Copyright 2026 Ciru.
import ctypes as C
from pathlib import Path
import torch

_libs = {}
_PHASE_SEGMENTS = ()


class DenseLayout(C.Structure):
    _fields_ = [(n, C.c_size_t) for n in
                ('workspace_bytes', 'transformed', 'low', 'high', 'scales', 'sums', 'projection')]


class RoutedLayout(C.Structure):
    _fields_ = [(n, C.c_size_t) for n in
                ('workspace_bytes', 'gate_x', 'gate_low', 'gate_high', 'gate_scales',
                 'gate_sums', 'gate_y', 'middle', 'down_x', 'down_low', 'down_high',
                 'down_scales', 'down_sums', 'route_out', 'final_f32')]


class HeadLayout(C.Structure):
    _fields_ = [(n, C.c_size_t) for n in
                ('workspace_bytes', 'transformed', 'activation_words', 'activation_scales', 'projection')]


def configure(settings, capacity, shapes):
    routed_activation_bits = settings.get('routed_activation_bits', 8)
    routed_prefill_activation_bits = settings.get('routed_prefill_activation_bits', 8)
    routed_decode_n32 = settings.get('routed_decode_n32', False)
    routed_storage_n32 = settings.get('routed_storage_n32', False)
    if routed_storage_n32 and routed_decode_n32:
        raise ValueError('Choose sole N32 storage or N32 decode shadows')
    if settings.get('routed_n32_max_rows', 8) not in (8, 16):
        raise ValueError('N32 routed decode crossover must be 8 or 16 rows')
    if routed_activation_bits not in (4, 8) or routed_prefill_activation_bits not in (4, 8):
        raise ValueError('G256 routed activation bits must be 4 or 8')
    if routed_activation_bits == routed_prefill_activation_bits == 4:
        raise ValueError('Choose either global routed A4 or prefill-only routed A4')
    if routed_decode_n32 and routed_activation_bits != 8:
        raise ValueError('N32 routed decode requires A8 verification')
    routed_suffix = ('_a4' if routed_activation_bits == 4 else
                     '_a4_prefill' if routed_prefill_activation_bits == 4 else '')
    signatures = {
        'dense': ('ornith_dense_g256', DenseLayout, [C.c_int]*4,
                  [C.c_void_p]*4+[C.c_size_t]+[C.c_void_p]*2+[C.c_int]*7+[C.c_void_p]),
        'routed': ('ornith_routed_direct', RoutedLayout, [C.c_int],
                   [C.c_void_p]*8+[C.c_size_t]+[C.c_void_p]*2+[C.c_int]*2+[C.c_void_p]),
        'head': ('ornith_head_i8_tile', HeadLayout, [C.c_int]*2,
                 [C.c_void_p]*4+[C.c_size_t]+[C.c_void_p]*2+[C.c_size_t]+[C.c_void_p]+[C.c_int]*4+[C.c_void_p]),
    }
    if routed_decode_n32:
        if not settings.get('routed_n32_library'):
            raise ValueError('routed_decode_n32 requires routed_n32_library')
        # The isolated N32 library retains the routed layout/launch C ABI.
        signatures['routed_n32'] = signatures['routed']
    for kind, (symbol, layout, query_args, launch_args) in signatures.items():
        path = str(Path(settings[kind+'_library']).resolve(strict=True))
        launch_name = symbol+'_launch'+(routed_suffix if kind == 'routed' else '')
        if kind in _libs:
            if _libs[kind][0] != path: raise RuntimeError("Cannot replace a native library in a live worker")
            if _libs[kind][3].__name__ != launch_name:
                raise RuntimeError('Cannot change activation precision in a live worker')
            continue
        lib = C.CDLL(path)
        if kind == 'routed' and routed_storage_n32:
            marker = lib.ornith_routed_storage_n32
            marker.argtypes, marker.restype = [], C.c_int
            if marker() != 32:
                raise ValueError('Sole N32 storage requires the full N32 consumer')
        query, launch = getattr(lib, symbol+'_get_layout'), getattr(lib, launch_name)
        query.argtypes, query.restype = query_args+[C.POINTER(layout)], C.c_int
        launch.argtypes, launch.restype = launch_args, C.c_int
        if kind == 'dense':
            prepare = lib.ornith_dense_g256_prepare_bf16
            prepare.argtypes = [C.c_void_p]*6+[C.c_int]*3+[C.c_void_p]
            prepare.restype = C.c_int
        _libs[kind] = (path, lib, query, launch)
    path, lib, query, launch = _libs['routed']
    for kind, symbol in [('routed_verify','ornith_routed_direct_launch'),
                         ('routed_prefill','ornith_routed_direct_launch_a4')]:
        fn=getattr(lib,symbol)
        fn.argtypes,fn.restype=launch.argtypes,launch.restype
        _libs[kind]=(path,lib,query,fn)
    sizes = []
    for n, k in shapes:
        layout = DenseLayout()
        check(_libs['dense'][2](capacity, n, k, 8, C.byref(layout)), 'dense layout')
        sizes.append(layout.workspace_bytes)
        sizes.append((capacity*k+n*k)*2)
    layout = RoutedLayout()
    check(_libs['routed'][2](capacity, C.byref(layout)), 'routed layout')
    sizes.append(layout.workspace_bytes)
    if routed_decode_n32:
        n32_layout = RoutedLayout()
        check(_libs['routed_n32'][2](capacity, C.byref(n32_layout)), 'N32 routed layout')
        if bytes(n32_layout) != bytes(layout):
            raise ValueError('N32 routed library must preserve the parent workspace layout')
    layout = HeadLayout()
    check(_libs['head'][2](64, 248320, C.byref(layout)), 'head layout')
    sizes.append(layout.workspace_bytes)
    if not sizes or min(sizes) <= 0: raise RuntimeError("Native workspace query failed")
    return max(sizes)


def ptr(t): return C.c_void_p(t.data_ptr())
def stream(x): return C.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
def check(status, operation):
    if status: raise RuntimeError(f"{operation} failed with native status {status}")


def validate(x, out, capacity, k):
    if (x.ndim != 2 or x.shape[1] != k or not 0 <= x.shape[0] <= capacity
            or x.dtype != torch.bfloat16 or x.device.type != 'cuda'
            or out.dtype != x.dtype or out.device != x.device
            or not x.is_contiguous() or not out.is_contiguous()):
        raise ValueError("Native G256 requires contiguous BF16 input/output within capacity")


@torch.library.custom_op('ornith_g256::dense_out',
                        mutates_args={'workspace', 'out', 'flags'}, device_types='cuda')
def dense_out(x: torch.Tensor, codes: torch.Tensor, metadata: torch.Tensor,
              workspace: torch.Tensor, out: torch.Tensor, flags: torch.Tensor,
              capacity: int, n: int, k: int, geometry: int, a8_max_rows: int) -> None:
    if _PHASE_SEGMENTS and x.shape[0] >= _PHASE_SEGMENTS[-1][1]:
        for start,end,prefill in _PHASE_SEGMENTS:
            _dense_impl(x[start:end],codes,metadata,workspace,out[start:end],flags,
                        capacity,n,k,geometry,0 if prefill else capacity)
        end=_PHASE_SEGMENTS[-1][1]
        if end<x.shape[0]:
            _dense_impl(x[end:],codes,metadata,workspace,out[end:],flags,
                        capacity,n,k,geometry,0)
        return
    _dense_impl(x,codes,metadata,workspace,out,flags,capacity,n,k,geometry,a8_max_rows)


def _dense_impl(x,codes,metadata,workspace,out,flags,capacity,n,k,geometry,a8_max_rows):
    validate(x, out, capacity, k)
    if x.shape[0] > a8_max_rows:
        # Reuse the arena for ephemeral BF16 transformed X and dequantized W.
        # This A16 prefill path is intentionally distinct from A8 decode.
        weight_offset = capacity*k*2
        transformed = workspace[:x.shape[0]*k*2].view(torch.bfloat16).view(x.shape[0], k)
        weight = workspace[weight_offset:weight_offset+n*k*2].view(torch.bfloat16).view(n, k)
        check(_libs['dense'][1].ornith_dense_g256_prepare_bf16(
            ptr(x), ptr(codes), ptr(metadata), ptr(transformed), ptr(weight), ptr(flags),
            x.shape[0], n, k, stream(x)), 'G256 BF16 prefill preparation')
        torch.mm(transformed, weight.t(), out=out)
        return
    check(_libs['dense'][3](ptr(x), ptr(codes), ptr(metadata), ptr(workspace), workspace.numel(),
                           ptr(out), ptr(flags), x.shape[0], capacity, n, k, 8, 128, geometry,
                           stream(x)), 'G256 dense')


@torch.library.custom_op('ornith_g256::routed_out',
                        mutates_args={'workspace', 'out', 'flags'}, device_types='cuda')
def routed_out(x: torch.Tensor, routes: torch.Tensor, ids: torch.Tensor,
               gate: torch.Tensor, gate_meta: torch.Tensor, down: torch.Tensor, down_meta: torch.Tensor,
               workspace: torch.Tensor, out: torch.Tensor, flags: torch.Tensor, capacity: int) -> None:
    if _PHASE_SEGMENTS and x.shape[0] >= _PHASE_SEGMENTS[-1][1]:
        segments=list(_PHASE_SEGMENTS)
        if segments[-1][1]<x.shape[0]:segments.append((segments[-1][1],x.shape[0],True))
        for start,end,prefill in segments:
            _routed_out('routed_prefill' if prefill else 'routed_verify',
                        x[start:end],routes[start:end],ids[start:end],
                        gate,gate_meta,down,down_meta,workspace,out[start:end],flags,capacity)
        return
    _routed_out('routed', x, routes, ids, gate, gate_meta, down, down_meta,
                workspace, out, flags, capacity)


def _routed_out(kind, x, routes, ids, gate, gate_meta, down, down_meta,
                workspace, out, flags, capacity):
    validate(x, out, capacity, 2048)
    if (routes.dtype != torch.float32 or ids.dtype != torch.int32
            or routes.shape != (x.shape[0], 8) or ids.shape != routes.shape
            or routes.device != x.device or ids.device != x.device
            or not routes.is_contiguous() or not ids.is_contiguous()):
        raise ValueError("G256 routed input requires FP32 weights and INT32 IDs [T,8]")
    check(_libs[kind][3](*[ptr(t) for t in
                             (x, routes, ids, gate, gate_meta, down, down_meta, workspace)],
                            workspace.numel(), ptr(out), ptr(flags), x.shape[0], capacity, stream(x)),
          'G256 '+kind)


@torch.library.custom_op('ornith_g256::routed_out_n32',
                        mutates_args={'workspace', 'out', 'flags'}, device_types='cuda')
def routed_out_n32(x: torch.Tensor, routes: torch.Tensor, ids: torch.Tensor,
                   gate: torch.Tensor, gate_meta: torch.Tensor,
                   down: torch.Tensor, down_meta: torch.Tensor,
                   gate_n32: torch.Tensor, gate_meta_n32: torch.Tensor,
                   down_n32: torch.Tensor, down_meta_n32: torch.Tensor,
                   workspace: torch.Tensor, out: torch.Tensor,
                   flags: torch.Tensor, capacity: int, n32_max_rows: int = 8) -> None:
    # Shape dispatch stays inside the opaque op; capture records the selected
    # launch with persistent banks and the caller's existing scratch/stream.
    if x.shape[0] <= n32_max_rows:
        kind, bank = 'routed_n32', (gate_n32, gate_meta_n32, down_n32, down_meta_n32)
    else:
        kind, bank = 'routed', (gate, gate_meta, down, down_meta)
    _routed_out(kind, x, routes, ids, *bank, workspace, out, flags, capacity)


@torch.library.custom_op('ornith_g256::head_out',
                        mutates_args={'workspace', 'out', 'flags'}, device_types='cuda')
def head_out(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
             workspace: torch.Tensor, out: torch.Tensor, flags: torch.Tensor, geometry: int) -> None:
    validate(x, out, 64, 2048)
    check(_libs['head'][3](ptr(x), ptr(codes), ptr(scales), ptr(workspace), workspace.numel(),
                          ptr(out), None, 0, ptr(flags), x.shape[0], 64, 248320, geometry, stream(x)),
          'W8 head')


@dense_out.register_fake
def _dense_fake(x, codes, metadata, workspace, out, flags, capacity, n, k, geometry, a8_max_rows): return None
@routed_out.register_fake
def _routed_fake(x, routes, ids, gate, gate_meta, down, down_meta, workspace, out, flags, capacity): return None
@routed_out_n32.register_fake
def _routed_n32_fake(x, routes, ids, gate, gate_meta, down, down_meta,
                     gate_n32, gate_meta_n32, down_n32, down_meta_n32,
                     workspace, out, flags, capacity, n32_max_rows=8): return None
@head_out.register_fake
def _head_fake(x, codes, scales, workspace, out, flags, geometry): return None
