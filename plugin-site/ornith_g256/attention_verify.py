"""Optional page1104 multi-query verification, with prebound GPU metadata."""
# Copyright 2026 Ciru.
import ctypes
from types import SimpleNamespace

import torch
from .attention_fast import PagedLayout, PagedStrides


class NativeVerifyFP32:
    page_size = 1104

    def __init__(self, library, *, Ccap, Lcap, device):
        if not 1 <= Ccap <= 64 or not 1 <= Lcap <= 8192:
            raise ValueError('Verify attention supports <=64 queries, context <=8192')
        self.Ccap, self.Lcap, self.device = Ccap, Lcap, torch.device(device)
        self.library = str(library)
        self.lib = ctypes.CDLL(self.library)
        self.lib.ornith_verify_paged_abi_version.restype = ctypes.c_uint32
        if self.lib.ornith_verify_paged_abi_version() != 1:
            raise ValueError('Expected verification ABI1')
        self.lib.ornith_verify_paged_get_layout.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(PagedLayout)]
        self.lib.ornith_verify_paged_get_layout.restype = ctypes.c_int
        self.lib.ornith_verify_paged_launch.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_size_t]
            + [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [PagedStrides, ctypes.c_void_p])
        self.lib.ornith_verify_paged_launch.restype = ctypes.c_int
        self.lib.ornith_verify_metadata_launch.argtypes = ([ctypes.c_void_p] * 6
            + [ctypes.c_int] * 4 + [ctypes.c_int64, ctypes.c_void_p])
        self.lib.ornith_verify_metadata_launch.restype = ctypes.c_int
        layout = PagedLayout()
        if self.lib.ornith_verify_paged_get_layout(Ccap, Lcap, ctypes.byref(layout)):
            raise ValueError('Verify workspace query failed')
        self.workspace_bytes = layout.bytes
        self.output_offset = (layout.bytes + 255) // 256 * 256
        self.table_offset = self.output_offset + Ccap * 16 * 256 * 4
        self.table_cols = (Lcap + self.page_size - 1) // self.page_size
        self.lengths_offset = self.table_offset + Ccap * self.table_cols * 4
        self.arena_bytes = self.lengths_offset + Ccap * 4

    def bind_arena(self, arena):
        if (arena.dtype != torch.uint8 or arena.ndim != 1 or not arena.is_contiguous()
                or arena.numel() < self.arena_bytes or arena.data_ptr() % 256
                or arena.device != self.device):
            raise ValueError('Invalid verification attention arena')
        return SimpleNamespace(
            workspace=arena[:self.workspace_bytes],
            output=arena[self.output_offset:self.table_offset].view(torch.float32),
            table=arena[self.table_offset:self.lengths_offset].view(torch.int32),
            lengths=arena[self.lengths_offset:self.arena_bytes].view(torch.int32))

    def launch_out(self, query, key, value, table, lengths, starts, actual, buffers, flag, output):
        stream = torch.cuda.current_stream(query.device).cuda_stream
        status = self.lib.ornith_verify_metadata_launch(
            *[t.data_ptr() for t in (table, lengths, starts, buffers.table, buffers.lengths, flag)],
            query.shape[0], lengths.shape[0], actual, self.table_cols, table.stride(0), stream)
        if status:
            raise RuntimeError(f'Verify metadata launch failed: {status}')
        strides = PagedStrides(*query.stride()[:2], *key.stride(), *value.stride(), self.table_cols)
        status = self.lib.ornith_verify_paged_launch(
            *[t.data_ptr() for t in (query, key, value, buffers.table, buffers.lengths, buffers.workspace)],
            buffers.workspace.numel(), buffers.output.data_ptr(), output.data_ptr(), flag.data_ptr(),
            query.shape[0], self.Ccap, self.Lcap, key.shape[0], strides, stream)
        if status:
            raise RuntimeError(f'Verify attention launch failed: {status}')
