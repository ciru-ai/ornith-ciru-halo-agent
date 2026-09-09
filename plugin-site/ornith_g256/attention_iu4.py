# Copyright 2026 Ciru. Isolated C1 attention binding; no installed runtime edits.
import ctypes
import torch


class NativeAttention:
    """Preallocated signed IU4 scratch and stream-only native enqueue calls."""

    def __init__(self, device, library_path, max_model_len=65536):
        if not 1120 <= max_model_len <= 262144:
            raise ValueError("IU4 prefill capacity must be1120..262144")
        self.max_keys = 65536 if max_model_len <= 65536 else 262144
        groups = self.max_keys // 32
        self.library = ctypes.CDLL(str(library_path))
        self.library.iu4_prepare.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        self.library.iu4_prepare.restype = ctypes.c_int
        self.library.iu4_attention.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] + [ctypes.c_void_p]
        self.library.iu4_attention.restype = ctypes.c_int
        specs = [((16, 1120, 32), torch.int32), ((16, 1120), torch.float16),
                 ((2, groups, 32, 32), torch.int32), ((2, self.max_keys), torch.float16),
                 ((2, groups, 4, 256), torch.int32), ((2, groups, 256), torch.float16)]
        self.packed = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in specs]
        self.pointers = [tensor.data_ptr() for tensor in self.packed]
        self.output = torch.empty((1120, 16, 256), dtype=torch.bfloat16, device=device)
        self.bytes = self.output.numel() * self.output.element_size() + sum(tensor.numel() * tensor.element_size() for tensor in self.packed)

    def forward(self, query, key, value, output, starts, lengths, max_keys):
        # C1 max_query_len1120 and exact CPU metadata max_seq_len are required.
        # The unchanged consumer uses its original contiguous fixed1120 output.
        # No allocation, tensor copy to CPU, synchronization or cache mutation here.
        if not 1120 <= max_keys <= self.max_keys or max_keys % 32:
            raise ValueError("IU4 prefill requires aligned keys within reserved capacity")
        stream = torch.cuda.current_stream().cuda_stream
        rc = self.library.iu4_prepare(query.data_ptr(), key.data_ptr(), value.data_ptr(),
            *self.pointers, max_keys, query.stride(0), query.stride(1), stream)
        if rc:
            raise RuntimeError(f'IU4 attention preparation launch failed: HIP {rc}')
        rc = self.library.iu4_attention(*self.pointers, self.output.data_ptr(), max_keys, stream)
        if rc:
            raise RuntimeError(f'IU4 attention launch failed: HIP {rc}')
        output[:1120].copy_(self.output)
