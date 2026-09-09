"""Reuse the retained exact-parent column kernel without its old admission campaign."""
# Copyright 2026 Ciru.
from collections import Counter
from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend, RocmAttentionImpl
from .column_backend import OrnithColumnAttentionImpl


class OrnithG256AttentionBackend(RocmAttentionBackend):
    @staticmethod
    def get_name(): return 'CUSTOM'
    @staticmethod
    def get_impl_cls(): return OrnithG256AttentionImpl


class OrnithG256AttentionImpl(OrnithColumnAttentionImpl):
    def __init__(self, *args, **kwargs):
        RocmAttentionImpl.__init__(self, *args, **kwargs)
        self.dispatch_counts, self.capture_by_C, self.eager_by_C = Counter(), Counter(), Counter()
        self.context_bounds = [None, None]
