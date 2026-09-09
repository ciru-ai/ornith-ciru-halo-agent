"""Strict direct packed-bank loader for pinned MoERunner delegation.

Checkpoint keys use model.language_model.layers.N.mlp.experts.routed_experts.FIELD.
Qwen3.5's wrapper maps only the outer model prefix. MoERunner passes the
routed_experts.FIELD suffix to this bound RoutedExperts method unchanged.
This avoids upstream's per-expert/fused-weight matcher, which consumes and
silently drops unmatched direct parameter names. No installed class is patched.
"""
# Copyright 2026 Ciru.
from types import MethodType

FIELDS=('w13_tilebank_codes','w13_tilebank_metadata',
        'w2_tilebank_codes','w2_tilebank_metadata')
PREFIX='routed_experts.'


def _load_direct(self,weights):
    import torch
    if getattr(self,'_ornith_weights_ready',False):raise ValueError('Tilebank weights already sealed')
    for key,value in weights:
        if not key.startswith(PREFIX) or key[len(PREFIX):] not in FIELDS:
            raise ValueError('Unsupported tilebank checkpoint field '+key)
        field=key[len(PREFIX):]
        if field in self._tilebank_loaded_fields:raise ValueError('Duplicate tilebank field '+field)
        parameter=self._parameters.get(field)
        if parameter is None:raise ValueError('Missing registered tilebank parameter '+field)
        if value.shape!=parameter.shape or value.dtype!=parameter.dtype:
            raise ValueError('Tilebank shape/dtype mismatch for '+field)
        if value.device.type=='meta' or parameter.device.type=='meta':
            raise ValueError('Tilebank loader requires materialized tensors')
        if field.endswith('_metadata'):
            if value.device.type!='cpu' or value.dtype!=torch.uint32:
                raise ValueError('Tilebank metadata admission expects CPU uint32 packed pairs')
            pairs=value.contiguous().view(torch.float16).reshape(*value.shape,2)
            if not bool(torch.isfinite(pairs).all()) or not bool((pairs[...,0]>=2**-14).all()):
                raise ValueError('Tilebank metadata requires finite offset and normal-floor scale')
        with torch.no_grad():parameter.copy_(value)
        self._tilebank_loaded_fields.add(field)
        # Relative to the MoERunner, not its RoutedExperts child.
        yield PREFIX+field


def install_tilebank_loader(layer):
    """Call once from the project quant method's create_weights on RoutedExperts."""
    if hasattr(layer,'_tilebank_loaded_fields'):raise ValueError('Tilebank loader already installed')
    if any(name not in layer._parameters for name in FIELDS):raise ValueError('Create all four tilebank parameters first')
    layer._tilebank_loaded_fields=set()
    layer.load_weights=MethodType(_load_direct,layer)


def require_complete_tilebank_load(layer):
    """Call during process_weights_after_loading, before runtime binding."""
    if getattr(layer,'_tilebank_loaded_fields',set())!=set(FIELDS):
        raise ValueError('Incomplete tilebank packed-bank checkpoint coverage')
