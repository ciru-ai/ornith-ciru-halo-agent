"""Run the Qwen shared expert as a ninth routed slot in the N32 storage decode path."""
import ctypes as C
import torch

_INSTALLED = False
_STATE = {}


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from . import native
    from .method import G256MoEMethod
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
    original_apply = G256MoEMethod.apply
    original_forward = SharedExperts.forward

    def supported(mlp, x):
        lib = native._libs.get('routed')
        return (lib is not None and hasattr(lib[1], 'ornith_routed_direct_launch_shared') and x.shape[0] <= 16
                and getattr(mlp, 'expert_gate', None) is not None
                and hasattr(mlp.gate_up_proj, '_dense_n32_codes') and hasattr(mlp.down_proj, '_dense_n32_codes'))

    def forward(self, shared_experts_input, order):
        if not supported(self._layer, shared_experts_input):
            return original_forward(self, shared_experts_input, order)
        if order != self._determine_shared_experts_order(shared_experts_input):
            return None
        _STATE['pending'] = torch.empty_like(shared_experts_input)
        self._output[self._output_idx] = _STATE['pending']

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        mlp = None if shared_experts is None else shared_experts._layer
        if (mlp is None or shared_experts_input is None or self.runtime is None
                or _STATE.get('pending') is None or not supported(mlp, x)):
            return original_apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
        lib = native._libs['routed'][1]
        if 'launch' not in _STATE:
            launch = lib.ornith_routed_direct_launch_shared
            launch.argtypes = [C.c_void_p] * 13 + [C.c_size_t] + [C.c_void_p] * 3 + [C.c_int] * 2 + [C.c_void_p]
            launch.restype = C.c_int
            layout = (C.c_size_t * 20)()
            native.check(lib.ornith_routed_shared_get_layout(16, layout), 'routed shared layout')
            _STATE['launch'] = launch
            _STATE['workspace'] = torch.empty(int(layout[0]), dtype=torch.uint8, device=x.device)
        x = x.contiguous()
        logit = torch.ops.ornith_g256.small_gemm(x, mlp.expert_gate.weight, None, False)
        out = torch.empty_like(x)
        shared = _STATE.pop('pending')
        workspace = _STATE['workspace']
        native.check(_STATE['launch'](
            native.ptr(x), native.ptr(topk_weights), native.ptr(topk_ids),
            native.ptr(layer.w13_tilebank_codes), native.ptr(layer.w13_tilebank_metadata),
            native.ptr(layer.w2_tilebank_codes), native.ptr(layer.w2_tilebank_metadata),
            native.ptr(mlp.gate_up_proj._dense_n32_codes), native.ptr(mlp.gate_up_proj._dense_n32_metadata),
            native.ptr(mlp.down_proj._dense_n32_codes), native.ptr(mlp.down_proj._dense_n32_metadata),
            native.ptr(logit), native.ptr(workspace), workspace.numel(), native.ptr(out), native.ptr(shared),
            native.ptr(self.runtime.slots[self.slot]), x.shape[0], 16, native.stream(x)), 'G256 routed shared')
        return out

    G256MoEMethod.apply = apply
    SharedExperts.forward = forward
    _INSTALLED = True
