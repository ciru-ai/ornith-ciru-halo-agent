"""Keep DFlash convolution request boundaries dynamic inside token-count graphs."""
# Copyright 2026 Ciru. Original convolution arithmetic from vLLM, Apache-2.0.
from contextvars import ContextVar
import inspect
import linecache
import logging
import torch
import torch.nn.functional as F

_ACTIVE_MASK = ContextVar('ornith_dflash_query_mask', default=None)
_INSTALLED = False
_ORIGINAL_CONVOLVE = None
_INPUT_KERNEL = None
logger = logging.getLogger(__name__)


def grouped_conv_dynamic(hidden_states, delta, base, query_mask, num_groups, group_size, taps):
    # Float expressions and evaluation order are unchanged from upstream.
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    position = position & query_mask
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


def convolve(self, hidden_states, delta, side):
    mask = getattr(self, '_ornith_query_mask', None)
    if mask is None:
        return _ORIGINAL_CONVOLVE(self, hidden_states, delta, side)
    return grouped_conv_dynamic(hidden_states, delta, self.base_kernel[side], mask,
                                self.num_groups, self.group_size, self.taps)


def build_input_kernel():
    """Add one scalar store to the existing input-preparation launch."""
    global _INPUT_KERNEL
    if _INPUT_KERNEL is not None:
        return _INPUT_KERNEL
    from vllm.v1.spec_decode import utils
    original = utils.copy_and_expand_dflash_inputs_kernel
    source = inspect.getsource(original.fn)
    arg = '    out_token_indices_ptr,  # [num_reqs * num_speculative_tokens] (output)\n'
    body = '    block_idx = tl.program_id(axis=1)\n'
    if source.count(arg) != 1 or source.count(body) != 1:
        raise RuntimeError('Unsupported DFlash input-kernel source')
    source = source.replace('def copy_and_expand_dflash_inputs_kernel(',
                            'def _copy_and_expand_with_query_mask(', 1)
    source = source.replace(arg, arg + '    out_query_mask_ptr,  # persistent scalar for captured convolutions\n', 1)
    source = source.replace(body, body + '    tl.store(out_query_mask_ptr, num_query_per_req - 1,\n'
                            '             mask=(req_idx == 0) & (block_idx == 0))\n', 1)
    filename = __file__ + '.input.generated'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(vars(utils))
    exec(compile(source, filename, 'exec'), namespace)
    _INPUT_KERNEL = namespace['_copy_and_expand_with_query_mask']
    return _INPUT_KERNEL


class _InputKernelProxy:
    def __init__(self, original):
        self.original = original

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            mask = _ACTIVE_MASK.get()
            if mask is None:
                return self.original[grid](*args, **kwargs)
            return build_input_kernel()[grid](*args, out_query_mask_ptr=mask, **kwargs)
        return launch


def install():
    global _INSTALLED, _ORIGINAL_CONVOLVE
    if _INSTALLED:
        return
    from vllm.v1.spec_decode import dflash
    from vllm.model_executor.models.qwen3_dflash2 import DFlashGroupedConv
    original_load = dflash.DFlashProposer.load_model
    original_inputs = dflash.DFlashProposer.set_inputs_first_pass
    _ORIGINAL_CONVOLVE = DFlashGroupedConv._convolve

    def load_model(self, target_model):
        result = original_load(self, target_model)
        if not self.is_dflash2:
            return result
        width = 1 + self.speculative_config.num_speculative_tokens
        if width not in (8, 16):
            raise ValueError('Current boundary fix supports original Q8/Q16 only')
        self._ornith_query_mask = torch.full((1,), width - 1, dtype=torch.int32, device=self.device)
        model = self.model.unwrap() if hasattr(self.model, 'unwrap') else self.model
        count = 0
        for module in model.modules():
            if isinstance(module, DFlashGroupedConv):
                module.register_buffer('_ornith_query_mask', self._ornith_query_mask, persistent=False)
                count += 1
        if count != 12:
            raise RuntimeError(f'Expected current six-layer DFlash2 with12 convolutions, got {count}')
        logger.info('DFlash boundary correction:12 convs share one4-byte GPU mask; dynamicQ8/Q16')
        return result

    def set_inputs_first_pass(self, *args, **kwargs):
        mask = getattr(self, '_ornith_query_mask', None)
        if mask is None:
            return original_inputs(self, *args, **kwargs)
        width = 1 + self.num_speculative_tokens
        if width not in (8, 16):
            raise ValueError(f'Unexpected current DFlash query width{width}')
        token = _ACTIVE_MASK.set(mask)
        try:
            return original_inputs(self, *args, **kwargs)
        finally:
            _ACTIVE_MASK.reset(token)

    build_input_kernel()
    dflash.copy_and_expand_dflash_inputs_kernel = _InputKernelProxy(dflash.copy_and_expand_dflash_inputs_kernel)
    dflash.DFlashProposer.load_model = load_model
    dflash.DFlashProposer.set_inputs_first_pass = set_inputs_first_pass
    DFlashGroupedConv._convolve = convolve
    _INSTALLED = True
