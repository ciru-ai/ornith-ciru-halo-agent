"""Enumerate the released shared scalar functions for every BF16 input bit pattern."""
import hashlib
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors.torch import save_file
from torch._inductor.runtime.triton_helpers import libdevice

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True,
                    help='Fresh directory for regenerated reference data and verification receipt')
OUT = parser.parse_args().output.resolve()
OUT.mkdir(parents=True, exist_ok=False)


@triton.jit
def released(G, SILU, SIGMOID, COUNT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    g = tl.load(G+i, i < COUNT).to(tl.float32)
    denominator = 1.0 + libdevice.exp(-g)
    tl.store(SILU+i, g / denominator, i < COUNT)
    tl.store(SIGMOID+i, 1.0 / denominator, i < COUNT)


g = torch.arange(65536, dtype=torch.int32, device='cuda').to(torch.int16).view(torch.bfloat16)
silu = torch.empty_like(g, dtype=torch.float32)
sigmoid = torch.empty_like(g)
released[(256,)](g, silu, sigmoid, 65536, 256, num_warps=4, enable_fp_fusion=True)
original_silu = torch.compile(lambda t: torch.nn.functional.silu(t.float()), fullgraph=True)(g)
original_sigmoid = torch.compile(torch.sigmoid, fullgraph=True)(g)
finite = torch.isfinite(g)
checks = {
    'finite_inputs': int(finite.sum()),
    'silu_FP32_bit_differences': int(((silu.view(torch.int32) != original_silu.view(torch.int32)) & finite).sum()),
    'sigmoid_BF16_bit_differences': int(((sigmoid.view(torch.int16) != original_sigmoid.view(torch.int16)) & finite).sum())}
assert checks['silu_FP32_bit_differences'] == checks['sigmoid_BF16_bit_differences'] == 0, checks
path = OUT / 'shared_math_reference.safetensors'
save_file({'silu_fp32': silu.cpu(), 'sigmoid_bf16': sigmoid.cpu()}, str(path))
receipt = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'checks': checks,
           'input_index': 'Raw sixteen BF16 bits; every one of 65536 patterns has an entry.',
           'payload_bytes': silu.numel()*4 + sigmoid.numel()*2,
           'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'torch_version': torch.__version__, 'triton_version': triton.__version__,
           'device': torch.cuda.get_device_name(0),
           'semantics': 'Released compiled shared SiLU FP32 result before multiplication by BF16 up; released shared sigmoid stored as BF16.',
           'verification_scope': 'All 65280 finite BF16 input patterns match independent compiled released expressions by bits; nonfinite behavior remains subject to existing runtime flags.'}
(OUT / 'shared_math_reference.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
