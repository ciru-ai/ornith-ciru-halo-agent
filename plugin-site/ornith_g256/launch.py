"""Launch the retained Ornith G256 runtime from explicit local artifact paths."""
# Copyright 2026 Ciru.
import argparse
import json
import os
from pathlib import Path
from .cache_full1120 import install as install_full1120
install_full1120()


LIBRARIES = {
    'dense_library': 'libornith_dense_g256.so',
    'routed_library': 'libornith_routed_direct.so',
    'head_library': 'libornith_head_i8_tile.so',
    'attention_library': 'libornith_paged-v2.so',
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--draft', type=Path, help='DFlash2 checkpoint (required for dflash)')
    p.add_argument('--native-library-directory', type=Path, required=True)
    p.add_argument('--mode', choices=('dflash', 'ar', 'mtp1'), default='dflash')
    p.add_argument('--draft-tokens', type=int, choices=(0, 7, 15), default=None,
                   help='Fixed C1 draft depth, including above 32K: 0, 7 or 15. '
                        'Omit for adaptive C1; C2–8 retains 7. 0 retains drafter allocations.')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--served-name', default='ornith-g256-dflash2')
    p.add_argument('--enable-images', action='store_true',
                   help='Enable one image per request, preserving max8 concurrent sequences')
    p.add_argument('--text-only', dest='enable_images', action='store_false',
                   help='Disable the image encoder for a text-only process')
    p.add_argument('--image-max-pixels', type=int, default=1048576,
                   help='Image preprocessing pixel budget when --enable-images is set')
    p.add_argument('--enable-tools', action='store_true',
                   help='Enable automatic tool calling with the original Qwen3 XML format')
    p.add_argument('--context', type=int, default=4096)
    p.add_argument('--max-seqs', type=int, default=8)
    p.add_argument('--cache-gib', type=float, default=32)
    p.add_argument('--block-size', type=int, help='Optional vLLM KV block size; default keeps runtime selection')
    p.add_argument('--prefix-cache', action='store_true',
                   help='Experimental DFlash7 aligned prefix reuse; selects KV block size1120')
    p.add_argument('--fine-prefix-cache', action='store_true',
                   help='With --prefix-cache, retain draft history for8-token prompt-tail matching')
    p.add_argument('--compact-prefill', action='store_true', help='Experimental compact KV FlashAttention prefill')
    p.add_argument('--iu4-prefill-library', type=Path,
                   help='Optional fixed1120 signed IU4 prefill library; requires the current agents64k profile')
    p.add_argument('--folded-decode', action='store_true', help='Experimental query/head shared KV decode')
    p.add_argument('--token-major-kv', action='store_true', help='Store target KV token-major in existing cache pages')
    p.add_argument('--routed-prefill-a4', action='store_true', help='Experimental one-pass A4 experts for rows above64; A8 generation')
    p.add_argument('--routed-n32-library', type=Path,
                   help='Optional N32 expert decode library; adds resident N32 weight banks')
    p.add_argument('--routed-n32-max-rows', type=int, choices=(8, 16), default=8,
                   help='Largest verification row count using the optional N32 expert library')
    p.add_argument('--routed-n32-storage-library', type=Path,
                   help='Full N32 expert consumer; replace loaded N16 banks instead of retaining shadows')
    p.add_argument('--cache-directory', type=Path,
                   default=Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'ornith-g256')
    p.add_argument('--settings-output', type=Path)
    p.add_argument('--dry-run', action='store_true', help='Print settings/environment without importing vLLM')
    p.add_argument('--inspect-only', action='store_true', help='Parse installed vLLM arguments without loading a model')
    return p


def make_settings(a):
    context_limit = 262144 if a.mode == 'dflash' else 8192
    if not 1 <= a.max_seqs <= 8 or not 1 <= a.context <= context_limit:
        raise ValueError(f'The {a.mode} runtime supports 1..8 sequences and 1..{context_limit} context')
    if not 1024 <= a.port <= 65535 or not a.cache_gib > 0 or not a.served_name:
        raise ValueError('Use a valid unprivileged port, positive cache size and model name')
    if a.mode == 'dflash' and a.draft is None:
        raise ValueError('--draft is required for DFlash2')
    if a.mode != 'dflash' and a.draft is not None:
        raise ValueError('--draft is only used with --mode dflash; MTP1 uses its target derivative')
    if a.block_size is not None and a.block_size <= 0:
        raise ValueError('--block-size must be positive')
    if a.prefix_cache and (a.mode != 'dflash' or a.block_size not in (None, 1120)):
        raise ValueError('--prefix-cache requires DFlash7 and --block-size1120 (or omitted)')
    if a.fine_prefix_cache and not a.prefix_cache:
        raise ValueError('--fine-prefix-cache requires --prefix-cache')
    if a.prefix_cache and a.draft_tokens != 7:
        raise ValueError('Prefix reuse currently requires7 draft tokens; test15 with prefix caching off')
    if a.mode == 'dflash' and a.draft_tokens == 15 and a.max_seqs != 1:
        raise ValueError('The experimental sixteen-token profile currently requires --max-seqs1; '
                         'concurrent verification needs its own expert precision/dispatch policy')
    if a.routed_n32_max_rows != 8 and a.routed_n32_library is None:
        raise ValueError('--routed-n32-max-rows requires --routed-n32-library')
    if a.routed_n32_storage_library is not None and a.routed_n32_library is not None:
        raise ValueError('Choose sole N32 storage or N32 decode shadows')
    if a.iu4_prefill_library is not None and not (
            a.mode == 'dflash' and a.draft_tokens == 7 and a.max_seqs == 8
            and 65536 <= a.context <= 262144 and a.block_size in (None, 1120)
            and a.prefix_cache and a.fine_prefix_cache
            and a.compact_prefill and a.folded_decode and a.token_major_kv and a.routed_prefill_a4):
        raise ValueError('--iu4-prefill-library requires agents64k: DFlash7, max8/64K, '
                         'page1120, fine prefix reuse and compact/folded token-major A4')
    native = a.native_library_directory.expanduser().resolve()
    settings = dict(
        model=str(a.model.expanduser().resolve()), load_format='safetensors', dtype='bfloat16',
        tensor_parallel_size=1, max_model_len=a.context, max_num_seqs=a.max_seqs,
        max_num_batched_tokens=2048, gpu_memory_utilization=0.8,
        kv_cache_memory_bytes=int(a.cache_gib*(1 << 30)), mamba_ssm_cache_dtype='float32',
        enable_prefix_caching=a.prefix_cache, enable_chunked_prefill=True, enforce_eager=False,
        async_scheduling=False, limit_mm_per_prompt={'image': int(a.enable_images), 'video': 0},
        seed=15035, disable_log_stats=False, quantization='ornith_g256', max_logprobs=248320, logprobs_mode='raw_logprobs',
        worker_cls='ornith_g256.worker.OrnithG256Worker',
        additional_config={'ornith_g256': dict(
            {key: str(native/name) for key, name in LIBRARIES.items()},
            attention_mode='fast_fp32', dense_a8_max_rows=64, attention_query_tile=16)},
        compilation_config=dict(mode=3, cudagraph_mode='FULL_DECODE_ONLY',
                                cudagraph_capture_sizes=list(range(8, 65, 8))),
        attention_config={'backend': 'CUSTOM'},
    )
    if a.context > 8192:
        # DFlash's long-context path uses the installed ROCm backend. The
        # retained fast native decoder only handles page1056 and <=8192;
        # it is not used for the page1120 DFlash cache and needs no allocation.
        settings['attention_config'] = {'backend': 'ROCM_ATTN'}
        native_settings = settings['additional_config']['ornith_g256']
        native_settings['attention_mode'] = 'stock_rocm'
        del native_settings['attention_library']
    if a.block_size is not None:
        settings['block_size'] = a.block_size
    if a.prefix_cache:
        settings['block_size'] = 1120
        # The final Mamba boundary may not have DFlash's full lookahead block.
        # Retain the preceding boundary and its draft window for cache lookup.
        settings['prefix_cache_retention_interval'] = 1120
    if a.fine_prefix_cache:
        settings['prefix_match_unit'] = 8
        settings['additional_config']['ornith_g256']['draft_full_retention'] = True
    if a.compact_prefill:
        if a.context <= 8192:
            raise ValueError('--compact-prefill requires the long-context ROCm profile')
        settings['additional_config']['ornith_g256']['compact_prefill'] = True
    if a.folded_decode:
        settings['additional_config']['ornith_g256']['folded_decode'] = True
    if a.token_major_kv:
        if not (a.compact_prefill and a.folded_decode):
            raise ValueError('--token-major-kv requires --compact-prefill and --folded-decode')
        settings['additional_config']['ornith_g256']['token_major_kv'] = True
    if a.routed_prefill_a4:
        settings['additional_config']['ornith_g256']['routed_prefill_activation_bits'] = 4
    if a.iu4_prefill_library is not None:
        settings['additional_config']['ornith_g256']['iu4_prefill_library'] = str(
            a.iu4_prefill_library.expanduser().resolve())
    if a.routed_n32_library is not None:
        settings['additional_config']['ornith_g256'].update(
            routed_decode_n32=True,
            routed_n32_max_rows=a.routed_n32_max_rows,
            routed_n32_library=str(a.routed_n32_library.expanduser().resolve()))
    if a.routed_n32_storage_library is not None:
        settings['additional_config']['ornith_g256'].update(
            routed_storage_n32=True,
            routed_library=str(a.routed_n32_storage_library.expanduser().resolve()))
    if a.mode == 'dflash':
        settings['speculative_config'] = dict(
            method='dflash', model=str(a.draft.expanduser().resolve()),
            num_speculative_tokens=a.draft_tokens, quantization=None, attention_backend='ROCM_ATTN')
        block = a.draft_tokens + 1
        settings['compilation_config']['cudagraph_capture_sizes'] = list(
            range(block, block * a.max_seqs + 1, block))
        if a.draft_tokens == 15:
            if not (a.context > 8192 and a.compact_prefill and a.folded_decode
                    and a.token_major_kv):
                raise ValueError('15 draft tokens require the long-context compact/folded token-major KV path')
            settings['additional_config']['ornith_g256']['folded_decode_max_queries'] = 16
        settings['mamba_cache_mode'] = 'align'
    elif a.mode == 'mtp1':
        settings['speculative_config'] = dict(
            method='mtp', num_speculative_tokens=1, moe_backend='triton', attention_backend='ROCM_ATTN')
        settings['mamba_cache_mode'] = 'align'
        settings['compilation_config']['cudagraph_capture_sizes'] = list(range(1, 17))
    else:
        settings['compilation_config']['cudagraph_capture_sizes'] = list(range(1, 9))
    if a.enable_images:
        if a.image_max_pixels < 65536:
            raise ValueError('--image-max-pixels must be at least the processor minimum65536')
        settings['mm_processor_kwargs'] = {'max_pixels': a.image_max_pixels}
        settings['mm_encoder_attn_backend'] = 'TRITON_ATTN'
    return settings


def environment(cache):
    cache = str(cache.expanduser().resolve())
    return dict(
        VLLM_PLUGINS='ornith_g256', OMP_NUM_THREADS='2', VLLM_MOE_SKIP_PADDING='1',
        VLLM_ROCM_USE_AITER='0', VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS='0',
        VLLM_DISABLE_SHARED_EXPERTS_STREAM='1', VLLM_USE_V2_MODEL_RUNNER='0',
        VLLM_ENABLE_V1_MULTIPROCESSING='1', VLLM_ALLOW_INSECURE_SERIALIZATION='0',
        VLLM_WORKER_MULTIPROC_METHOD='spawn', OPENBLAS_NUM_THREADS='2',
        VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
        PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        XDG_CACHE_HOME=cache, AITER_JIT_DIR=cache+'/aiter', TRITON_CACHE_DIR=cache+'/triton',
        TORCHINDUCTOR_CACHE_DIR=cache+'/inductor', VLLM_CACHE_ROOT=cache+'/vllm')


def main(argv=None):
    a = parser().parse_args(argv)
    settings = make_settings(a)
    env = environment(a.cache_directory)
    env['ORNITH_C1_POLICY'] = ('k' + str(a.draft_tokens)
                              if a.draft_tokens is not None
                              else os.environ.get('ORNITH_C1_POLICY', 'auto'))
    frontend = (dict(enable_auto_tool_choice=True, tool_call_parser='qwen3_xml', reasoning_parser='qwen3')
                if a.enable_tools else {})
    if a.prefix_cache:
        frontend['enable_prompt_tokens_details'] = True
    if a.settings_output:
        a.settings_output.parent.mkdir(parents=True, exist_ok=True)
        a.settings_output.write_text(json.dumps(settings, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(settings=settings, environment=env, frontend=frontend, host=a.host, port=a.port,
                          served_name=a.served_name), indent=2, allow_nan=False), flush=True)
    if a.dry_run:
        return
    for key in ('model',):
        if not Path(settings[key]).is_dir():
            raise FileNotFoundError(settings[key])
    if a.draft and not a.draft.expanduser().is_dir():
        raise FileNotFoundError(a.draft)
    for key in (*LIBRARIES, 'iu4_prefill_library'):
        path = settings['additional_config']['ornith_g256'].get(key)
        if path is None:
            continue
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    os.environ.update(env)
    # Keep all vLLM imports after environment selection. The project's installed
    # plugin entry point is also visible to worker processes via PYTHONPATH.
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from .dynamic_graphs import install_config_hook
    install_config_hook()
    from vllm.engine.arg_utils import AsyncEngineArgs
    import torch
    cli_env_setup()
    server_parser = make_arg_parser(FlexibleArgumentParser(description='Ornith G256 serving'))
    args = server_parser.parse_args([])
    for name, value in settings.items():
        if name not in AsyncEngineArgs.__dataclass_fields__ or not hasattr(args, name):
            raise ValueError('Unsupported installed vLLM engine setting: '+name)
        setattr(args, name, value)
    args.host, args.port, args.served_model_name = a.host, a.port, [a.served_name]
    args.disable_uvicorn_access_log = True
    for name, value in frontend.items():
        if not hasattr(args, name):
            if name == 'enable_prompt_tokens_details':
                print('ORNITH_G256_PROMPT_TOKEN_DETAILS_UNAVAILABLE: use prefix-cache metrics', flush=True)
                continue
            raise ValueError('Unsupported installed vLLM frontend setting: '+name)
        setattr(args, name, value)
    validate_parsed_serve_args(args)
    AsyncEngineArgs.from_cli_args(args)
    if a.inspect_only:
        if torch.cuda.is_initialized():
            raise RuntimeError('Argument inspection unexpectedly initialized CUDA')
        print('ORNITH_G256_ARGUMENTS_OK', flush=True)
        return
    if a.enable_tools:
        from .parser_contract import install as install_parser_contract
        install_parser_contract()
    import uvloop
    from vllm.entrypoints.launchers.api_server.entry import run_server
    uvloop.run(run_server(args))




# Isolated experiment: the original agents64k CLI establishes every base setting.
_original_make_settings = make_settings
def make_settings(a):
    # Base validator describes the retained DF15/7 graph envelope. Explicit
    # C1 depth is applied by the scheduler policy, not by resizing that envelope.
    base_args = argparse.Namespace(**vars(a))
    base_args.draft_tokens = 7
    settings = _original_make_settings(base_args)
    assert (a.mode == 'dflash' and a.max_seqs == 8
            and a.prefix_cache and a.fine_prefix_cache and a.iu4_prefill_library)
    settings['block_size'] = 1120
    settings['prefix_cache_retention_interval'] = 1120
    # Upstream Mamba alignment supports intermediate sub-block chunks. Keep
    # Q1120 to retain the measured IU4 prefill path and current A4 PP shape.
    settings['long_prefill_token_threshold'] = 1120
    settings['speculative_config'].update(num_speculative_tokens=15,
        num_speculative_tokens_per_batch_size=[(1, 1, 15), (2, 8, 7)])
    settings['compilation_config']['cudagraph_capture_sizes'] = [16, 32, 48, 64]
    native = settings['additional_config']['ornith_g256']
    native.update(dynamic_spec_profile=True, adaptive_c1_fallback=True, folded_decode_max_queries=16,
        routed_storage_n32=True,
        routed_library=str(Path(__file__).resolve().parents[2] / 'native' / 'libornith_routed_storage_n32.so'))
    return settings


_shared_head_settings = make_settings
def make_settings(a):
    s = _shared_head_settings(a)
    s['additional_config']['ornith_g256']['head_library'] = str(Path(__file__).resolve().parents[2] / 'native' / 'libornith_head_i8_tile.so')
    return s

_fine_settings = make_settings
def make_settings(a):
    s = _fine_settings(a)
    s['prefix_match_unit'] = None
    s['prefix_cache_retention_interval'] = 0
    s['additional_config']['ornith_g256']['draft_full_retention'] = False
    return s

if __name__ == '__main__':
    main()
