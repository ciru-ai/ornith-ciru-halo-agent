"""Own configurable worker; reuse the existing step-level flag transport."""
# Copyright 2026 Ciru.
import torch
from vllm.v1.worker.gpu_worker import Worker as VllmGPUWorker
from .worker_base import OrnithWorkerBase
from .lifecycle import WorkerStepLifecycle
from .config import OrnithG256Config
from .runtime import Runtime
from .prefix_cache import validate_prefix_cache_config


class OrnithG256Worker(OrnithWorkerBase):
    def __init__(self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False):
        from .no_spec import install as install_no_spec
        install_no_spec()
        c = vllm_config
        self.settings = dict(c.additional_config.get('ornith_g256', {}))
        for key in ('dense_library', 'routed_library', 'head_library'):
            if not self.settings.get(key): raise ValueError(f"additional_config.ornith_g256.{key} is required")
        tile = self.settings.get('attention_query_tile', 32)
        if tile == 16:
            from .attention_tile import install as install_attention_tile
            install_attention_tile(compact_prefill=self.settings.get("compact_prefill", False),
                                   folded_decode=self.settings.get("folded_decode", False),
                                   folded_decode_max_queries=self.settings.get("folded_decode_max_queries", 8))
        elif tile != 32:
            raise ValueError('Attention query tile must be16 or32')
        if self.settings.get('token_major_kv', False):
            if (tile != 16 or not self.settings.get('compact_prefill', False)
                    or not self.settings.get('folded_decode', False)
                    or c.cache_config.block_size not in (560, 1120)):
                raise ValueError('Token-major KV requires compact/folded target attention and page1120/2240')
            from .attention_storage import install as install_attention_storage
            install_attention_storage()
        if __import__('os').environ.get('ORNITH_PERSISTENT_IU4_LIBRARY'):
            from .attention_iu4_persistent import install as install_persistent_iu4
            install_persistent_iu4()
        spec = c.speculative_config
        context = c.model_config.max_model_len
        if (self.settings.get('iu4_prefill_library')
                and not self.settings.get('dynamic_spec_profile')):
            graph = c.compilation_config
            graph_mode = getattr(graph.cudagraph_mode, 'name', graph.cudagraph_mode)
            capture_sizes = graph.cudagraph_capture_sizes
            if (spec is None or spec.method != 'dflash' or spec.num_speculative_tokens != 7
                    or not 65536 <= context <= 262144 or c.scheduler_config.max_num_seqs != 8
                    or c.scheduler_config.async_scheduling
                    or c.cache_config.block_size != 1120
                    or not c.cache_config.enable_prefix_caching
                    or tile != 16 or self.settings.get('attention_mode') != 'stock_rocm'
                    or not all(self.settings.get(key, False) for key in
                               ('compact_prefill', 'folded_decode', 'token_major_kv', 'draft_full_retention'))
                    or self.settings.get('routed_prefill_activation_bits') != 4
                    or graph_mode != 'FULL_DECODE_ONLY' or not capture_sizes
                    or max(capture_sizes) > 64):
                raise ValueError('IU4 prefill requires the synchronous agents64k profile: '
                                 'DFlash7, max8/64K, page1120, fine prefix reuse, compact/folded '
                                 'token-major A4 and FULL_DECODE_ONLY graphs up to64 rows')
        if context > 8192 and (
                context > 262144 or spec is None or spec.method != 'dflash'
                or self.settings.get('attention_mode') != 'stock_rocm'):
            raise ValueError('Context above8192 requires DFlash with stock_rocm attention, up to262144')
        self._spec_enabled = spec is not None
        validate_prefix_cache_config(c)
        if spec is not None:
            if (spec.method not in ('mtp', 'dflash')
                    or (spec.method == 'mtp' and spec.num_speculative_tokens != 1)
                    or (spec.method == 'dflash' and spec.num_speculative_tokens not in (3, 7, 15))
                    or c.cache_config.mamba_cache_mode not in ('align', 'none')):
                raise ValueError("G256 speculation supports MTP1 or DFlash block4/8/16; prefix reuse is restricted to DFlash7")
            from .gdn_spec import install
            install()
            if spec.method == 'dflash':
                from .dflash_spec import install as install_dflash
                install_dflash()
        p = c.parallel_config
        if (not isinstance(c.quant_config, OrnithG256Config) or c.model_config.dtype != torch.bfloat16
                or any(getattr(p, key) != 1 for key in
                       ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size', 'prefill_context_parallel_size'))
                or p.enable_expert_parallel or p.enable_dbo or p.use_sequence_parallel_moe
                or c.use_v2_model_runner or c.scheduler_config.async_scheduling
                or c.scheduler_config.max_num_seqs > 8 or c.scheduler_config.max_num_batched_tokens > 2048
                or c.lora_config is not None
                or c.model_config.runner_type != 'generate' or c.model_config.enable_sleep_mode):
            raise ValueError("G256 prototype requires BF16, TP1 V1 generation, synchronous scheduling, <=8 sequences/2048 tokens")
        if self.settings.get('dynamic_spec_profile'):
            from .dynamic_graphs import install
            install()
        self._ornith_binding = self._ornith_lifecycle = self._ornith_pending = None
        VllmGPUWorker.__init__(self, vllm_config=vllm_config, local_rank=local_rank, rank=rank,
                              distributed_init_method=distributed_init_method, is_driver_worker=is_driver_worker)

    def load_model(self, *, load_dummy_weights=False):
        from .dense_source import install as install_dense_source
        install_dense_source()
        from .dense_n32 import install as install_dense_n32
        install_dense_n32()
        from .gdn_compact import install
        install()
        from .phase_dispatch import install
        install()
        if load_dummy_weights or self._ornith_binding is not None:
            raise RuntimeError("G256 prototype loads the complete checkpoint once")
        from .dflash_conv_boundary import install as install_conv_boundaries
        install_conv_boundaries()
        VllmGPUWorker.load_model(self, load_dummy_weights=False)
        spec = self.vllm_config.speculative_config
        if spec is not None and spec.method == 'dflash':
            # Installed AOT selector can specialize on batch1 and then
            # reject real batch4/8. Keep only this small selector eager;
            # target verification and the draft transformer retain graphs.
            draft = self.model_runner.drafter.model
            if hasattr(draft, 'unwrap'):
                draft = draft.unwrap()
            draft.model.candidate_selector.do_not_compile = True
        model = self.model_runner.get_model()
        model.eval()
        before = torch.cuda.memory_allocated(self.device)
        self._ornith_binding = Runtime(model, self.settings,
                                      self.vllm_config.scheduler_config.max_num_batched_tokens, self.device,
                                      max_model_len=self.vllm_config.model_config.max_model_len,
                                      max_num_seqs=self.vllm_config.scheduler_config.max_num_seqs)
        if self.settings.get('compact_prefill', False):
            from .attention_compact import configure
            configure(max_num_seqs=self.vllm_config.scheduler_config.max_num_seqs,
                      max_model_len=self.vllm_config.model_config.max_model_len,
                      max_num_batched_tokens=self.vllm_config.scheduler_config.max_num_batched_tokens,
                      device=self.device,
                      iu4_prefill_library=self.settings.get('iu4_prefill_library'))
        self._ornith_resident_bytes = torch.cuda.memory_allocated(self.device)-before
        self.model_runner.model_memory_usage += self._ornith_resident_bytes
        self._ornith_lifecycle = WorkerStepLifecycle(
            self._ornith_binding, max_inflight=self.vllm_config.max_concurrent_batches+1)

    def get_kv_cache_spec(self):
        from .prefix_cache import retained_draft_specs
        return retained_draft_specs(self.vllm_config, super().get_kv_cache_spec())

    def execute_model(self, scheduler_output):
        # In this installed V1 runner normal sample logits run in execute_model.
        # Prompt logits run later in sample_tokens, after the inherited snapshot.
        for request in scheduler_output.scheduled_new_reqs:
            sampling = request.sampling_params
            if sampling is not None and sampling.prompt_logprobs is not None:
                raise ValueError("ornith_g256 prototype does not support prompt_logprobs")
        if not self._spec_enabled:
            return super().execute_model(scheduler_output)
        lifecycle = self._require_lifecycle()
        if self._ornith_pending is not None:
            raise RuntimeError('sample_tokens must finish the preceding speculative step')
        ticket = lifecycle.begin('execute-model-and-draft')
        try:
            output = VllmGPUWorker.execute_model(self, scheduler_output)
        except Exception as error:
            lifecycle.abort(ticket, error)
        if output is None:
            # Proposal generation, including the shared native W8 head, runs
            # inside sample_tokens. Snapshot only after those launches.
            self._ornith_pending = ticket
            return None
        lifecycle.queue(ticket)
        return lifecycle.checked_output(output, ticket)

    def compile_or_warm_up_model(self):
        if __import__('os').environ.get('ORNITH_PERSISTENT_IU4_LIBRARY'):
            from .attention_iu4_persistent import bind
            added = bind(self.model_runner)
            self._ornith_resident_bytes += added
            self.model_runner.model_memory_usage += added
        if not self._spec_enabled:
            return super().compile_or_warm_up_model()
        # Exploratory capture uses vLLM's synthetic GDN inputs/state. Their
        # output values are discarded; unlike real requests they are not a
        # finite-input contract. Preserve other faults and log the observed
        # numerical flags. Every real execute/sample still resets and checks
        # the complete flag bank through the ordinary lifecycle.
        lifecycle = self._require_lifecycle()
        ticket = lifecycle.begin('speculative-dummy-warmup')
        try:
            result = VllmGPUWorker.compile_or_warm_up_model(self)
            flags = self._ornith_binding.error_flags.cpu().tolist()
            observed = {}
            from .method import G256LinearMethod, G256MoEMethod
            for i, (name, code) in enumerate(zip(self._ornith_binding.layer_prefixes, flags)):
                method = (self._ornith_binding.layers[i].quant_method
                          if i < len(self._ornith_binding.layers) else None)
                # Larger DFlash capture batches use the BF16 prefill path,
                # which propagates undefined dummy GDN values downstream.
                # Dense bits 1/2/4/8 and routed bits 1/2/8/16/32 are
                # numerical; routed bit 4 (invalid expert ID) stays fatal.
                numeric_mask = (15 if type(method) is G256LinearMethod else
                                59 if isinstance(method, G256MoEMethod) else 0)
                if code and code & ~numeric_mask == 0:
                    observed[name] = code
                    self._ornith_binding.error_flags[i] = 0
            if observed:
                from vllm.logger import init_logger
                init_logger(__name__).info(
                    'Exploratory synthetic warmup numerical flags (real-request checks remain enabled): %s', observed)
            lifecycle.queue(ticket)
            lifecycle.complete(ticket)
            return result
        except Exception as error:
            lifecycle.abort(ticket, error)

    def sample_tokens(self, grammar_output):
        if not self._spec_enabled:
            return super().sample_tokens(grammar_output)
        lifecycle = self._require_lifecycle()
        ticket = self._ornith_pending
        if ticket is None:
            return VllmGPUWorker.sample_tokens(self, grammar_output)
        self._ornith_pending = None
        try:
            output = VllmGPUWorker.sample_tokens(self, grammar_output)
        except Exception as error:
            lifecycle.abort(ticket, error)
        lifecycle.queue(ticket)
        return lifecycle.checked_output(output, ticket)

    def ornith_persistent_snapshot(self):
        from .attention_iu4_persistent import snapshot
        return snapshot()

    def ornith_served_snapshot(self):
        self._ornith_lifecycle.drain()
        binding = self._ornith_binding
        return dict(flags=binding.check_error_flags(), arena_bytes=binding.workspace.numel(),
                    dense_shapes=binding.shapes, native_layers=len(binding.layer_prefixes),
                    resets=self._ornith_lifecycle.resets, copies=self._ornith_lifecycle.copies,
                    completions=self._ornith_lifecycle.completions)
