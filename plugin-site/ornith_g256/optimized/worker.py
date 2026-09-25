"""Retained optimizations around the published worker lifecycle."""
import hashlib
import inspect
import os
from pathlib import Path

from ornith_g256.worker import OrnithG256Worker as Parent


class Worker(Parent):
    def init_device(self):
        result = super().init_device()
        self._optimized_ready = False
        from .pipeline.adapter import install
        self._optimized_pipeline = install(lambda: self._optimized_ready)

        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
        namespace = qwen_gdn_linear_attn.QwenGatedDeltaNetAttention._forward_core.__globals__
        original = namespace['fused_post_conv_prep']
        digest = hashlib.sha256(Path(inspect.getsourcefile(original)).read_bytes()).hexdigest()
        if digest != '590717b2107abc90d72548de06b5d315229eb2775c94614f3f545df11cc2b5df':
            raise RuntimeError('The installed GDN postconv source differs from the qualified parent')
        from .postconv import fused_post_conv_prep

        def postconv(*args, **kwargs):
            selected = fused_post_conv_prep if self._optimized_ready else original
            return selected(*args, **kwargs)

        namespace['fused_post_conv_prep'] = postconv
        return result

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        cache_root = Path(os.environ['ORNITH_OPTIMIZED_CACHE'])
        audit = cache_root / 'optimized-runtime' / str(os.getpid())
        audit.mkdir(parents=True, exist_ok=False)
        for name in ['qk', 'draft', 'sampler']:
            (audit / name).mkdir()

        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        original_tuner_run = CachingAutotuner.run
        from .qk import install as install_qk
        from .draft import install as install_draft
        from .compile_scope import install as install_compile_scope
        from .qk_schedule import install as install_qk_schedule
        install_qk(audit / 'qk')
        install_draft(audit / 'draft', cache_root)
        self._optimized_compile_scope = install_compile_scope(original_tuner_run, audit)
        install_qk_schedule(audit)

        from .greedy import install as install_greedy
        from .prefix import install as install_prefix
        install_greedy(audit / 'sampler')
        self._optimized_prefix = install_prefix()
        return result

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        # Original traversal selects the initial tuner configurations. A new
        # signature later executes its original stage once before switching.
        self._optimized_ready = True
        return result

    def ornith_optimized_snapshot(self):
        return dict(ready=self._optimized_ready,
                    pipeline=self._optimized_pipeline.snapshot(),
                    compile_scope=self._optimized_compile_scope(),
                    prefix=dict(self._optimized_prefix))
