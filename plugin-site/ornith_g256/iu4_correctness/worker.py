"""Production-only corrected IU4 worker; no benchmark or host dependencies."""
import hashlib
import json
import os
from pathlib import Path

from ornith_g256.optimized.worker import Worker as Released
from .integration import load_selection, verify_sources


class Worker(Released):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        self._iu4_production_selection = load_selection(require_concurrency=True)
        pins = verify_sources()
        self._iu4_correctness_audit = Path(os.environ['ORNITH_OPTIMIZED_CACHE']) / 'iu4-correctness' / str(os.getpid())
        self._iu4_correctness_audit.mkdir(parents=True, exist_ok=False)
        from .final_router_ablation import install_components
        install_components(self, self._iu4_correctness_audit, dict(
            variant='all', repairs=self._iu4_production_selection['repairs']))
        from .final_sampler import install as install_filter
        install_filter(self._iu4_correctness_audit)
        from .final_native_sampling import install as install_native_forward
        install_native_forward(self, self._iu4_correctness_audit)
        receipt = dict(schema='ciru.ornith.iu4.production-worker.v1', pid=os.getpid(),
            numerical_parent='ornith_g256.optimized.worker.Worker',
            selection=self._iu4_production_selection, source_pins=pins,
            worker_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            model_weights_changed=False, native_libraries_changed=False,
            qualification='SOURCE_PROPOSAL_REQUIRES_SEPARATE_RELEASE_GATES')
        (self._iu4_correctness_audit / 'production-worker.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print('ORNITH_IU4_PRODUCTION_WORKER ' + json.dumps(receipt), flush=True)
        return result

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        from .rng_exact_restore import install
        install(self.model_runner, self._iu4_correctness_audit / 'discard-rng.jsonl')
        return result
