"""Ciru G256 prototype; no installed vLLM files are modified."""


from .runtime_correctness import install as _install_correctness
_install_correctness()


def register():
    from .graph_phase_guard import install as install_graph_phase_guard
    install_graph_phase_guard()
    from .cache_full1120 import install as install_full1120
    install_full1120()
    from vllm.model_executor.layers.quantization import (
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG, register_quantization_config,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    from .config import OrnithG256Config

    existing = _CUSTOMIZED_METHOD_TO_QUANT_CONFIG.get("ornith_g256")
    if existing is None:
        register_quantization_config("ornith_g256")(OrnithG256Config)
    elif existing is not OrnithG256Config:
        raise RuntimeError("Another plugin owns ornith_g256")
    register_backend(AttentionBackendEnum.CUSTOM, "ornith_g256.attention_fast.OrnithG256SelectableAttentionBackend")
    # Install in every vLLM process: cache groups are also built in the engine
    # core, which does not instantiate our worker. The wrapper is a no-op for
    # prefix-off runs and models outside this project.
    from .prefix_cache import install
    install()

    from .adaptive_c1 import install as install_adaptive_c1
    install_adaptive_c1()
