"""Engine registry. Maps CLI names -> Engine subclasses (imported lazily by the worker)."""


def get_engine(name: str):
    if name == "nllb":
        from .nllb_ct2 import NllbCt2Engine
        return NllbCt2Engine()
    if name == "terjman":
        from .terjman_hf import TerjmanEngine
        return TerjmanEngine()
    if name == "atlaschat":
        from .atlaschat_gguf import AtlasChatEngine
        return AtlasChatEngine()
    raise ValueError(f"unknown engine: {name}")
