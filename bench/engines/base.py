"""Engine interface shared by every candidate translator."""

from __future__ import annotations

# Language directions used across the harness.
EN2DAR = "en2dar"
DAR2EN = "dar2en"


class Engine:
    """
    A translation engine wraps one model + inference backend.

    Lifecycle (driven by bench/worker.py, one engine per OS process):
        e = EngineImpl()
        e.load()                       # download/convert + load weights into RAM
        out = e.translate(text, dir)   # dir is EN2DAR or DAR2EN; return None if unsupported
        e.unload()                     # best-effort free (process exit reclaims the rest)
    """

    #: Human-readable column name in the report.
    name = "base"

    #: Directions this engine can actually perform.
    supported = (EN2DAR, DAR2EN)

    #: Whether Arabic-script input is required (True) or raw Arabizi is fine (False).
    #: MT models need Arabic script; the LLM can take Arabizi directly.
    needs_arabic_script = True

    def load(self) -> None:  # pragma: no cover - backend specific
        raise NotImplementedError

    def translate(self, text: str, direction: str):  # pragma: no cover - backend specific
        raise NotImplementedError

    def unload(self) -> None:
        pass
