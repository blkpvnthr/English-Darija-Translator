"""
Atlas-Chat-2B (Darija-instruct LLM) via llama-cpp-python — optional contender.

Best fluency + handles code-switching/Arabizi directly (like the old Gemini prompt), but
~5-15 s/translation on 2 CPU cores. Only loaded when run_bench.py is given --with-atlaschat.
Downloads a Q4_K_M GGUF from QuantFactory/Atlas-Chat-2B-GGUF.
"""

from __future__ import annotations

from .base import Engine, EN2DAR

REPO_ID = "QuantFactory/Atlas-Chat-2B-GGUF"
QUANT = "Q4_K_M"

SYSTEM = (
    "You are a professional Moroccan Darija translator. Translate the user's text naturally "
    "and colloquially. Output ONLY the translation, no labels, no commentary, no quotes."
)


class AtlasChatEngine(Engine):
    name = "Atlas-Chat-2B"
    supported = (EN2DAR, "dar2en")
    needs_arabic_script = False  # the LLM can read raw Arabizi

    def __init__(self):
        self.llm = None

    def load(self) -> None:
        from huggingface_hub import hf_hub_download, list_repo_files
        from llama_cpp import Llama

        # Find the exact Q4_K_M filename in the repo (naming varies across GGUF packagers).
        files = list_repo_files(REPO_ID)
        gguf = [f for f in files if f.lower().endswith(".gguf")]
        match = [f for f in gguf if QUANT.lower() in f.lower()]
        if not match:
            raise RuntimeError(f"No {QUANT} GGUF found in {REPO_ID}; available: {gguf}")
        path = hf_hub_download(repo_id=REPO_ID, filename=match[0])

        self.llm = Llama(model_path=path, n_ctx=2048, n_threads=2, verbose=False)

    def translate(self, text: str, direction: str):
        tgt = "Moroccan Darija" if direction == EN2DAR else "English"
        user = f"Translate the following into {tgt}:\n{text}"
        # Gemma chat format has no system role, so we fold the instructions into the user turn.
        resp = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": f"{SYSTEM}\n\n{user}"}],
            max_tokens=256,
            temperature=0.3,
        )
        return resp["choices"][0]["message"]["content"].strip()

    def unload(self) -> None:
        self.llm = None
