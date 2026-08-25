"""
Terjman-Large-v2.0 (AtlasIA) via transformers — EN->Darija only.

Fine-tuned from Helsinki-NLP/opus-mt-tc-big-en-ar specifically on Darija, ~240M params.
Unidirectional per the model card, so DAR2EN returns None (shown as N/A in the report).
"""

from __future__ import annotations

from .base import Engine, EN2DAR

# atlasia/ is the canonical repo; BounharAbdelaziz/Terjman-Large-v2.0 is the author mirror.
MODEL_NAME = "atlasia/Terjman-Large-v2.0"


class TerjmanEngine(Engine):
    name = "Terjman-Large"
    supported = (EN2DAR,)
    needs_arabic_script = True  # only ever receives English here, so irrelevant in practice

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.set_num_threads(2)  # match 2 vCPU
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.model.eval()

    def translate(self, text: str, direction: str):
        if direction != EN2DAR:
            return None
        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=256, num_beams=4)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
