"""
Darija-specialist MT engine (Helsinki opus-mt fine-tuned on Darija) via transformers — EN->Darija.

Default model: lachkarsalim/Helsinki-translation-English_Moroccan-Arabic  (ungated, MarianMT).
This fills the same niche as AtlasIA's Terjman but WITHOUT the HF gating that blocked it
(atlasia/Terjman-Large-v2.0 and the BounharAbdelaziz mirror are both gated).

To benchmark the gated Terjman instead, get an HF token, accept its terms on the model page, then:
    export HF_TOKEN=hf_xxx
    export TERJMAN_MODEL=atlasia/Terjman-Large-v2.0
transformers/huggingface_hub pick up HF_TOKEN automatically. Unidirectional either way (EN->Darija);
DAR2EN returns None (shown as N/A in the report).
"""

from __future__ import annotations

import os

from .base import Engine, EN2DAR

DEFAULT_MODEL = "lachkarsalim/Helsinki-translation-English_Moroccan-Arabic"
MODEL_NAME = os.environ.get("TERJMAN_MODEL", DEFAULT_MODEL)


class TerjmanEngine(Engine):
    supported = (EN2DAR,)
    needs_arabic_script = True  # only ever receives English here, so irrelevant in practice

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = MODEL_NAME
        # Report which model actually ran.
        self.name = "Helsinki-Darija" if MODEL_NAME == DEFAULT_MODEL else MODEL_NAME.split("/")[-1]

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.set_num_threads(2)  # match 2 vCPU
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
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
