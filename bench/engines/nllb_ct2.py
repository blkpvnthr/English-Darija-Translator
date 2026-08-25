"""
NLLB-200-distilled-600M via CTranslate2 (int8) — bidirectional EN<->Darija.

Moroccan Arabic is a first-class NLLB language: eng_Latn <-> ary_Arab.
On first load we convert facebook/nllb-200-distilled-600M to an int8 CTranslate2 model
(cached under bench/models/), which is what keeps RAM ~1 GB and inference fast on CPU.
"""

from __future__ import annotations

import os
import subprocess
import sys

from .base import Engine, EN2DAR, DAR2EN

HF_MODEL = "facebook/nllb-200-distilled-600M"
LANG = {EN2DAR: ("eng_Latn", "ary_Arab"), DAR2EN: ("ary_Arab", "eng_Latn")}

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CT2_DIR = os.path.join(_HERE, "models", "nllb-600m-ct2-int8")


class NllbCt2Engine(Engine):
    name = "NLLB-600M"
    supported = (EN2DAR, DAR2EN)
    needs_arabic_script = True

    def __init__(self):
        self.translator = None
        self.tokenizer = None

    def load(self) -> None:
        import ctranslate2
        import transformers

        if not os.path.isdir(CT2_DIR):
            os.makedirs(os.path.dirname(CT2_DIR), exist_ok=True)
            print(f"[NLLB] converting {HF_MODEL} -> int8 CTranslate2 (one time)...", flush=True)
            subprocess.run(
                [
                    "ct2-transformers-converter",
                    "--model", HF_MODEL,
                    "--output_dir", CT2_DIR,
                    "--quantization", "int8",
                ],
                check=True,
            )

        # intra_threads=2 matches the 2 vCPU box; int8 kernels run on CPU.
        self.translator = ctranslate2.Translator(
            CT2_DIR, device="cpu", compute_type="int8", inter_threads=1, intra_threads=2
        )
        # The SentencePiece tokenizer still comes from the HF repo.
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(HF_MODEL)

    def translate(self, text: str, direction: str):
        src_lang, tgt_lang = LANG[direction]
        self.tokenizer.src_lang = src_lang
        source = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        results = self.translator.translate_batch(
            [source], target_prefix=[[tgt_lang]], beam_size=4, max_decoding_length=256
        )
        target = results[0].hypotheses[0]
        if target and target[0] == tgt_lang:  # drop the forced language token
            target = target[1:]
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target))

    def unload(self) -> None:
        self.translator = None
        self.tokenizer = None
