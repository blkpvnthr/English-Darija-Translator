"""
NLLB-200-distilled-600M via CTranslate2 (int8) — bidirectional EN<->Darija.

Moroccan Arabic is a first-class NLLB language: eng_Latn <-> ary_Arab.

We download a PRE-CONVERTED int8 CTranslate2 build (entai2965/nllb-200-distilled-600M-ctranslate2)
rather than converting on the box. Converting locally loads the full ~2.5 GB fp32 model into RAM and
gets OOM-killed on a 4 GB VPS; the pre-converted model is ~600 MB to download and ~1 GB resident at
inference. The repo ships model.bin + tokenizer files at its root, so both load from one directory.
"""

from __future__ import annotations

from .base import Engine, EN2DAR, DAR2EN

CT2_REPO = "entai2965/nllb-200-distilled-600M-ctranslate2"
TOKENIZER_FALLBACK = "facebook/nllb-200-distilled-600M"  # public; used only if the CT2 repo tokenizer fails
LANG = {EN2DAR: ("eng_Latn", "ary_Arab"), DAR2EN: ("ary_Arab", "eng_Latn")}


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
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(CT2_REPO)  # cached under ~/.cache/huggingface

        # intra_threads=2 matches the 2 vCPU box; the model is already int8.
        self.translator = ctranslate2.Translator(
            model_dir, device="cpu", compute_type="int8", inter_threads=1, intra_threads=2
        )
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
            self.tokenizer.src_lang = "eng_Latn"  # sanity-check this is really an NLLB tokenizer
        except Exception:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_FALLBACK)

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
