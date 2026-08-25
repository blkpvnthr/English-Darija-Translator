"""
darija_normalizer.py

Python port of the repo's DarijaNormalizer.js. Maps Arabizi / Arabic-chat-alphabet
substitutions (numbers-as-sounds and a few digraphs) to standard Arabic script, so that
Arabic-script MT models (NLLB, Terjman) receive input in the script they expect.

IMPORTANT LIMITATION (surfaced by the benchmark, not hidden):
This only transliterates the digit-sounds and a handful of digraphs. It does NOT do full
Arabizi->Arabic transliteration -- pure Latin letters (e.g. the "andi" in "3andi") are left
as-is. Full romanization handling is an open Stage-2 decision (either an LLM front-door like
Atlas-Chat, or a real transliterator). See bench/README.md.
"""

# Ported verbatim from ARABIC_CHAT_MAP in DarijaNormalizer.js
ARABIC_CHAT_MAP = {
    # Single-character number -> Arabic letter
    "3": "ع",   # 'ayn
    "7": "ح",   # ḥā’
    "9": "ق",   # qāf
    "2": "ء",   # hamza / glottal stop
    "5": "خ",   # khā’
    "6": "ط",   # ṭā’
    "8": "غ",   # ghayn
    # Digraphs (must be applied before the single-char keys above)
    "dh": "ذ",
    "d7": "ظ",  # emphatic dh
    "sh": "ش",
    "s5": "ص",
    "s9": "ص",
    "d9": "ض",
    "z7": "ز",
}


def normalize_darija(text: str) -> str:
    """
    Normalize Arabizi/Arabic-chat-alphabet text toward standard Arabic script.

    Fixes the JS version's ordering hazard by replacing *all* keys longest-first, so multi-char
    keys ('sh', 's5', 'd9', ...) always win over the single-char keys they contain ('5', '9', ...).
    """
    if not text or not isinstance(text, str):
        return ""

    normalized = text.lower()

    # Longest keys first so e.g. 's5' -> 'ص' beats '5' -> 'خ', and 'sh' -> 'ش' beats a bare 's'.
    for key in sorted(ARABIC_CHAT_MAP, key=len, reverse=True):
        normalized = normalized.replace(key, ARABIC_CHAT_MAP[key])

    # Collapse any accidental double spaces.
    normalized = " ".join(normalized.split())
    return normalized


if __name__ == "__main__":
    samples = [
        "3andi 7ob l dar, 9albi dima fi bladi.",
        "shkun ghadi ydir 2chghal d9i9a?",
        "labas? kolchi mzyan?",
    ]
    for s in samples:
        print(f"{s!r}\n  -> {normalize_darija(s)!r}")
