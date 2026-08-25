# Darija translator — model benchmark (Stage 1)

Goal: decide **which self-hosted model** the production translator should run on, using real
quality + latency + RAM measured on **your** VPS (2 vCPU / 4 GB / CPU-only) — not on reputation.

Contenders:

| Engine | Model | Direction | Notes |
|---|---|---|---|
| `nllb` | `entai2965/nllb-200-distilled-600M-ctranslate2` (int8, **pre-converted**) | EN ↔ Darija | bidirectional, native `ary_Arab` |
| `terjman` | `lachkarsalim/Helsinki-translation-English_Moroccan-Arabic` (transformers) | EN → Darija | ungated Darija-fine-tuned Helsinki, MarianMT |
| `atlaschat` *(optional)* | `QuantFactory/Atlas-Chat-2B-GGUF` Q4_K_M (llama.cpp) | both | LLM, best fluency, slow on CPU |

> **Why not AtlasIA's Terjman?** `atlasia/Terjman-Large-v2.0` is a **gated** HF repo (needs login +
> access approval), so the `terjman` engine defaults to the *ungated* Helsinki-Darija fine-tune, which
> fills the same niche. To benchmark the real Terjman instead, get a free HF token, accept its terms on
> the model page, then run with `HF_TOKEN=hf_xxx TERJMAN_MODEL=atlasia/Terjman-Large-v2.0 python bench/run_bench.py`.
>
> **Why pre-converted NLLB?** Converting NLLB to int8 on the box loads the full ~2.5 GB fp32 model into
> RAM and gets **OOM-killed on 4 GB**. Downloading the pre-converted int8 build sidesteps that (~1 GB
> resident at inference).

## Run it on the VPS

```bash
# 1. system deps (build-essential/cmake only needed for the optional Atlas-Chat engine)
sudo apt-get update && sudo apt-get install -y python3-venv git

# 2. get the code onto the VPS (git clone/pull this repo, or scp the bench/ folder), then:
cd English-Darija-Translator
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install -r bench/requirements.txt

# 3. run the core comparison (NLLB + Terjman). First run downloads/converts ~1.5 GB.
python bench/run_bench.py

# 4. (optional) also benchmark the 2B LLM — needs build tools + the extra dep:
sudo apt-get install -y build-essential cmake
pip install "llama-cpp-python>=0.2.80"
python bench/run_bench.py --with-atlaschat
```

Outputs land in `bench/out/`:
- `report.html` — open this; side-by-side translations + a summary table (load time, peak RAM,
  latency, chrF). **RAM/latency are decided by numbers; quality you judge by eye.**
- `results.json` — raw data.

## Design notes

- **One engine per subprocess.** Each model is benchmarked in its own process so no two large
  models are ever resident at once (safe on 4 GB) and peak RAM (`ru_maxrss`) is measured cleanly.
- **Normalization is load-bearing here.** Arabizi Darija inputs are run through
  `darija_normalizer.py` (the ported `DarijaNormalizer.js`) before the MT models, which need
  Arabic script. That normalizer only maps digit-sounds (`3→ع`, `7→ح`…), **not** full romanization —
  so Arabizi rows will look rough on NLLB/Terjman. That gap is shown on purpose; deciding how to
  close it (LLM front-door vs. a real transliterator) is a Stage-2 question.
- **chrF references are approximate** hand-written Darija — a rough sanity signal, not a leaderboard.

## After you pick a winner

Send me `bench/out/report.html` (or just tell me which engine won). Stage 2 builds the production
FastAPI service around it: it serves the existing web UI and a `/api/translate` endpoint, loads the
winning model once at startup, and needs no API key. See the plan in
`~/.claude/plans/hashed-petting-widget.md`.
