"""
worker.py — run ONE engine end-to-end in an isolated process.

Isolation per engine is deliberate: on a 4 GB box we never want two large models resident at
once, and a fresh process gives an accurate peak-RSS reading (resource.ru_maxrss) with no
cross-engine allocator carryover.

Usage (invoked by run_bench.py, but runnable directly):
    python bench/worker.py <engine> <phrases.jsonl> <out.json>
where <engine> is one of: nllb | terjman | atlaschat
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from darija_normalizer import normalize_darija  # noqa: E402
from engines import get_engine  # noqa: E402


def peak_rss_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS.
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        kb /= 1024.0
    return round(kb / 1024.0, 1)


def main() -> None:
    engine_name, phrases_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(phrases_path, encoding="utf-8") as fh:
        phrases = [json.loads(line) for line in fh if line.strip()]

    engine = get_engine(engine_name)

    t0 = time.time()
    engine.load()
    load_time_s = round(time.time() - t0, 1)

    rows = []
    for p in phrases:
        direction = p["direction"]
        if direction not in engine.supported:
            rows.append({"id": p["id"], "output": None, "latency_ms": None, "skipped": "unsupported"})
            continue

        # MT models need Arabic script; normalize Arabizi first. The LLM reads Arabizi directly.
        text = p["src"]
        used_normalizer = False
        if p.get("src_is_arabizi") and engine.needs_arabic_script:
            text = normalize_darija(text)
            used_normalizer = True

        t1 = time.time()
        try:
            output = engine.translate(text, direction)
            err = None
        except Exception as exc:  # keep the bench alive; record the failure per phrase
            output, err = None, f"{type(exc).__name__}: {exc}"
        latency_ms = int((time.time() - t1) * 1000)

        rows.append({
            "id": p["id"],
            "input_used": text,
            "used_normalizer": used_normalizer,
            "output": output,
            "latency_ms": latency_ms,
            "error": err,
        })

    result = {
        "engine_key": engine_name,
        "engine_name": engine.name,
        "load_time_s": load_time_s,
        "peak_rss_mb": peak_rss_mb(),
        "rows": rows,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"[{engine.name}] done: load {load_time_s}s, peak {result['peak_rss_mb']} MB", flush=True)


if __name__ == "__main__":
    main()
