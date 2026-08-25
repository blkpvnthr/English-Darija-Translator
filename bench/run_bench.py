"""
run_bench.py — orchestrate the head-to-head benchmark.

Spawns one worker subprocess per engine (so no two big models are ever resident at once on the
4 GB box), aggregates their JSON, scores chrF against references where available, and renders a
self-contained report.html.

    python bench/run_bench.py                 # NLLB + Terjman
    python bench/run_bench.py --with-atlaschat   # also the 2B LLM (slow, +download)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PHRASES = os.path.join(HERE, "phrases.jsonl")
OUT_DIR = os.path.join(HERE, "out")


def load_phrases():
    with open(PHRASES, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_worker(engine_key: str) -> dict | None:
    out_path = os.path.join(OUT_DIR, f"results_{engine_key}.json")
    print(f"\n=== running engine: {engine_key} (isolated process) ===", flush=True)
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "worker.py"), engine_key, PHRASES, out_path]
    )
    if proc.returncode != 0:
        print(f"!! engine {engine_key} failed (exit {proc.returncode}); skipping", flush=True)
        return None
    with open(out_path, encoding="utf-8") as fh:
        return json.load(fh)


def score_chrf(engine_result: dict, phrases_by_id: dict) -> float | None:
    """chrF over rows that have a reference and produced output."""
    try:
        from sacrebleu.metrics import CHRF
    except Exception:
        return None
    hyps, refs = [], []
    for row in engine_result["rows"]:
        ref = phrases_by_id[row["id"]].get("ref")
        if ref and row.get("output"):
            hyps.append(row["output"])
            refs.append([ref])
    if not hyps:
        return None
    return round(CHRF().corpus_score(hyps, list(zip(*refs))).score, 1)


def summarize(engine_result: dict, phrases_by_id: dict) -> dict:
    lats = [r["latency_ms"] for r in engine_result["rows"] if r.get("latency_ms")]
    return {
        "engine_name": engine_result["engine_name"],
        "load_time_s": engine_result["load_time_s"],
        "peak_rss_mb": engine_result["peak_rss_mb"],
        "mean_latency_ms": int(statistics.mean(lats)) if lats else None,
        "median_latency_ms": int(statistics.median(lats)) if lats else None,
        "chrf": score_chrf(engine_result, phrases_by_id),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-atlaschat", action="store_true", help="also benchmark Atlas-Chat-2B (slow)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    phrases = load_phrases()
    phrases_by_id = {p["id"]: p for p in phrases}

    engine_keys = ["nllb", "terjman"]
    if args.with_atlaschat:
        engine_keys.append("atlaschat")

    results = [r for r in (run_worker(k) for k in engine_keys) if r is not None]
    if not results:
        sys.exit("All engines failed — see errors above.")

    summary = [summarize(r, phrases_by_id) for r in results]
    aggregate = {"phrases": phrases, "results": results, "summary": summary}

    agg_path = os.path.join(OUT_DIR, "results.json")
    with open(agg_path, "w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, ensure_ascii=False, indent=2)

    from report import render
    report_path = os.path.join(OUT_DIR, "report.html")
    render(aggregate, report_path)

    print("\n" + "=" * 60)
    print(f"results.json -> {agg_path}")
    print(f"report.html  -> {report_path}")
    print("=" * 60)
    for s in summary:
        print(f"  {s['engine_name']:>14}  "
              f"load {s['load_time_s']}s  peak {s['peak_rss_mb']}MB  "
              f"med {s['median_latency_ms']}ms  chrF {s['chrf']}")


if __name__ == "__main__":
    main()
