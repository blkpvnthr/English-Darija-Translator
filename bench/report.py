"""
report.py — render aggregated benchmark results into a self-contained report.html.

Two tables:
  1. Summary  — load time, peak RAM, latency, chrF per engine (the numbers you choose on).
  2. Side-by-side — every phrase's source next to each engine's translation, RTL-aware.
"""

from __future__ import annotations

import html
import re

_ARABIC = re.compile(r"[؀-ۿ]")


def _dir_attr(text: str | None) -> str:
    return ' dir="rtl"' if text and _ARABIC.search(text) else ""


def _cell(text, latency_ms=None, note=None) -> str:
    if text is None:
        body = '<span class="na">—</span>'
    else:
        body = html.escape(text)
    meta = ""
    if latency_ms is not None:
        meta = f'<div class="meta">{latency_ms} ms</div>'
    if note:
        meta += f'<div class="note">{html.escape(note)}</div>'
    return f'<td{_dir_attr(text)}>{body}{meta}</td>'


def render(aggregate: dict, out_path: str) -> None:
    phrases = aggregate["phrases"]
    results = aggregate["results"]
    summary = aggregate["summary"]

    # results[engine_name][phrase_id] -> row
    by_engine = {r["engine_name"]: {row["id"]: row for row in r["rows"]} for r in results}
    engine_names = [r["engine_name"] for r in results]

    # --- summary table ---
    sum_rows = "".join(
        f"<tr><td>{html.escape(s['engine_name'])}</td>"
        f"<td>{s['load_time_s']} s</td>"
        f"<td>{s['peak_rss_mb']} MB</td>"
        f"<td>{s['mean_latency_ms']} ms</td>"
        f"<td>{s['median_latency_ms']} ms</td>"
        f"<td>{'—' if s['chrf'] is None else s['chrf']}</td></tr>"
        for s in summary
    )

    # --- side-by-side table ---
    head = "".join(f"<th>{html.escape(n)}</th>" for n in engine_names)
    body_rows = []
    for p in phrases:
        arabizi = " <span class='tag'>Arabizi</span>" if p.get("src_is_arabizi") else ""
        direction = "EN → Darija" if p["direction"] == "en2dar" else "Darija → EN"
        src_cell = (
            f'<td class="src"{_dir_attr(p["src"])}>'
            f'<div class="dirtag">{direction}{arabizi}</div>'
            f'{html.escape(p["src"])}</td>'
        )
        cells = [src_cell]
        for n in engine_names:
            row = by_engine[n].get(p["id"], {})
            note = None
            if row.get("skipped"):
                note = "not supported"
            elif row.get("error"):
                note = row["error"]
            elif row.get("used_normalizer"):
                note = f"normalized → {row.get('input_used', '')}"
            cells.append(_cell(row.get("output"), row.get("latency_ms"), note))
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Darija translator — model benchmark</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem; color: #1a202c; background:#f7fafc; }}
  h1 {{ margin-bottom: .25rem; }} .sub {{ color:#4a5568; margin-top:0; }}
  table {{ border-collapse: collapse; width: 100%; background:#fff; margin: 1rem 0 2rem;
           box-shadow: 0 1px 3px rgba(0,0,0,.1); border-radius: 8px; overflow: hidden; }}
  th, td {{ border: 1px solid #e2e8f0; padding: .6rem .7rem; text-align: left; vertical-align: top; }}
  th {{ background: #edf2f7; font-weight: 700; }}
  td.src {{ background: #f8fafc; font-weight: 600; max-width: 260px; }}
  .dirtag {{ font-size: .7rem; color:#3182ce; font-weight:700; text-transform:uppercase; margin-bottom:.3rem; }}
  .tag {{ background:#fefcbf; color:#975a16; border-radius:4px; padding:0 .3rem; font-size:.65rem; }}
  .meta {{ color:#718096; font-size:.7rem; margin-top:.35rem; }}
  .note {{ color:#c05621; font-size:.7rem; margin-top:.2rem; font-style: italic; }}
  .na {{ color:#a0aec0; }}
  td[dir=rtl] {{ font-size: 1.05rem; }}
  .caveat {{ background:#fffaf0; border:1px solid #fbd38d; border-radius:8px; padding:.8rem 1rem;
             font-size:.85rem; color:#744210; }}
</style></head><body>
<h1>Darija translator — model benchmark</h1>
<p class="sub">Head-to-head on this VPS (CPU-only). Lower latency / RAM is better; higher chrF is better.
Translation quality is ultimately your call — read the side-by-side.</p>

<h2>Summary</h2>
<table>
  <tr><th>Engine</th><th>Load time</th><th>Peak RAM</th><th>Mean latency</th><th>Median latency</th><th>chrF ↑</th></tr>
  {sum_rows}
</table>

<h2>Side-by-side translations</h2>
<table>
  <tr><th>Source</th>{head}</tr>
  {''.join(body_rows)}
</table>

<p class="caveat"><b>Reading the Arabizi rows:</b> the MT models (NLLB, Terjman) require Arabic script,
so Arabizi inputs are first passed through <code>DarijaNormalizer</code> — which only maps digit-sounds
(3→ع, 7→ح…), not full romanization. Expect degraded MT output there; it's shown honestly to surface
the gap. An LLM engine (Atlas-Chat) reads Arabizi directly. chrF references are approximate and meant
as a rough sanity signal, not a leaderboard.</p>
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
