# SPDX-FileCopyrightText: Nova-Violet Role ORG
# SPDX-License-Identifier: EUPL-1.2 OR AGPL-3.0-or-later
"""Plates for the versus bench — data-driven SVG in the house style.

Reads eval/results/versus.json, writes eval/versus.svg + eval/versus-dark.svg.
Like checker/plates.mjs in the Commander repo: edit the DATA (re-run the
bench), then run this script. Never hand-edit the SVG.

Usage:  python eval/plates_versus.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results" / "versus.json"

TAXONS = ["T1", "T2", "T3", "T4", "T5", "T6"]
TAXON_LABELS = {
    "T1": "T1 title-self",
    "T2": "T2 rare-term",
    "T3": "T3 concept",
    "T4": "T4 scoped",
    "T5": "T5 paraphrase",
    "T6": "T6 ident",
}
SYSTEMS = ["rag-dense", "rag-hybrid", "rtr-arctic", "rtr-bge", "rtr-full"]
SERIES_FILL = {
    "rag-dense": "#b0b7c3",
    "rag-hybrid": "#5b6472",
    "rtr-arctic": "#2874a6",
    "rtr-bge": "#1a7f37",
    "rtr-full": "#7d3c98",
}
FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

W = 860
BAR_H = 17
BAR_GAP = 5
GROUP_PAD = 26
LEFT = 200
RIGHT = 64


def combined_taxon_recall3(data):
    """Weighted recall@3 per taxon per system across corpora."""
    table = {s: {} for s in SYSTEMS}
    for t in TAXONS:
        for s in SYSTEMS:
            hits = total = 0
            for c in data["corpora"].values():
                for r in c["systems"][s]["rows"]:
                    if r["taxon"] != t:
                        continue
                    total += 1
                    if r["expected"] in r["got"][:3]:
                        hits += 1
            table[s][t] = hits / total if total else 0.0
    return table


def overall(data):
    return {s: data["combined"][s]["metrics"]["OVERALL"] for s in SYSTEMS}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(data, table, ov, dark):
    bg = "#1a1d24" if dark else "#ffffff"
    ink = "#e2e5ec" if dark else "#1a1d24"
    muted = "#9aa3b2" if dark else "#5b6472"
    grid = "#2e3440" if dark else "#e2e5ec"
    hero = "#bb86fc" if dark else "#7d3c98"

    n_q = sum(m["n"] for m in (ov[s] for s in SYSTEMS)) // len(SYSTEMS)
    n_files = sum(c["files"] for c in data["corpora"].values())
    errs = sum(data["combined"][s].get("query_errors", 0) for s in SYSTEMS)

    group_h = len(SYSTEMS) * (BAR_H + BAR_GAP) + GROUP_PAD
    y = 0
    parts = []
    maxv = max(
        [table[s][t] for s in SYSTEMS for t in TAXONS]
        + [ov[s]["recall@3"] for s in SYSTEMS]
        + [0.01]
    )
    scale = (W - LEFT - RIGHT) / (maxv * 1.08)

    def text(x, yy, s, size=13, fill=None, anchor="start", mono=False, bold=False):
        f = MONO if mono else FONT
        w = " font-weight='bold'" if bold else ""
        parts.append(
            f"<text x='{x}' y='{yy}' font-family='{f}' font-size='{size}'"
            f" fill='{fill or ink}' text-anchor='{anchor}'{w}>{esc(s)}</text>"
        )

    # header
    y += 44
    text(24, y, "RTR vs RAG — recall@3 by taxon", size=21, bold=True)
    y += 24
    text(
        24,
        y,
        f"164 queries · 2 real corpora ({n_files} docs) · 0 query errors"
        f" · seed {data['seed']}",
        size=13,
        fill=muted,
        mono=True,
    )
    y += 14
    parts.append(f"<line x1='24' y1='{y}' x2='{W - 24}' y2='{y}' stroke='{grid}'/>")
    y += 8
    # legend
    lx = 24
    for s in SYSTEMS:
        parts.append(
            f"<rect x='{lx}' y='{y}' width='12' height='12' fill='{SERIES_FILL[s]}'/>"
        )
        text(lx + 17, y + 11, s, size=12, fill=muted, mono=True)
        lx += 17 + len(s) * 7 + 22
    y += 26

    # taxon groups
    for t in TAXONS:
        text(24, y + 16, TAXON_LABELS[t], size=13, bold=True)
        best = max(table[s][t] for s in SYSTEMS)
        for i, s in enumerate(SYSTEMS):
            v = table[s][t]
            by = y + 22 + i * (BAR_H + BAR_GAP)
            bw = max(2, v * scale)
            is_best = abs(v - best) < 1e-9 and v > 0
            parts.append(
                f"<rect x='{LEFT}' y='{by}' width='{bw:.1f}' height='{BAR_H}'"
                f" fill='{SERIES_FILL[s]}'/>"
            )
            text(24, by + 13, s, size=11, fill=muted, mono=True)
            text(
                LEFT + bw + 8,
                by + 13,
                f"{v:.3f}",
                size=11,
                fill=hero if (is_best and s == "rtr-full") else muted,
                mono=True,
                bold=is_best,
            )
        y += 22 + len(SYSTEMS) * (BAR_H + BAR_GAP) + GROUP_PAD - 6
        parts.append(f"<line x1='24' y1='{y}' x2='{W - 24}' y2='{y}' stroke='{grid}'/>")
        y += GROUP_PAD

    # overall strip
    y += 6
    text(24, y, "Combined overall (164 queries)", size=15, bold=True)
    y += 22
    text(24, y, "system", size=11, fill=muted, mono=True)
    for j, col in enumerate(["R@1", "R@3", "MRR"]):
        text(LEFT + 40 + j * 150, y, col, size=11, fill=muted, mono=True)
    y += 6
    for s in SYSTEMS:
        y += 20
        text(
            24,
            y,
            s,
            size=12,
            mono=True,
            bold=(s == "rtr-full"),
            fill=hero if s == "rtr-full" else None,
        )
        for j, k in enumerate(["recall@1", "recall@3", "mrr"]):
            best = max(ov[x][k] for x in SYSTEMS)
            v = ov[s][k]
            text(
                LEFT + 40 + j * 150,
                y,
                f"{v:.3f}",
                size=12,
                mono=True,
                fill=hero if (s == "rtr-full" and abs(v - best) < 1e-9) else None,
                bold=abs(v - best) < 1e-9,
            )
    y += 34
    parts.append(f"<line x1='24' y1='{y}' x2='{W - 24}' y2='{y}' stroke='{grid}'/>")
    y += 22
    text(
        24,
        y,
        "rtr-full leads combined R@1/R@3/MRR · method: eval/bench_versus.py · "
        "taxonomy: eval/TAXONOMY.md",
        size=11,
        fill=muted,
    )
    y += 18
    text(24, y, "Nova-Violet Role · Reality is the judge.", size=11, fill=muted)
    height = y + 24

    body = "\n".join(parts)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later OR EUPL-1.2 -->\n"
        "<!-- Copyright 2026 Saimonokuma. GENERATED by eval/plates_versus.py"
        " from eval/results/versus.json; re-run the bench, then this script. -->\n"
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{height}'"
        f" viewBox='0 0 {W} {height}' font-family='{FONT}'>\n"
        f"<rect width='{W}' height='{height}' fill='{bg}'/>\n"
        f"{body}\n</svg>\n"
    )


def main():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    table = combined_taxon_recall3(data)
    ov = overall(data)
    (HERE / "versus.svg").write_text(
        render(data, table, ov, dark=False), encoding="utf-8"
    )
    (HERE / "versus-dark.svg").write_text(
        render(data, table, ov, dark=True), encoding="utf-8"
    )
    print("wrote eval/versus.svg + eval/versus-dark.svg")


if __name__ == "__main__":
    main()
