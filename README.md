<!-- SPDX-FileCopyrightText: Nova-Violet Role ORG -->
<!-- SPDX-License-Identifier: EUPL-1.2 OR AGPL-3.0-or-later -->
# 🔍 RTR — Role Type Retrieval

**RRF-heuristic, DTD-semantic-capable retriever. CPU-first. Local-first. Zero API cost.**

[![Ko-fi](https://img.shields.io/badge/Support-Ko--fi-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/saimonokuma) [![Open Source](https://img.shields.io/badge/Open-Source-9cf?style=for-the-badge)](https://github.com/Nova-Violet-Role) [![Copyleft](https://img.shields.io/badge/Licence-AGPL--3.0--or--later_OR_EUPL--1.2-634bA2?style=for-the-badge)](#-licence--philosophy)

> *Retrieval is not search. It is routing by role, parsing by type — and where the
> models disagree is where the ranking lives.*

---

## 📜 About

**RTR** is the persistent memory and retrieval layer for coding agents — first
opencode, then every CLI that speaks MCP. Two ONNX embedding models
(Snowflake Arctic-m 768 + BAAI bge-base 768) run on plain CPU, fused with
reciprocal rank fusion over a BM25 + recency + importance hybrid. Document
spans are pre-classified the DTD way — PCDATA, CDATA, NDATA — so the models
receive typed evidence instead of generic soup, and a lookup-first
Lexikon keeps expansion chilly.

**🎯 Mission** — Give every agent a memory that is local, measurable, and
honest: every recall number in `eval/` re-runs from real data on your machine.

**🌟 Vision** — Retrieval that converges: roles decide *what* is fetched,
types decide *how* it is read — the RoT × DtD meeting point.

**⚖️ Method** — *Reality is the judge.* The `eval/` bench pits RTR against
classic RAG on real documentation and prints the scoreboard as data, diagram,
and SVG. If one of our numbers is wrong, telling us is the most welcome
contribution we accept.

---

## ⚡ The three pillars

| # | Pillar | What it does |
|---|---|---|
| 1 | **Multi-query expansion over RRF** | Original + lexikon/alias + workspace-delta sub-queries (+ optional PRF), fused across models with reciprocal rank fusion. Rankings vote; raw scores never mix. |
| 2 | **DTD-typed retrieval** | Spans classified structurally — `PCDATA` prose (hybrid lane), `CDATA` code/ids/hashes/paths/URLs (exact-match lane), `NDATA` base64/non-text (masked, never embedded, always carrying its NOTATION). Grounded in the DocBook Definitive Guide and the DITA 2.0 architecture. |
| 3 | **Lexikon + zero regression** | Per-project and hand-curated `_global` user glossary; expansion is a lookup, not inference. Every release passes the `rtr_eval` golden gate — no ship on regression. |

Plus the plumbing that makes it disappear: `uvx` delivery (no clone, no sync),
multi-root projects with absolute paths anywhere, per-model ChromaDB
collections, 4+4 / 8+8 thread profiles, polyglot indexing (`.md`, `.nt`, `.yaml`).

### 🏁 Bi-turbo philosophy & overseeding

Two powerplants — Arctic precision, bge recall — fused by RRF. Rally engineers
didn't twin-charge because one turbo was broken; they did it from love of the
paradigm, and the discoveries followed. Same here: the architecture takes any
fastembed model through one variable, and per-model collections make
overseeding **zero-migration**. Eight cores fly the balanced 4+4; stronger
silicon deserves larger turbos (`bge-large-en-v1.5`, `arctic-embed-l`) —
measure them on your corpus with `eval/bench_versus.py` and tell us what won.
Falsification is a contribution.

---

## 🚀 Quick start (opencode, 60 seconds)

```jsonc
// opencode.jsonc
"mcp": {
  "memory": {
    "type": "local",
    "command": ["uvx", "--from", "rtr-role-type-retrieval", "rtr-mcp"],
    "enabled": true,
    "timeout": 180000,
    "environment": {
      "OPENCODE_MEMORY_RTR_PROFILE": "performance", // balanced 4+4 | performance 8+8
      "FASTEMBED_CACHE_PATH": "C:/Users/you/.config/opencode/memory/.fastembed_cache"
    }
  }
}
```

Then: `rtr-cli init <project path-or-name>` → `rtr-cli index` → memory tools
appear in every session. Full instruction density lives in
[`README-PYPI.md`](README-PYPI.md) — every env knob, every CLI, every other
agent CLI and local-LLM setup.

---

## 🧪 Evaluation

`eval/` holds the versus bench: **RTR vs classic RAG**, measured on real
documentation with a published taxonomy (self-retrieval, rare-term lookup,
conceptual recall, scoped vs global), rendered as Mermaid and SVG.

```
uvx --from <wheel> python -m execution.rtr_eval   # release gate (no regressal)
python eval/bench_versus.py                        # full versus bench -> eval/results/
```

---

## 🥚 The Easter Egg — why this retriever thinks in roles, and where it is going

> *"Welcome to Ikebukuro — a city where countless stories intersect. And the
> solution **I found when inspecting how Models and Rankings vote together**.
> I continued to search. For a memory that never loses its head. Joy. The joy
> of recall **or the Artificial Memory**. Every document the protagonist of
> its own story. However, **what if** the faceless votes of the crowd
> **(the RRF consensus that derives from them)** converge on the one page
> that was lost?"*
> — Saimonokuma, **The Convergence Equation**

The original is the premise of *Durarara!!*: no single protagonist, only a
city where stories collide and something coherent walks out. The insertions
are RTR, and each one names a component that is now code:

| # | inserted | what it turned on |
|---|---|---|
| 1 | *"Models and Rankings **vote** together"* | the ensemble ballot. RRF fuses rankings, never raw scores — the Dollars vote faceless, and consensus needs no identity |
| 2 | *"memory that never loses its **head**"* | Celty's search is the product spec. Agents lose context the way she lost her head; retrieval is the city helping her find it |
| 3 | *"or the **Artificial Memory**"* | names the target: a memory that is constructed, persistent, local — what RTR ships |
| 4 | *"the **RRF consensus** that derives from them"* | points at `_fused_multi`. The intersection is a *mechanism*, not a mood |
| 5 | *"**what if** … **?**"* | turns an assertion into an open question — and the `eval/` bench is the instrument that answers it, per release |

### The same thing applies to RoT, our oT variants

Roles are turbos. Lenses, models, sub-queries — anything that can vote gets
a ballot and a weight, and the fusion never asks what stage anyone performed
on. That is the whole approach, and it ports: **whatever RoT routes by role,
RTR retrieves by role**, same ballot box.

### 🔮 Symbiose — announced, not shipped

The coming model uses the same RoT approach to be **Dense and MoE at the
same time** — dense parameters that route like experts, experts that share
one dense heart — with the RTR approach fused in from the first layer rather
than bolted on after. When it lands, the bench will be the announcement:
same corpora, same taxons, no adjectives. Until then this paragraph is a
promise, and promises are not results.

> *— Note of (Saimonokuma) the Creator: rally engineers twin-charged out of
> love for the paradigm and rewrote the car world by accident. We run twin
> models for the same reason. If Symbiose does to AI what bi-turbo did to
> rally, it will be because somebody loved the convergence more than the
> crown.*

## 🤝 Contributing

| Area | How you can help |
|---|---|
| 🎯 **Falsify a claim** | Prove one of our eval numbers wrong. Credited in the changelog. |
| 💻 **Code** | Pillars, profiles, new model slugs, new lanes |
| 📖 **Documentation** | An unanswerable question is our defect, not yours |
| 🧪 **Testing** | Break it on a platform we do not own, and tell us where |

## 💬 Connect

- **Org** — [Nova-Violet Role](https://github.com/Nova-Violet-Role), non-profit
- **Ko-fi** — [ko-fi.com/saimonokuma](https://ko-fi.com/saimonokuma) — buys time, never priority

## 📄 Licence & Philosophy

Our work is licensed **AGPL-3.0-or-later OR EUPL-1.2**. These are **copyleft**
licences, chosen deliberately over permissive ones: what is shared here cannot
be enclosed later, by anyone, including us.

🔓 **Open Source** — freely available, and freely *stays* available ·
🔬 **Falsifiable** — every claim re-runnable, reportable false ·
🎯 **Mission focused** — convergent frameworks, open to all

---

### ✨ Nova-Violet Role

*Decompiling Reality — One Framework at a Time*

[![Support Our Journey](https://img.shields.io/badge/❤️_Support_Our_Journey-Ko--fi-FF5E5B?style=for-the-badge)](https://ko-fi.com/saimonokuma)

© 2026 Nova-Violet Role · Non-Profit Organization — AGPL-3.0-or-later OR EUPL-1.2
