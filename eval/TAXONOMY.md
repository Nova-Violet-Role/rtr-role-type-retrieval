<!-- SPDX-FileCopyrightText: Nova-Violet Role ORG -->
<!-- SPDX-License-Identifier: EUPL-1.2 OR AGPL-3.0-or-later -->
# RTR versus RAG — taxionometry

How we measure, what each taxon means, and what the numbers honestly say.
Methodology first, scoreboard second. Reality is the judge.

## Corpora (both real, both local)

| Corpus | Files | Nature | Queries |
|---|---|---|---|
| `nt-studies` | 36 NestedText research notes | keyword-dense, overlapping vocab | 87 |
| `dita-spec` | 30 DITA 2.0 spec topics (XML) | long structured prose, element names | 77 |

Same index per run for every system — the ablation is purely on the retrieval
path. Each bench run uses an **isolated temp vector DB**
(`OPENCODE_MEMORY_DB_DIR`) so no run contaminates another.

## Systems

| System | Models | Hybrid | Expansion | Notes |
|---|---|---|---|---|
| `rag-dense` | bge-small, dense-only | no | no | classic single-vector RAG |
| `rag-hybrid` | bge-small + BM25 | yes | no | classic hybrid baseline |
| `rtr-arctic` | arctic-m + BM25 | yes | yes | ablation: arctic alone in RTR plumbing |
| `rtr-bge` | bge-base + BM25 | yes | yes | ablation: bge-base alone in RTR plumbing |
| `rtr-full` | arctic-m + bge-base, confidence-weighted RRF | yes | yes | the dual ensemble (default ship) |

## Taxons

| Taxon | Construction | What it isolates |
|---|---|---|
| T1 title-self | file's own distinctive title → file | general retrieval health |
| T2 rare-term | df==1 term → its file | exact/BM25 discrimination |
| T3 concept | mid-df keyword pair → file, top-3 | multi-term conceptual recall |
| T4 scoped | T1 subset under project filter | isolation (no cross-project bleed) |
| T5 paraphrase | true synonym phrase, absent corpus-wide (BM25 blind) + seeded lexikon | lexikon lane / semantic reach |
| T6 ident | filename stems + opaque digit/underscore ids | exact-match lane |

Metrics per taxon: recall@1, recall@3, MRR. Queries auto-derived (seed 20260928),
T5 pairs runtime-validated (source present, synonym absent, single owner).

## Scoreboard (`results/versus.json`, `versus.mmd`, `versus.svg`)

164 queries, 2 corpora, **0 query errors** on every system.

| System | R@1 | R@3 | MRR |
|---|---|---|---|
| **rtr-full** | **0.738** | **0.854** | **0.785** |
| rtr-arctic | 0.732 | 0.787 | 0.754 |
| rag-hybrid | 0.726 | 0.817 | 0.763 |
| rtr-bge | 0.713 | 0.817 | 0.755 |
| rag-dense | 0.530 | 0.701 | 0.607 |

rtr-full leads on all three combined metrics and decisively on dita-spec
(0.987/1.000/0.991); nt-studies is split with hybrid (R@1 by one query).
rtr-full also takes T6 identifiers (0.92). T5 (n=3) sits beyond bi-encoder
reach on this corpus (pure-cosine probe: expected docs rank ~16th) — tracked
future work for cross-encoder rerank, not a claim.

## Incident log (why you can trust these numbers)

Mid-campaign the bench caught itself lying: hundreds of
`Error finding id` failures on Arctic queries and run-to-run swings.
Root causes found and fixed in-product:
- Repeated delete/add churn across many short-lived ChromaDB clients desyncs
  the persistent HNSW segment → single shared client per DB path.
- Purge covered only configured models (105 orphan chunks accumulated in the
  bge-small collection) → purge fans out over all existing RTR collections,
  verifies each delete, plus a `janitor` command and `OPENCODE_MEMORY_DB_DIR`
  override. The live DB was janitored back to zero.
- Earlier dual/arctic numbers from contaminated runs were invalidated, not
  explained away. The scoreboard above is the clean re-run.

## Limits (read before quoting us)

- Two corpora, 164 queries, file-level self-supervision, no human labels.
  Measures retrieval mechanics, not graded relevance depth.
- Margins on nt-studies are narrow (±1–2 queries); dita-spec is decisive.
- Latency: RTR ~30–55ms p50 vs ~10ms classic (in-process; MCP adds transport).
- Dual-ensemble value concentrates in identifiers and cross-corpus robustness;
  per-model ablations ship in the same table so anyone can check our work.

## Reproduce

```
python eval/bench_versus.py   # needs installed rtr wheel for deps
```

Outputs `results/versus.json` + `versus.mmd`; render with
`npx -y @mermaid-js/mermaid-cli -i eval/versus.mmd -o eval/versus.svg`.
The release gate (`python -m execution.rtr_eval`) is separate and must stay
green: no ship on regression.
