<!-- SPDX-FileCopyrightText: Nova-Violet Role ORG -->
<!-- SPDX-License-Identifier: EUPL-1.2 OR AGPL-3.0-or-later -->
# rtr-role-type-retrieval — dense usage manual

RTR (Role Type Retrieval) 0.3.0: RRF-heuristic, DTD-semantic-capable persistent
memory for coding agents. Dual ONNX ensemble on CPU, zero API cost, local only.
By [Nova-Violet Role](https://github.com/Nova-Violet-Role) — non-profit.
**AGPL-3.0-or-later OR EUPL-1.2.** Support: [ko-fi.com/saimonokuma](https://ko-fi.com/saimonokuma).

## 1. Run it (pick one — no clone, no venv, no sync in any of them)

```bash
# A. Zero-install, every session (recommended): uvx caches + isolates automatically
uvx --from rtr-role-type-retrieval rtr-mcp            # MCP server over stdio
uvx --from rtr-role-type-retrieval rtr-cli stats      # CLI

# B. Persistent tool install
uv tool install rtr-role-type-retrieval
rtr-mcp
rtr-cli index

# C. Classic pip (you manage the venv)
pip install rtr-role-type-retrieval
python -m execution.memory_mcp_server
```

First run downloads embedding models once (~650 MB: Arctic-m 768 + bge-base 768)
into `FASTEMBED_CACHE_PATH` — **pin it to persistent storage** (default is the
system temp dir and can be wiped). Models then load warm every session.

## 2. Wire it into your agent CLI (model-agnostic: works with any LLM, local or API)

The server is plain MCP-over-stdio. It never calls an LLM itself — embeddings
are local ONNX — so it works with Claude, GPT, GLM, Kimi, Ollama-driven harnesses,
anything with an MCP client.

```jsonc
// opencode.jsonc  — global config, automatic every session
"mcp": {
  "memory": {
    "type": "local",
    "command": ["uvx", "--from", "rtr-role-type-retrieval", "rtr-mcp"],
    "enabled": true,
    "timeout": 180000,   // cold model load on very first start
    "environment": {
      "OPENCODE_MEMORY_RTR_PROFILE": "performance",  // balanced 4+4 | performance 8+8
      "OPENCODE_MEMORY_PROJECTS_DIR": "C:/Users/you/projects",  // ;-separated multi-root
      "FASTEMBED_CACHE_PATH": "C:/Users/you/.config/opencode/memory/.fastembed_cache"
    }
  }
}
```

```json
// Claude Code / any MCP-stdIO client: same command, same env
{ "mcpServers": { "rtr-memory": {
  "command": "uvx",
  "args": ["--from", "rtr-role-type-retrieval", "rtr-mcp"],
  "env": { "OPENCODE_MEMORY_RTR_PROFILE": "balanced" }
} } }
```

```bash
# Local-LLM harnesses without MCP: use the CLI as tools
rtr-cli session-start /path/to/project     # restore goals/blockers/prefetch
rtr-cli search "buffer API quirks" --project myproj --k 5
rtr-cli session-commit /path/to/project --chat "...session log..."
```

## 3. Project bootstrap (works anywhere — absolute paths auto-register)

```bash
rtr-cli init /any/where/myproj              # harness tree; dir joins discovery roots
rtr-cli init myproj --variant design        # design_wiki flavour
rtr-cli index                               # delta index (hash-guarded, per-model collections)
rtr-cli index --project /any/where/myproj   # force-register then index
```

Indexed suffixes: `.md`, `.nt`, `.yaml`, `.yml` (`OPENCODE_MEMORY_FILE_SUFFIXES`
overrides). Sources of truth stay in each project's `directives/`; the vector
store (`~/.config/opencode/memory/.local_vector_db`) is a derived global cache,
partitioned per project. Reopen the same project name → memory restored.

## 4. Every environment knob (one global local config)

| Variable | Default | Meaning |
|---|---|---|
| `OPENCODE_MEMORY_EMBED_MODELS` | `Snowflake/snowflake-arctic-embed-m,BAAI/bge-base-en-v1.5` | comma-separated fastembed models; one ChromaDB collection each |
| `OPENCODE_MEMORY_RTR_PROFILE` | `balanced` | `balanced` = 4+4 threads, `performance` = 8+8 |
| `OPENCODE_MEMORY_EMBED_THREADS` | profile default | int for all models, or per-model `arctic:4,bge:8` |
| `OPENCODE_MEMORY_PROJECTS_DIR` | `~/projects` | `os.pathsep`-separated discovery roots |
| `OPENCODE_MEMORY_FILE_SUFFIXES` | `.md,.nt,.yaml,.yml` | indexed file types (DtD polyglot) |
| `OPENCODE_MEMORY_RRF_K` | `60` | RRF dampening; fuses rankings, never raw scores |
| `OPENCODE_MEMORY_PRF` | `0` | `1` = pseudo-relevance-feedback second pass |
| `OPENCODE_MEMORY_DB_DIR` | `<memory>/.local_vector_db` | override vector DB home (bench isolation) |
| `FASTEMBED_CACHE_PATH` | system temp | **override this** — model home, persistent |
| `PYTHONIOENCODING` | — | set `utf-8` on Windows for the TUI panels |

## 5. Tools exposed (MCP) and CLI map

`memory_session_start` · `memory_session_commit` · `memory_reflect_session` ·
`memory_materialize_concepts` · `memory_update_graph` · `memory_search` ·
`memory_explain_search` · `memory_discover` · `memory_list` · `memory_delete` ·
`memory_inbox_write` · `memory_inbox_read` · `memory_write` · `memory_organize` ·
`memory_init_project` · `memory_consolidate` · `memory_stats` · `memory_index` ·
`memory_lexikon_add` · `memory_lexikon_show` · `memory_lexikon_build`.
CLI mirrors all of them (`rtr-cli <verb> --help` each).

Lexikon: per-project glossary auto-derived from registry + sessions, plus a
hand-curated `_global` user lexikon (`rtr-cli lexikon-add _global "term"
--definition "..." --alias ...`) — lookup-first expansion, no inference.
Maintenance: `rtr-cli janitor [--dry-run]` deletes orphan chunks whose files
are gone (also `memory_janitor`).

## 6. How retrieval works (the 30-second version)

Per query: original + lexikon/alias + workspace-delta sub-queries (+ optional
PRF) × per-model collections → RRF rank fusion → min-max normalized dense into
`0.45·dense + 0.30·BM25 + 0.15·recency + 0.10·importance`, plus a bounded
CDATA exact-match lane for code/ids/hashes/paths. Spans are DTD-typed at index
(`PCDATA`/`CDATA`/`NDATA` + NOTATION recorded, NDATA masked pre-embedding).
Same-name projects across roots resolve first-match; absolute paths always exact.

## 6b. Overseeding — bigger turbos welcome (bi-turbo philosophy)

RTR runs twin powerplants — Arctic precision plus bge recall — fused by RRF,
the way rally engineers ran bi-turbo: not because one turbo wasn't enough,
but out of love for solving the paradigm. Discoveries live there.

Any fastembed model slots in via one variable, and because every model owns
its collection, **overseeding is zero-migration**: point, re-index, the old
collections janitor away. Measured guidance from our own bench:

| Your hardware | Suggestion | Why |
|---|---|---|
| 8 cores | default dual (arctic-m + bge-base), `balanced` 4+4 | the efficiency crown |
| 12–16 cores | same dual, `performance` 8+8 | headroom well spent |
| Strong CPU + RAM | `BAAI/bge-large-en-v1.5` and/or `Snowflake/snowflake-arctic-embed-l` (1024d) | precision ceiling rises with dim |
| Keyword-heavy corpus | try single `BAAI/bge-base-en-v1.5` | our bench: BM25-adjacent turf favors focus |

```bash
OPENCODE_MEMORY_EMBED_MODELS="BAAI/bge-large-en-v1.5,Snowflake/snowflake-arctic-embed-l"
rtr-cli index          # new collections fill; old ones: rtr-cli janitor
python eval/bench_versus.py   # prove it on your own corpus before you believe us
```

Honest note: bigger is not always better — on keyword-dense corpora our bench
favors focus over size. Measure, don't assume. If your turbo wins, tell us;
falsification is a contribution here.

## 7. Verify it (falsifiable or it didn't happen)

```bash
uvx --from rtr-role-type-retrieval python -m execution.rtr_eval  # release gate
python -m unittest execution.test_memory                          # unit suite
python eval/bench_versus.py                                       # RTR-vs-RAG bench, real docs
```

`eval/results/` ships the scoreboard JSON, the Mermaid source, and the SVG.

## 8. Troubleshooting

- **Slow first start** — model download (~650 MB); later starts are warm. Raise MCP `timeout` once.
- **HF rate limits** — set `HF_TOKEN`; downloads resume.
- **Windows paths** — forward slashes work everywhere in this config.
- **Empty results** — project must live under a discovery root or be touched once by absolute path (auto-registers); check `rtr-cli stats`.
- **ChromaDB lock errors** — one writer at a time; close stale sessions.

## 9. Licence

Dual-licensed, copyleft, binding: **AGPL-3.0-or-later OR EUPL-1.2**
(`LICENSE-AGPL-3.0.txt`, `LICENSE-EUPL-1.2.txt`, `NOTICE`). Upstream note: the
retrieval core derives in part from `github.com/xpajonx/opencode-memory`, which
ships no licence file — public redistribution of the combined work awaits
upstream clarification.
