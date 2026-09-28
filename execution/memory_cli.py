# SPDX-FileCopyrightText: Nova-Violet Role ORG
# SPDX-License-Identifier: EUPL-1.2 OR AGPL-3.0-or-later
import sys
import argparse
import time
import re
from pathlib import Path

# Add execution directory to sys.path to load memory_engine correctly
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from memory_engine import MemoryEngine
import memory_engine as mem
from tui_renderer import TUIRenderer


def format_search(query: str, project: str = None, k: int = 3, hybrid: bool = True):
    engine = MemoryEngine()
    try:
        results = engine.search(query, k=k, project=project, hybrid=hybrid)
    except Exception as e:
        print(
            TUIRenderer.render_panel(
                "Search Failure", [f"Error during search: {e}"], status="error"
            )
        )
        sys.exit(1)

    if not results:
        print(
            TUIRenderer.render_panel(
                "Search: 0 results",
                [f"No matching documents for: '{query}'"],
                status="search",
            )
        )
        return

    lines = [f'Query: "{query}"', f"Candidates returned: {len(results)}", ""]
    for idx, res in enumerate(results):
        meta = res["metadata"]
        lines.append(
            f"{idx + 1}. [{meta['rel_path']}] (Similarity: {res['score']:.4f})"
        )
        lines.append(f"   Excerpt: {res['document'][:180].replace(chr(10), ' ')}...")
        lines.append("")

    print(TUIRenderer.render_panel("Semantic Search Results", lines, status="search"))


def format_write(project: str, filename: str, content: str):
    engine = MemoryEngine()
    try:
        filepath, related = engine.write_wiki(project, filename, content)
    except Exception as e:
        print(
            TUIRenderer.render_panel(
                "Write Failure", [f"Error writing wiki entry: {e}"], status="error"
            )
        )
        sys.exit(1)

    lines = [
        f"Saved to: {Path(filepath).name}",
        f"Auto-linked relations: {len(related)}",
    ]
    for r in related:
        lines.append(f"  {r}")

    print(TUIRenderer.render_panel("Direct Wiki Write", lines, status="write"))


def format_index(force: bool = False, projects=None):
    start_time = time.time()
    engine = MemoryEngine()
    try:
        stats = engine.index_all(force=force, projects=projects or [])
    except Exception as e:
        print(
            TUIRenderer.render_panel(
                "Indexing Failure", [f"Error during indexing: {e}"], status="error"
            )
        )
        sys.exit(1)

    duration = time.time() - start_time
    lines = [
        f"Files Scanned: {stats['scanned']}",
        f"New Indexed:   {stats['new']}",
        f"Updated:       {stats['updated']}",
        f"Unchanged:     {stats['unchanged']}",
        f"Failed:        {stats['failed']}",
        f"Duration:      {duration:.2f}s",
    ]
    print(TUIRenderer.render_panel("Incremental Indexing", lines, status="index"))


def format_stats():
    engine = MemoryEngine()
    try:
        stats = engine.stats()
    except Exception as e:
        print(
            TUIRenderer.render_panel(
                "Stats Failure", [f"Error retrieving stats: {e}"], status="error"
            )
        )
        sys.exit(1)

    lines = [
        f"Total Files Indexed:  {stats['total_files']}",
        f"Total Vector Chunks:  {stats['total_chunks']}",
        f"Vector Database Size: {stats['db_size_mb']:.2f} MB",
        "",
        "Project Breakdown:",
    ]
    for proj, count in sorted(stats["projects"].items()):
        lines.append(f"  - {proj}: {count} files")

    print(TUIRenderer.render_panel("Memory Database Status", lines, status="stats"))


def format_init(project: str, variant: str = "wiki"):
    engine = MemoryEngine()
    try:
        res = engine.init_project(project, variant)
    except Exception as e:
        print(
            TUIRenderer.render_panel(
                "Bootstrap Failure", [f"Error: {e}"], status="error"
            )
        )
        sys.exit(1)

    lines = [res["message"], "", "Created Directories:"]
    for d in res["created_directories"]:
        lines.append(f"  📁 {d}")

    lines.append("")
    lines.append("Created Baseline Documents:")
    for f in res["created_files"]:
        lines.append(f"  📄 {f}")

    print(TUIRenderer.render_panel("Harness Bootstrapping", lines, status="bootstrap"))


def format_consolidate(project: str):
    engine = MemoryEngine()
    try:
        res = engine.consolidate_wiki(project)
    except Exception as e:
        print(
            TUIRenderer.render_panel(
                "Consolidation Failure", [f"Error: {e}"], status="error"
            )
        )
        sys.exit(1)

    if res.get("status") == "error":
        print(
            TUIRenderer.render_panel(
                "Consolidation Error", [res["message"]], status="error"
            )
        )
        sys.exit(1)

    lines = [res["message"], "", "Consolidation Log:"]
    if res.get("merged"):
        for m in res["merged"]:
            lines.append(f"  🧹 {m}")
    else:
        lines.append("  No duplicate files discovered.")

    print(TUIRenderer.render_panel("Corpus Consolidation", lines, status="organize"))


def format_session_start(project: str):
    engine = MemoryEngine()
    try:
        res = engine.restore_session_state(project)
    except Exception as e:
        print(
            TUIRenderer.render_panel(
                "Restoration Failure", [f"Error: {e}"], status="error"
            )
        )
        sys.exit(1)

    if res.get("status") == "error":
        print(
            TUIRenderer.render_panel(
                "Restoration Error", [res["message"]], status="error"
            )
        )
        sys.exit(1)

    lines = [
        f"Project context parsed successfully.",
        f'Query context: "{res["query_context"]}..."',
        "---",
        "Active Goals:",
    ]

    if res["active_goals"]:
        for g in res["active_goals"]:
            lines.append(f"  🎯 {g}")
    else:
        lines.append("  No active goals parsed.")

    lines.append("")
    lines.append("Blockers / Open Issues:")
    if res["blockers"]:
        for b in res["blockers"]:
            lines.append(f"  ⚠️ {b}")
    else:
        lines.append("  No blockers discovered.")

    lines.append("")
    lines.append("Recent Discoveries / Decisions:")
    if res["recent_discoveries"]:
        for d in res["recent_discoveries"]:
            lines.append(f"  💡 {d}")
    else:
        lines.append("  No recent discoveries recorded.")

    lines.append("")
    lines.append("Pre-fetched Relevant Memory Concepts:")
    if res["recommended_files"]:
        for f in res["recommended_files"]:
            lines.append(f"  📄 {f}")
    else:
        lines.append("  No high-signal historical concepts matched.")

    print(
        TUIRenderer.render_panel(f"Session Start: {project}", lines, status="restore")
    )


def format_explain(query: str, project: str = None, k: int = 3):
    engine = MemoryEngine()
    try:
        results = engine.explain_search(query, k=k, project=project)
    except Exception as e:
        print(
            TUIRenderer.render_panel("Explain Failure", [f"Error: {e}"], status="error")
        )
        sys.exit(1)

    if not results:
        print(
            TUIRenderer.render_panel(
                "Explain: 0 results",
                [f"No matching documents for: '{query}'"],
                status="explain",
            )
        )
        return

    lines = [f'Query: "{query}"', "Mathematical Score Breakdown Table:", ""]
    for idx, res in enumerate(results):
        lines.append(f"{idx + 1}. File: {res['rel_path']}")
        lines.append(f"   Final Score: {res['final']:.4f}")
        lines.append(f"   ┌────────────┬────────┬──────────┐")
        lines.append(f"   │ Metric     │ Value  │ Weight   │")
        lines.append(f"   ├────────────┼────────┼──────────┤")
        lines.append(f"   │ Dense      │ {res['dense']:.4f} │ x0.45    │")
        lines.append(f"   │ BM25       │ {res['bm25']:.4f} │ x0.30    │")
        lines.append(f"   │ Recency    │ {res['recency']:.4f} │ x0.15    │")
        lines.append(f"   │ Importance │ {res['importance']:.4f} │ x0.10    │")
        lines.append(f"   └────────────┴────────┴──────────┘")
        lines.append("")

    print(
        TUIRenderer.render_panel(
            "Observability: Explain Search", lines, status="explain"
        )
    )


def format_session_commit(project: str, chat: str = ""):
    engine = MemoryEngine()
    try:
        res = engine.session_commit(project, chat)
    except Exception as e:
        print(
            TUIRenderer.render_panel(
                "Session Commit Failure", [f"Error: {e}"], status="error"
            )
        )
        sys.exit(1)

    events = res.get("knowledge_events", [])
    event_counts = {}
    for ev in events:
        t = ev.get("type", "observation")
        event_counts[t] = event_counts.get(t, 0) + 1

    lines = [
        f"Master session commit succeeded.",
        f"Report path: {Path(res['report_path']).name}",
        "",
        "Extracted Knowledge Events:",
    ]
    if event_counts:
        for t, count in sorted(event_counts.items()):
            lines.append(f"  - {t.title()}: {count} events")
    else:
        lines.append("  - None")

    lines.extend(
        [
            "",
            "Graph & Index Evolution:",
            f"  Materialized:      {len(res['materialized_log'])} concepts",
            f"  Bidirectional links: {res['links_updated']} links updated",
            f"  Vector Index stats: {res['index_stats']['new']} new, {res['index_stats']['updated']} updated chunks",
        ]
    )
    print(TUIRenderer.render_panel("Session Commit Complete", lines, status="ok"))


def format_reflect(project: str, chat: str = ""):
    engine = MemoryEngine()
    try:
        res = engine.reflect_session(project, chat)
    except Exception as e:
        print(
            TUIRenderer.render_panel(
                "Reflection Failure", [f"Error: {e}"], status="error"
            )
        )
        sys.exit(1)

    events = res.get("knowledge_events", [])
    lines = [
        f"Session reflection generated report: {Path(res['report_path']).name}",
        "",
        "Knowledge Events Extracted:",
    ]
    for ev in events if events else []:
        lines.append(
            f"  - [{ev.get('type', 'observation').upper()}] {ev.get('concept', 'General')} — {ev.get('evidence', '')}"
        )
    if not events:
        lines.append("  - None")

    print(TUIRenderer.render_panel("Session Reflection", lines, status="ok"))


def format_materialize(project: str):
    engine = MemoryEngine()
    try:
        res = engine.materialize_concepts(project)
    except Exception as e:
        print(
            TUIRenderer.render_panel(
                "Materialization Failure", [f"Error: {e}"], status="error"
            )
        )
        sys.exit(1)

    if res.get("status") == "error":
        print(
            TUIRenderer.render_panel(
                "Materialization Error", [res["message"]], status="error"
            )
        )
        sys.exit(1)

    lines = [
        res.get("message", "Concept materialization pipeline completed."),
        "",
        "Materialization Log:",
    ]
    if res.get("materialized"):
        for m in res["materialized"]:
            lines.append(f"  {m}")
    else:
        lines.append("  No new or modified concepts found in reports to materialize.")

    print(TUIRenderer.render_panel("Concept Materialization", lines, status="ok"))


def format_update_graph(project: str):
    engine = MemoryEngine()
    try:
        res = engine.update_concept_graph(project)
    except Exception as e:
        print(
            TUIRenderer.render_panel(
                "Graph Update Failure", [f"Error: {e}"], status="error"
            )
        )
        sys.exit(1)

    if res.get("status") == "error":
        print(
            TUIRenderer.render_panel(
                "Graph Update Error", [res["message"]], status="error"
            )
        )
        sys.exit(1)

    lines = [
        res.get("message", "Graph structure updated."),
        f"Total links added or normalized: {res.get('links_added_or_normalized', 0)}",
    ]
    print(
        TUIRenderer.render_panel(
            "Bidirectional Concept Graph Builder", lines, status="organize"
        )
    )


def format_lexikon_add(project: str, term: str, definition: str = "", aliases=None):
    # SPDX-FileCopyrightText: Nova-Violet Role ORG
    # SPDX-License-Identifier: EUPL-1.2 OR AGPL-3.0-or-later
    engine = MemoryEngine()
    res = engine.add_lexikon_term(project, term, definition, aliases or [])
    status = "ok" if res.get("status") == "success" else "error"
    print(TUIRenderer.render_panel("Lexikon", [res.get("message", "")], status=status))


def format_lexikon_show(project: str):
    engine = MemoryEngine()
    lex = engine.get_lexikon(project)
    if not lex:
        print(TUIRenderer.render_panel("Lexikon", ["No terms yet."], status="info"))
        return
    lines = [f"Terms: {len(lex)}", ""]
    for key in sorted(lex)[:20]:
        e = lex[key] if isinstance(lex[key], dict) else {}
        aliases = ", ".join(e.get("aliases", []) or [])
        extra = f" (aka: {aliases})" if aliases else ""
        lines.append(f"  - {e.get('term', key)}{extra} [{e.get('source', '?')}]")
    if len(lex) > 20:
        lines.append(f"  ... and {len(lex) - 20} more")
    print(TUIRenderer.render_panel(f"Lexikon: {project}", lines, status="stats"))


def format_lexikon_build(project: str):
    engine = MemoryEngine()
    res = engine.build_lexikon(project)
    status = "ok" if res.get("status") == "success" else "error"
    print(
        TUIRenderer.render_panel(
            "Lexikon Build", [res.get("message", "")], status=status
        )
    )


def format_janitor(dry_run: bool = False):
    # SPDX-FileCopyrightText: Nova-Violet Role ORG
    # SPDX-License-Identifier: EUPL-1.2 OR AGPL-3.0-or-later
    engine = MemoryEngine()
    res = engine.janitor(dry_run=dry_run)
    lines = [f"Total chunks freed: {res['freed']}", ""]
    for name, info in sorted(res["collections"].items()):
        if "error" in info:
            lines.append(f"  - {name}: ERROR {info['error']}")
        else:
            lines.append(f"  - {name}: {info['chunks']} chunks, freed {info['freed']}")
    print(TUIRenderer.render_panel("Janitor", lines, status="organize"))


def main():
    parser = argparse.ArgumentParser(description="Local Memory RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Search subparser
    search_parser = subparsers.add_parser(
        "search", help="Semantic search in knowledge base"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--project", type=str, default=None, help="Filter results by project name"
    )
    search_parser.add_argument(
        "--k", type=int, default=3, help="Number of results to return"
    )
    search_parser.add_argument(
        "--dense-only", action="store_true", help="Disable sparse BM25 reranking"
    )

    # Write subparser
    write_parser = subparsers.add_parser(
        "write", help="Write new wiki entry to a project"
    )
    write_parser.add_argument("project", type=str, help="Target project name")
    write_parser.add_argument(
        "filename", type=str, help="Descriptive filename (e.g. buffer_api_quirks)"
    )
    write_parser.add_argument("content", type=str, help="Markdown content of the entry")

    # Index subparser
    index_parser = subparsers.add_parser("index", help="Rebuild or update search index")
    index_parser.add_argument(
        "--force", action="store_true", help="Force re-indexing of all documents"
    )
    index_parser.add_argument(
        "--project",
        action="append",
        default=[],
        help="Extra project ref to resolve+register first (repeatable)",
    )

    # Stats subparser
    subparsers.add_parser("stats", help="Show index and database statistics")

    # Init project subparser
    init_parser = subparsers.add_parser(
        "init", help="Bootstrap a new Harness project directory"
    )
    init_parser.add_argument("project", type=str, help="Project name to initialize")
    init_parser.add_argument(
        "--variant",
        type=str,
        choices=["wiki", "design"],
        default="wiki",
        help="Naming variant (wiki or design)",
    )

    # Consolidate subparser
    consolidate_parser = subparsers.add_parser(
        "consolidate", help="Consolidate duplicate wiki markdown concepts"
    )
    consolidate_parser.add_argument(
        "project", type=str, help="Target project to clean up"
    )

    # Session start subparser
    start_parser = subparsers.add_parser(
        "session-start", help="Auto-retrieve memory based on session startup files"
    )
    start_parser.add_argument("project", type=str, help="Project to inspect")

    # Explain subparser
    explain_parser = subparsers.add_parser(
        "explain", help="Explain search query mathematical score breakdown"
    )
    explain_parser.add_argument("query", type=str, help="Search query")
    explain_parser.add_argument(
        "--project", type=str, default=None, help="Filter results by project name"
    )
    explain_parser.add_argument(
        "--k", type=int, default=3, help="Number of results to return"
    )

    # Session commit subparser
    commit_parser = subparsers.add_parser(
        "session-commit",
        help="Master session end: reflect, materialize, link, and index",
    )
    commit_parser.add_argument("project", type=str, help="Target project name")
    commit_parser.add_argument(
        "--chat",
        type=str,
        default="",
        help="Optional raw conversation or pre-structured learnings to process",
    )

    # Reflect subparser
    reflect_parser = subparsers.add_parser(
        "reflect", help="Generate structured reflection report from session log"
    )
    reflect_parser.add_argument("project", type=str, help="Target project name")
    reflect_parser.add_argument(
        "--chat", type=str, default="", help="Session log or pre-structured learnings"
    )

    # Materialize subparser
    materialize_parser = subparsers.add_parser(
        "materialize",
        help="Materialize newest session learnings into kebab-case concepts",
    )
    materialize_parser.add_argument("project", type=str, help="Target project name")

    # Update graph subparser
    graph_parser = subparsers.add_parser(
        "update-graph", help="Update reciprocal bidirectional concept graph links"
    )
    graph_parser.add_argument("project", type=str, help="Target project name")

    # Lexikon subparsers (RTR pillar 3)
    lex_add = subparsers.add_parser("lexikon-add", help="Hand-add a Lexikon term")
    lex_add.add_argument("project", type=str, help="Project name (or _global)")
    lex_add.add_argument("term", type=str, help="Canonical term")
    lex_add.add_argument("--definition", type=str, default="")
    lex_add.add_argument("--alias", action="append", default=[])

    lex_show = subparsers.add_parser("lexikon-show", help="Show merged Lexikon terms")
    lex_show.add_argument("project", type=str, help="Project name (or _global)")

    lex_build = subparsers.add_parser(
        "lexikon-build", help="Derive project Lexikon from registry and sessions"
    )
    lex_build.add_argument("project", type=str, help="Target project name")

    jan_parser = subparsers.add_parser(
        "janitor", help="Delete orphan chunks whose files are gone"
    )
    jan_parser.add_argument(
        "--dry-run", action="store_true", help="Report only, delete nothing"
    )

    args = parser.parse_args()

    if args.command == "search":
        format_search(
            args.query, project=args.project, k=args.k, hybrid=not args.dense_only
        )
    elif args.command == "write":
        format_write(args.project, args.filename, args.content)
    elif args.command == "index":
        format_index(force=args.force, projects=args.project)
    elif args.command == "stats":
        format_stats()
    elif args.command == "init":
        format_init(args.project, variant=args.variant)
    elif args.command == "consolidate":
        format_consolidate(args.project)
    elif args.command == "session-start":
        format_session_start(args.project)
    elif args.command == "explain":
        format_explain(args.query, project=args.project, k=args.k)
    elif args.command == "session-commit":
        format_session_commit(args.project, chat=args.chat)
    elif args.command == "reflect":
        format_reflect(args.project, chat=args.chat)
    elif args.command == "materialize":
        format_materialize(args.project)
    elif args.command == "update-graph":
        format_update_graph(args.project)
    elif args.command == "lexikon-add":
        format_lexikon_add(args.project, args.term, args.definition, args.alias)
    elif args.command == "lexikon-show":
        format_lexikon_show(args.project)
    elif args.command == "lexikon-build":
        format_lexikon_build(args.project)
    elif args.command == "janitor":
        format_janitor(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
