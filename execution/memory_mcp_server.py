# SPDX-FileCopyrightText: Nova-Violet Role ORG
# SPDX-License-Identifier: EUPL-1.2 OR AGPL-3.0-or-later
import sys
import os
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.absolute()))

import memory_engine as mem
from memory_engine import MemoryEngine
from fastmcp import FastMCP
from tui_renderer import TUIRenderer

engine = MemoryEngine()
_ = engine.models  # warm every configured ONNX model (dual ensemble) at startup

mcp = FastMCP("memory")


def _wiki_paths(project: str) -> dict | str:
    """Detect which wiki naming convention a project uses.
    Returns {inbox, concepts} or an error string."""
    directives = mem.resolve_project_dir(project) / "directives"
    if not directives.exists():
        return f"Error: No directives/ folder in {project}"

    design_inbox = directives / "design_wiki.md"
    design_concepts = directives / "design_wiki_concepts"
    wiki_inbox = directives / "wiki_inbox.md"
    wiki_concepts = directives / "wiki_concepts"

    if design_inbox.exists() or design_concepts.exists():
        return {
            "inbox": design_inbox,
            "concepts": design_concepts,
            "variant": "design_wiki",
        }

    return {
        "inbox": wiki_inbox,
        "concepts": wiki_concepts,
        "variant": "wiki",
    }


@mcp.tool()
def memory_discover(project: str) -> str:
    """Discover wiki files and naming convention for a project."""
    paths = _wiki_paths(project)
    if isinstance(paths, str):
        return TUIRenderer.render_panel(f"Discover: {project}", [paths], status="info")

    inbox = paths["inbox"]
    concepts = paths["concepts"]

    lines = []
    if inbox.exists():
        size = len(inbox.read_text())
        lines.append(f"Inbox: {inbox.name} ({size} chars)")
    else:
        lines.append(f"Inbox: {inbox.name} (not yet created)")

    if concepts.exists():
        subdirs = sorted(d.name for d in concepts.iterdir() if d.is_dir())
        if subdirs:
            lines.append(f"Concepts ({len(subdirs)} subcategories):")
            for s in subdirs:
                files = [f.name for f in (concepts / s).iterdir() if f.suffix == ".md"]
                lines.append(f"  {s}/ ({len(files)} files)")
        else:
            lines.append("Concepts: (empty)")
    else:
        lines.append("Concepts: (not yet created)")

    lines.append(f"Convention variant: {paths['variant']}")
    return TUIRenderer.render_panel(f"Discover: {project}", lines, status="stats")


@mcp.tool()
def memory_search(query: str, k: int = 3, project: str = "") -> str:
    """Semantic & Hybrid search across indexed wikis."""
    try:
        results = engine.search(query, k=k, project=project or None)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Search Failure", [f"Error: {e}"], status="error"
        )

    if not results:
        return TUIRenderer.render_panel(
            "Search: 0 results",
            [f"No matching documents for: '{query}'"],
            status="search",
        )

    lines = [f'Query: "{query}"', f"Candidates returned: {len(results)}", ""]

    for idx, res in enumerate(results):
        meta = res["metadata"]
        lines.append(
            f"{idx + 1}. [{meta['rel_path']}] (Similarity: {res['score']:.4f})"
        )
        lines.append(f"   Title: {meta.get('title', 'Unknown')}")
        lines.append(f"   Excerpt: {res['document'][:180].replace(chr(10), ' ')}...")
        lines.append("")

    return TUIRenderer.render_panel("Semantic Search Results", lines, status="search")


@mcp.tool()
def memory_list(project: str = "") -> str:
    """List indexed memory files."""
    try:
        registry = engine.load_registry()
    except Exception as e:
        return TUIRenderer.render_panel("List Failure", [f"Error: {e}"], status="error")

    if not registry:
        return TUIRenderer.render_panel(
            "List: 0 files", ["No files indexed in the local database."], status="info"
        )

    files_by_project = {}
    want = mem.project_filter_key(project) if project else ""
    for file_path, hash_val in registry.items():
        proj, rel, _ = mem.describe_indexed_path(file_path)
        if project and proj != want:
            continue
        files_by_project.setdefault(proj, []).append((rel, hash_val[:8]))

    if not files_by_project:
        return TUIRenderer.render_panel(
            f"List: {project}",
            [f"No files indexed for project '{project}'"],
            status="info",
        )

    lines = []
    for proj in sorted(files_by_project):
        lines.append(f"📁 {proj}/ ({len(files_by_project[proj])} files):")
        for rel_path, h in sorted(files_by_project[proj]):
            lines.append(f"  📄 {rel_path} (sha: {h})")

    return TUIRenderer.render_panel("Indexed Files List", lines, status="stats")


@mcp.tool()
def memory_delete(path: str) -> str:
    """Delete a memory file from storage and index."""
    p = Path(path)
    full_path = p if p.is_absolute() else mem.PROJECTS_DIR / path
    if not full_path.exists():
        return TUIRenderer.render_panel(
            "Delete Failure", [f"Error: {path} does not exist."], status="error"
        )
    try:
        for col in engine.rtr_collections().values():
            existing = col.get(where={"source": str(full_path)})
            if existing and existing["ids"]:
                col.delete(ids=existing["ids"])
        full_path.unlink()
        registry = engine.load_registry()
        if str(full_path) in registry:
            del registry[str(full_path)]
            engine.save_registry(registry)
        return TUIRenderer.render_panel(
            "Memory Deleted",
            [f"Successfully removed {path} from filesystem and DB."],
            status="ok",
        )
    except Exception as e:
        return TUIRenderer.render_panel(
            "Delete Failure", [f"Error deleting {path}: {e}"], status="error"
        )


@mcp.tool()
def memory_inbox_write(project: str, content: str) -> str:
    """Append a learning block to the project's memory inbox."""
    paths = _wiki_paths(project)
    if isinstance(paths, str):
        return TUIRenderer.render_panel("Inbox Append Failure", [paths], status="error")

    inbox = paths["inbox"]
    inbox.parent.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M")
    if inbox.exists():
        inbox.write_text(
            inbox.read_text() + f"\n## Ingest {timestamp}\n\n{content.strip()}\n"
        )
    else:
        label = "Design Wiki" if paths["variant"] == "design_wiki" else "Wiki"
        inbox.write_text(
            f"# {label} Inbox\n## Ingest {timestamp}\n\n{content.strip()}\n"
        )

    return TUIRenderer.render_panel(
        "Inbox Appended",
        [f"Appended entry to {inbox.name} successfully."],
        status="write",
    )


@mcp.tool()
def memory_inbox_read(project: str) -> str:
    """Read the current project's memory inbox content."""
    paths = _wiki_paths(project)
    if isinstance(paths, str):
        return TUIRenderer.render_panel("Inbox Read Failure", [paths], status="error")

    if not paths["inbox"].exists():
        return TUIRenderer.render_panel(
            "Inbox Empty", ["Inbox file does not exist yet."], status="info"
        )

    lines = paths["inbox"].read_text().split("\n")
    return TUIRenderer.render_panel(
        f"Inbox: {project}",
        lines[:15] + (["... (truncated)"] if len(lines) > 15 else []),
        status="ok",
    )


@mcp.tool()
def memory_write(project: str, filename: str, content: str) -> str:
    """Write concept file directly and compute relations."""
    paths = _wiki_paths(project)
    if isinstance(paths, str):
        return TUIRenderer.render_panel("Write Failure", [paths], status="error")

    concepts_dir = paths["concepts"]
    concepts_dir.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".md"):
        filename += ".md"
    filepath = concepts_dir / filename
    filepath.write_text(content.strip())

    related_links = []
    try:
        query = filename.replace(".md", "").replace("_", " ")
        results = engine.search(query=query, k=4, project=None)
        seen = set()
        for res in results:
            src = Path(res["metadata"]["source"])
            if src == filepath or src in seen:
                continue
            seen.add(src)
            if res["score"] > 0.50:
                rel = os.path.relpath(src, start=concepts_dir).replace("\\", "/")
                related_links.append(
                    f"- [{src.name}]({rel}) (Similarity: {res['score']:.2f})"
                )
    except Exception:
        pass

    if related_links:
        filepath.write_text(
            filepath.read_text()
            + "\n\n## Related Knowledge\n"
            + "\n".join(related_links)
            + "\n"
        )

    s = engine.index_all()

    lines = [
        f"Saved to: {filepath.name}",
        f"Index status: {s['new']} new, {s['updated']} updated",
        f"Auto-linked relations: {len(related_links)}",
    ]
    for r in related_links:
        lines.append(f"  {r}")

    return TUIRenderer.render_panel("Direct Wiki Write", lines, status="write")


@mcp.tool()
def memory_organize(project: str) -> str:
    """Extract, link, and structure concepts from the project's inbox."""
    paths = _wiki_paths(project)
    if isinstance(paths, str):
        return TUIRenderer.render_panel("Organize Failure", [paths], status="error")

    inbox = paths["inbox"]
    concepts_dir = paths["concepts"]
    concepts_dir.mkdir(parents=True, exist_ok=True)

    if not inbox.exists():
        return TUIRenderer.render_panel(
            "Organize", ["No inbox found to organize."], status="info"
        )

    source = inbox.read_text().strip()
    if not source or len(source) < 20:
        return TUIRenderer.render_panel("Organize", ["Inbox is empty."], status="info")

    sections = re.split(r"\n##\s+", source)
    if len(sections) <= 1:
        return TUIRenderer.render_panel(
            "Organize", ["No concept sections found in inbox."], status="info"
        )

    sections.pop(0)
    processed = []

    for section in sections:
        lines = section.strip().split("\n")
        title = lines[0].strip() if lines else "untitled"
        body = "\n".join(lines[1:]).strip()
        if not title or not body:
            continue

        safe_name = (
            re.sub(r"[^a-zA-Z0-9_\-\s]", "", title).strip().replace(" ", "_").lower()
        )
        if not safe_name:
            safe_name = f"concept_{int(time.time())}"

        cats = sorted(d.name for d in concepts_dir.iterdir() if d.is_dir())
        best_cat = None
        best_score = 0.0
        if cats:
            try:
                cat_embs = list(
                    engine.model.embed(
                        [f"concept about {c.replace('_', ' ')}" for c in cats]
                    )
                )
                concept_emb = list(engine.model.embed([f"{title}: {body[:200]}"]))[0]
                for i, emb in enumerate(cat_embs):
                    sim = sum(a * b for a, b in zip(concept_emb, emb)) / (
                        (sum(a * a for a in concept_emb) ** 0.5)
                        * (sum(b * b for b in emb) ** 0.5)
                        + 1e-10
                    )
                    if sim > best_score:
                        best_score = sim
                        best_cat = cats[i]
            except Exception:
                pass

        cat_dir = concepts_dir / (
            best_cat if best_cat and best_score > 0.55 else f"topic_{safe_name}"
        )
        cat_dir.mkdir(parents=True, exist_ok=True)

        existing_files = sorted(cat_dir.glob("*.md"))

        related_links = []
        try:
            results = engine.search(
                query=f"{title}: {body[:200]}", k=3, project=project
            )
            seen = set()
            for res in results:
                src = Path(res["metadata"]["source"])
                if src in seen:
                    continue
                seen.add(src)
                if res["score"] > 0.60:
                    rel = os.path.relpath(src, start=cat_dir).replace("\\", "/")
                    related_links.append(f"- [{src.stem.replace('_', ' ')}]({rel})")
        except Exception:
            pass

        content = f"# {title}\n\n{body}\n"
        if related_links:
            content += "\n## Related\n" + "\n".join(related_links) + "\n"
        content += (
            f"\n---\n*Organized from inbox on {time.strftime('%Y-%m-%d %H:%M')}*\n"
        )

        concept_file = cat_dir / f"{safe_name}.md"
        concept_file.write_text(content)

        link_ref = f"[[{safe_name}]]"
        for ef in existing_files:
            if link_ref not in ef.read_text():
                ef.write_text(ef.read_text().rstrip() + f"\n- {link_ref} — {title}\n")

        processed.append(f"  {concept_file.name} -> {cat_dir.name}/")

    label = "Design Wiki" if paths["variant"] == "design_wiki" else "Wiki"
    inbox.write_text(f"# {label} Inbox\n")
    s = engine.index_all()

    lines = (
        [
            f"Inbox cleared and filed.",
            f"Filed concepts ({len(processed)}):",
        ]
        + processed
        + [f"Index updated: {s['new']} new, {s['updated']} updated"]
    )
    return TUIRenderer.render_panel(
        "Graph Maintainer Complete", lines, status="organize"
    )


@mcp.tool()
def memory_stats() -> str:
    """Retrieve corpus statistics."""
    try:
        stats = engine.stats()
    except Exception as e:
        return TUIRenderer.render_panel(
            "Stats Failure", [f"Error: {e}"], status="error"
        )

    lines = [
        f"Total Files Indexed:  {stats['total_files']}",
        f"Total Vector Chunks:  {stats['total_chunks']}",
        f"Vector Database Size: {stats['db_size_mb']:.2f} MB",
        "",
        "Project Breakdown:",
    ]
    for proj, count in sorted(stats["projects"].items()):
        lines.append(f"  - {proj}: {count} files")

    return TUIRenderer.render_panel("Memory Database Status", lines, status="stats")


@mcp.tool()
def memory_index(force: bool = False, project: str = "") -> str:
    """Force incremental re-indexing (project ref also registers out-of-roots dirs)."""
    start = time.time()
    try:
        s = engine.index_all(force=force, projects=[project] if project else [])
    except Exception as e:
        return TUIRenderer.render_panel(
            "Indexing Failure", [f"Error: {e}"], status="error"
        )

    duration = time.time() - start
    lines = [
        f"Files Scanned: {s['scanned']}",
        f"New Indexed:   {s['new']}",
        f"Updated:       {s['updated']}",
        f"Unchanged:     {s['unchanged']}",
        f"Failed:        {s['failed']}",
        f"Duration:      {duration:.2f}s",
    ]
    return TUIRenderer.render_panel("Incremental Indexing", lines, status="index")


@mcp.tool()
def memory_init_project(project: str, variant: str = "wiki") -> str:
    """Bootstrap standard Harness Engineering structures in a project directory."""
    try:
        res = engine.init_project(project, variant)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Bootstrap Failure", [f"Error: {e}"], status="error"
        )

    lines = [res["message"], "", "Created Directories:"]
    for d in res["created_directories"]:
        lines.append(f"  📁 {d}")

    lines.append("")
    lines.append("Created Baseline Documents:")
    for f in res["created_files"]:
        lines.append(f"  📄 {f}")

    return TUIRenderer.render_panel("Harness Bootstrapping", lines, status="bootstrap")


@mcp.tool()
def memory_consolidate(project: str) -> str:
    """Clean, merge, and de-duplicate redundant concept markdown files."""
    try:
        res = engine.consolidate_wiki(project)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Consolidation Failure", [f"Error: {e}"], status="error"
        )

    if res.get("status") == "error":
        return TUIRenderer.render_panel(
            "Consolidation Error", [res["message"]], status="error"
        )

    lines = [res["message"], "", "Consolidation Log:"]
    if res.get("merged"):
        for m in res["merged"]:
            lines.append(f"  🧹 {m}")
    else:
        lines.append("  No duplicate files discovered.")

    return TUIRenderer.render_panel("Corpus Consolidation", lines, status="organize")


@mcp.tool()
def memory_session_start(project: str) -> str:
    """Automate Session Start (Phase 1): Restores previous working state using Harness state logs."""
    try:
        res = engine.restore_session_state(project)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Restoration Failure", [f"Error: {e}"], status="error"
        )

    if res.get("status") == "error":
        return TUIRenderer.render_panel(
            "Restoration Error", [res["message"]], status="error"
        )

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

    return TUIRenderer.render_panel(
        f"Session Start: {project}", lines, status="restore"
    )


@mcp.tool()
def memory_explain_search(query: str, k: int = 3, project: str = "") -> str:
    """Explain search rankings with a clean mathematical score breakdown table."""
    try:
        results = engine.explain_search(query, k=k, project=project or None)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Explain Failure", [f"Error: {e}"], status="error"
        )

    if not results:
        return TUIRenderer.render_panel(
            "Explain: 0 results",
            [f"No matching documents for: '{query}'"],
            status="explain",
        )

    lines = [f'Query: "{query}"', "Mathematical Score Breakdown Table:", ""]

    for idx, res in enumerate(results):
        lines.append(f"{idx + 1}. File: {res['rel_path']}")
        lines.append(f"   Final Score: {res['final']:.4f}")
        # Table of details
        lines.append(f"   ┌────────────┬────────┬──────────┐")
        lines.append(f"   │ Metric     │ Value  │ Weight   │")
        lines.append(f"   ├────────────┼────────┼──────────┤")
        lines.append(f"   │ Dense      │ {res['dense']:.4f} │ x0.45    │")
        lines.append(f"   │ BM25       │ {res['bm25']:.4f} │ x0.30    │")
        lines.append(f"   │ Recency    │ {res['recency']:.4f} │ x0.15    │")
        lines.append(f"   │ Importance │ {res['importance']:.4f} │ x0.10    │")
        lines.append(f"   └────────────┴────────┴──────────┘")
        lines.append("")

    return TUIRenderer.render_panel(
        "Observability: Explain Search", lines, status="explain"
    )


@mcp.tool()
def memory_reflect_session(project: str, conversation_text: str = "") -> str:
    """Generate a structured session reflection report under directives/sessions/."""
    try:
        res = engine.reflect_session(project, conversation_text)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Reflection Failure", [f"Error: {e}"], status="error"
        )

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

    return TUIRenderer.render_panel("Session Reflection Logged", lines, status="ok")


@mcp.tool()
def memory_materialize_concepts(project: str) -> str:
    """Materialize parsed session learnings into kebab-case markdown concept nodes."""
    try:
        res = engine.materialize_concepts(project)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Materialization Failure", [f"Error: {e}"], status="error"
        )

    if res.get("status") == "error":
        return TUIRenderer.render_panel(
            "Materialization Error", [res["message"]], status="error"
        )

    lines = [
        res.get("message", "Concept materialization pipeline completed."),
        "",
        "Materialization Log:",
    ]
    if res.get("materialized"):
        for m in res["materialized"]:
            lines.append(f"  ✨ {m}")
    else:
        lines.append("  No new or modified concepts found to materialize.")

    return TUIRenderer.render_panel("Concept Materialization", lines, status="ok")


@mcp.tool()
def memory_update_graph(project: str) -> str:
    """Analyze concept nodes and establish reciprocal bidirectional Obsidian-style [[wikilinks]]."""
    try:
        res = engine.update_concept_graph(project)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Graph Update Failure", [f"Error: {e}"], status="error"
        )

    if res.get("status") == "error":
        return TUIRenderer.render_panel(
            "Graph Update Error", [res["message"]], status="error"
        )

    lines = [
        res.get("message", "Graph structure updated."),
        f"Total links added or normalized: {res.get('links_added_or_normalized', 0)}",
    ]
    return TUIRenderer.render_panel(
        "Bidirectional Graph Auto-Linker", lines, status="organize"
    )


@mcp.tool()
def memory_session_commit(project: str, conversation_text: str = "") -> str:
    """End Session Master Orchestrator: runs reflect, materialize, update graph, and vector index in one step."""
    try:
        res = engine.session_commit(project, conversation_text)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Session Commit Failure", [f"Error: {e}"], status="error"
        )

    events = res.get("knowledge_events", [])
    event_counts = {}
    for ev in events:
        t = ev.get("type", "observation")
        event_counts[t] = event_counts.get(t, 0) + 1

    lines = [
        f"Master session commit succeeded.",
        f"Report: {Path(res['report_path']).name}",
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
    return TUIRenderer.render_panel("Session Commit Complete", lines, status="ok")


@mcp.tool()
def memory_lexikon_add(
    project: str, term: str, definition: str = "", aliases: str = ""
) -> str:
    """Hand-add a Lexikon/Glossary term (project or _global user lexikon)."""
    alias_list = [a.strip() for a in aliases.split(",") if a.strip()]
    try:
        res = engine.add_lexikon_term(project, term, definition, alias_list)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Lexikon Failure", [f"Error: {e}"], status="error"
        )
    if res.get("status") == "error":
        return TUIRenderer.render_panel(
            "Lexikon Error", [res["message"]], status="error"
        )
    return TUIRenderer.render_panel("Lexikon Updated", [res["message"]], status="write")


@mcp.tool()
def memory_lexikon_show(project: str) -> str:
    """Show merged Lexikon/Glossary terms (_global + project)."""
    try:
        lex = engine.get_lexikon(project)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Lexikon Failure", [f"Error: {e}"], status="error"
        )
    if not lex:
        return TUIRenderer.render_panel(
            "Lexikon Empty", ["No terms yet."], status="info"
        )
    lines = [f"Terms: {len(lex)}", ""]
    for key in sorted(lex)[:20]:
        e = lex[key] if isinstance(lex[key], dict) else {}
        aliases = ", ".join(e.get("aliases", []) or [])
        extra = f" (aka: {aliases})" if aliases else ""
        lines.append(f"  - {e.get('term', key)}{extra} [{e.get('source', '?')}]")
    if len(lex) > 20:
        lines.append(f"  ... and {len(lex) - 20} more")
    return TUIRenderer.render_panel(f"Lexikon: {project}", lines, status="stats")


@mcp.tool()
def memory_lexikon_build(project: str) -> str:
    """Derive project Lexikon from concept registry and session reports."""
    try:
        res = engine.build_lexikon(project)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Lexikon Failure", [f"Error: {e}"], status="error"
        )
    if res.get("status") == "error":
        return TUIRenderer.render_panel(
            "Lexikon Error", [res["message"]], status="error"
        )
    return TUIRenderer.render_panel("Lexikon Built", [res["message"]], status="ok")


@mcp.tool()
def memory_janitor(dry_run: bool = False) -> str:
    """Delete orphan chunks whose source files no longer exist on disk."""
    try:
        res = engine.janitor(dry_run=dry_run)
    except Exception as e:
        return TUIRenderer.render_panel(
            "Janitor Failure", [f"Error: {e}"], status="error"
        )
    lines = [f"Total chunks freed: {res['freed']}", ""]
    for name, info in sorted(res["collections"].items()):
        if "error" in info:
            lines.append(f"  - {name}: ERROR {info['error']}")
        else:
            lines.append(f"  - {name}: {info['chunks']} chunks, freed {info['freed']}")
    return TUIRenderer.render_panel("Janitor", lines, status="organize")


def main():
    """Console-script entrypoint for uvx: run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
