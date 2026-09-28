import unittest

# SPDX-FileCopyrightText: Nova-Violet Role ORG
# SPDX-License-Identifier: EUPL-1.2 OR AGPL-3.0-or-later
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add execution directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Import memory_engine to allow path interception
import memory_engine
from memory_engine import MemoryEngine
from tui_renderer import TUIRenderer


class TestMemoryRAG(unittest.TestCase):
    def setUp(self):
        # Create a unique temp test directory for each test run to prevent SQLite lock collisions
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_projects = self.test_dir / "projects"
        self.test_db = self.test_dir / "db"
        self.test_registry = self.test_dir / "registry.json"

        # Intercept paths in memory_engine
        memory_engine.PROJECTS_DIR = self.test_projects
        memory_engine.PROJECT_ROOTS = [self.test_projects]
        memory_engine.DB_DIR = self.test_db
        memory_engine.REGISTRY_FILE = self.test_registry

        self.test_projects.mkdir(parents=True, exist_ok=True)
        self.test_db.mkdir(parents=True, exist_ok=True)

        # Setup dummy projects
        self.p1 = self.test_projects / "youtube-shorts"
        self.p1_directives = self.p1 / "directives"
        self.p1_concepts = self.p1_directives / "wiki_concepts"
        self.p1_concepts.mkdir(parents=True, exist_ok=True)

        self.p2 = self.test_projects / "research-slide"
        self.p2_directives = self.p2 / "directives"
        self.p2_directives.mkdir(parents=True, exist_ok=True)

        # Write dummy wiki files
        self.f1 = self.p1_directives / "wiki_inbox.md"
        with open(self.f1, "w", encoding="utf-8") as f:
            f.write(
                "# Youtube Hooks\n\nUse a visual pattern interrupt in the first 3 seconds.\n\n## Buffer API\n\nBuffer GraphQL API uses OIDC tokens."
            )

        self.f2 = self.p1_concepts / "voice_over.md"
        with open(self.f2, "w", encoding="utf-8") as f:
            f.write("# TTS Synthesizer\n\nPiper TTS works best at 22kHz sampling rate.")

        self.f3 = self.p2_directives / "design_wiki.md"
        with open(self.f3, "w", encoding="utf-8") as f:
            f.write(
                "# Layout Rules\n\nUse HSL harmonized grids for premium look.\n\n## Typography\n\nInter or Roboto from Google Fonts."
            )

        self.engine = MemoryEngine()

    def tearDown(self):
        # 1. Reset singleton caches and close DB connections BEFORE deleting directory
        if self.engine._chroma_client:
            try:
                self.engine._chroma_client.close()
            except Exception:
                pass
        self.engine._chroma_client = None
        self.engine._collections = {}
        self.engine._models = {}

        # 2. Delete the unique test directory safely
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_discovery(self):
        discovered = self.engine.discover_wikis()
        paths = [d["path"] for d in discovered]

        self.assertEqual(len(discovered), 3)
        self.assertIn(str(self.f1), paths)
        self.assertIn(str(self.f2), paths)
        self.assertIn(str(self.f3), paths)

        projects = [d["project"] for d in discovered]
        self.assertIn("youtube-shorts", projects)
        self.assertIn("research-slide", projects)

    def test_chunking(self):
        chunks = self.engine.chunk_markdown(
            self.f1, "youtube-shorts/directives/wiki_inbox.md"
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["title"], "Youtube Hooks")
        self.assertEqual(chunks[1]["title"], "Buffer API")

    def test_incremental_indexing(self):
        stats1 = self.engine.index_all()
        self.assertEqual(stats1["new"], 3)

        stats2 = self.engine.index_all()
        self.assertEqual(stats2["unchanged"], 3)

        with open(self.f1, "a", encoding="utf-8") as f:
            f.write("\n## New Section\nSome fresh learning added.")

        stats3 = self.engine.index_all()
        self.assertEqual(stats3["updated"], 1)

    def test_hybrid_search(self):
        self.engine.index_all()

        # Exact match check via hybrid search
        results = self.engine.search("Piper TTS", hybrid=True)
        self.assertGreater(len(results), 0)
        self.assertIn("Piper TTS", results[0]["document"])

        results_dense = self.engine.search("Piper TTS", hybrid=False)
        self.assertGreater(len(results_dense), 0)

    def test_project_init(self):
        res = self.engine.init_project("new-harness-project", variant="wiki")
        self.assertEqual(res["status"], "success")

        proj_dir = self.test_projects / "new-harness-project"
        self.assertTrue(proj_dir.exists())
        self.assertTrue((proj_dir / "directives" / "wiki_concepts").exists())
        self.assertTrue((proj_dir / "AGENTS.md").exists())
        self.assertTrue((proj_dir / "CLAUDE.md").exists())
        self.assertTrue((proj_dir / "state" / "workflow_state.json").exists())

    def test_consolidation(self):
        # Create duplicates
        dup1 = self.p1_concepts / "voice_over_tips.md"
        with open(dup1, "w", encoding="utf-8") as f:
            f.write(
                "# Voice Over Tips\n\nDetailed guidance about Piper TTS voice output."
            )

        self.engine.index_all()

        # Consolidate
        res = self.engine.consolidate_wiki("youtube-shorts")
        self.assertEqual(res["status"], "success")
        self.assertTrue(len(res["merged"]) >= 1)

        # Verify f2 content got updated
        with open(self.f2, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Voice Over Tips", content)

        # Verify dup1 is deleted
        self.assertFalse(dup1.exists())

    def test_tui_rendering(self):
        panel = TUIRenderer.render_panel(
            "Test Panel", ["Line 1", "Line 2"], status="ok"
        )
        self.assertIn("╔══", panel)
        self.assertIn("Test Panel", panel)
        self.assertIn("🟢 SUCCESS", panel)

    def test_recency_decay(self):
        import time

        now = time.time()
        score_now = self.engine.compute_recency_score(str(now))
        # 14 days ago
        two_weeks_ago = now - (14 * 24 * 3600)
        score_old = self.engine.compute_recency_score(str(two_weeks_ago))

        self.assertAlmostEqual(score_now, 1.0, places=2)
        # Should decay to approximately math.exp(-0.05 * 14) = 0.496
        self.assertLess(score_old, 0.6)
        self.assertGreater(score_old, 0.4)

    def test_importance_scoring(self):
        score_crit = self.engine.compute_importance_score("critical")
        score_high = self.engine.compute_importance_score("high")
        score_low = self.engine.compute_importance_score("low")

        self.assertEqual(score_crit, 1.0)
        self.assertEqual(score_high, 0.7)
        self.assertEqual(score_low, 0.2)

        # Test path mapping
        self.assertEqual(
            self.engine.derive_importance_rating("AGENTS.md", ""), "critical"
        )
        self.assertEqual(
            self.engine.derive_importance_rating("directives/walkthrough.md", ""),
            "high",
        )
        self.assertEqual(
            self.engine.derive_importance_rating(
                "directives/wiki_concepts/voice.md", ""
            ),
            "medium",
        )

    def test_explain_search(self):
        self.engine.index_all()
        results = self.engine.explain_search("Piper TTS")
        self.assertGreater(len(results), 0)
        self.assertIn("rel_path", results[0])
        self.assertIn("final", results[0])
        self.assertIn("dense", results[0])
        self.assertIn("bm25", results[0])

    def test_restore_session_state(self):
        res = self.engine.init_project("test-restore", variant="wiki")
        self.assertEqual(res["status"], "success")

        proj_dir = self.test_projects / "test-restore"

        # Write specific goals and issues
        goals_file = proj_dir / "state" / "current-goals.md"
        with open(goals_file, "w", encoding="utf-8") as f:
            f.write(
                "# Active Goals\n- Implement Piper TTS integration\n- Build webhook pipeline\n"
            )

        issues_file = proj_dir / "state" / "open-issues.md"
        with open(issues_file, "w", encoding="utf-8") as f:
            f.write("# Open Issues\n- Duplicate Stripe webhooks\n")

        self.engine.index_all()

        restored = self.engine.restore_session_state("test-restore")
        self.assertEqual(restored["status"], "success")
        self.assertIn("Stripe webhooks", restored["blockers"][0])
        self.assertIn("Piper TTS integration", restored["active_goals"][0])

    def test_session_reflection(self):
        concept_file = self.p1_concepts / "new_test_concept.md"
        with open(concept_file, "w", encoding="utf-8") as f:
            f.write("# New Test Concept\n\nSome concept body.")

        import json

        with open(self.test_registry, "w") as f:
            json.dump({}, f)

        chat_text = """
        We think BM25 weighting should be higher in our next launch.
        We tested the new BM25 weights on retrieval quality yesterday.
        We verified retrieval improved 18% with the new formula!
        However, the automatic top-3 auto-linking produced noisy graphs.
        We decided to use concept-centric wiki architecture instead.
        The old direct file linking strategy is retired and no longer recommended.
        """

        res = self.engine.reflect_session("youtube-shorts", chat_text)
        self.assertEqual(res["status"], "success")

        events = res["knowledge_events"]
        self.assertTrue(any(ev["type"] == "hypothesis" for ev in events))
        self.assertTrue(any(ev["type"] == "experiment" for ev in events))
        self.assertTrue(any(ev["type"] == "validation" for ev in events))
        self.assertTrue(any(ev["type"] == "failure" for ev in events))
        self.assertTrue(any(ev["type"] == "decision" for ev in events))
        self.assertTrue(any(ev["type"] == "deprecation" for ev in events))

    def test_concept_registry_lifecycle(self):
        self.engine.init_project("materialize-project", variant="wiki")
        proj_dir = self.test_projects / "materialize-project"
        sessions_dir = proj_dir / "directives" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # 1. First session: Create candidate concept (hypothesis)
        report_file = sessions_dir / "2026-06-01.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("""---
date: 2026-06-01
project: materialize-project
---
# Session Learning Report

## Knowledge Events
```yaml
knowledge_events:
  - type: hypothesis
    concept: hook-formats
    evidence: We think hook formats increase retention.
```
""")

        res = self.engine.materialize_concepts("materialize-project")
        self.assertEqual(res["status"], "success")

        # Candidate concepts should NOT write to disk
        hook_file = proj_dir / "directives" / "wiki_concepts" / "concept_001.md"
        self.assertFalse(hook_file.exists())

        # Verify candidate status in registry
        reg = self.engine._load_concept_registry("materialize-project")
        self.assertEqual(reg["concept_001"]["status"], "candidate")

        # 2. Second session: Emerging concept
        report_file2 = sessions_dir / "2026-06-02.md"
        with open(report_file2, "w", encoding="utf-8") as f:
            f.write("""---
date: 2026-06-02
project: materialize-project
---
# Session Learning Report

## Knowledge Events
```yaml
knowledge_events:
  - type: observation
    concept: hook-formats
    evidence: Observed hook formats again.
```
""")
        self.engine.materialize_concepts("materialize-project")
        reg = self.engine._load_concept_registry("materialize-project")
        self.assertEqual(reg["concept_001"]["status"], "emerging")
        self.assertFalse(hook_file.exists())

        # 3. Third session: Validations and evidence counts make it Validated
        report_file3 = sessions_dir / "2026-06-03.md"
        with open(report_file3, "w", encoding="utf-8") as f:
            f.write("""---
date: 2026-06-03
project: materialize-project
---
# Session Learning Report

## Knowledge Events
```yaml
knowledge_events:
  - type: validation
    concept: hook-formats
    evidence: Success metric 3 validated.
  - type: validation
    concept: hook-formats
    evidence: Confirmed by 5 analytics reports.
```
""")
        self.engine.materialize_concepts("materialize-project")
        reg = self.engine._load_concept_registry("materialize-project")
        self.assertEqual(reg["concept_001"]["status"], "validated")

        # Now validated node must be materialized as markdown node
        self.assertTrue(hook_file.exists())
        hook_text = hook_file.read_text()
        self.assertIn("status: validated", hook_text)

        # 4. Fourth session: Deprecation deletes node
        report_file4 = sessions_dir / "2026-06-04.md"
        with open(report_file4, "w", encoding="utf-8") as f:
            f.write("""---
date: 2026-06-04
project: materialize-project
---
# Session Learning Report

## Knowledge Events
```yaml
knowledge_events:
  - type: deprecation
    concept: hook-formats
    evidence: Old strategy is deprecated.
```
""")
        self.engine.materialize_concepts("materialize-project")
        reg = self.engine._load_concept_registry("materialize-project")
        self.assertEqual(reg["concept_001"]["status"], "deprecated")
        self.assertFalse(hook_file.exists())

    def test_bidirectional_linking_graph(self):
        self.engine.init_project("graph-project", variant="wiki")
        proj_dir = self.test_projects / "graph-project"
        concepts_dir = proj_dir / "directives" / "wiki_concepts"
        concepts_dir.mkdir(parents=True, exist_ok=True)

        # Pre-seed concept registry
        registry = {
            "concept_001": {
                "canonical_name": "concept-a",
                "aliases": ["concept a"],
                "status": "validated",
                "confidence": 3,
                "evidence_count": 3,
                "session_count": 3,
            },
            "concept_002": {
                "canonical_name": "concept-b",
                "aliases": ["concept b"],
                "status": "validated",
                "confidence": 3,
                "evidence_count": 3,
                "session_count": 3,
            },
        }
        self.engine._save_concept_registry("graph-project", registry)

        file_a = concepts_dir / "concept_001.md"
        with open(file_a, "w", encoding="utf-8") as f:
            f.write("""---
concept_id: concept_001
canonical_name: concept-a
status: validated
---
This is concept A which mentions concept-b.""")

        file_b = concepts_dir / "concept_002.md"
        with open(file_b, "w", encoding="utf-8") as f:
            f.write("""---
concept_id: concept_002
canonical_name: concept-b
status: validated
---
This is concept B which discusses general layouts.""")

        res = self.engine.update_concept_graph("graph-project")
        self.assertEqual(res["status"], "success")

        text_b = file_b.read_text()
        self.assertIn("[[concept_001|Concept A]]", text_b)

        text_a = file_a.read_text()
        self.assertIn("[[concept_002|Concept B]]", text_a)


if __name__ == "__main__":
    unittest.main()
