# SPDX-FileCopyrightText: Nova-Violet Role ORG
# SPDX-License-Identifier: EUPL-1.2 OR AGPL-3.0-or-later
"""RTR release gate.

Builds temp fixture projects (outside every root, exercising absolute-path
flow), indexes, and asserts the golden expectations recorded from the
previous release: top-1 stability (no regressal), recall, score bounds,
DTD metadata, lexikon behavior. Purges everything afterwards.

Exit 0 = ship. Exit 1 = no ship.

Run with the deployment env, e.g.:
  uvx --from <wheel> python -m execution.rtr_eval
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.absolute()))

import memory_engine as mem
from memory_engine import MemoryEngine

QUOKKA = """# Quokka Routing Notes

The quokka router6834 dispatches nocturnal packets through the southern gateway.
Latency budget is eleven milliseconds per hop. The fallback cache lives beside
the burrow index and warms on first moonrise.
"""

BADGER = """# Badger Latency Ledger

The badger arbiter counts midnight roundtrips across the northern bridge.
Throughput ceiling is fortytwo thousand packets per dusk cycle.
"""

PACKETS = """# General Packet Lore

Packets travel at dawn. Gateways open and close with the tide tables.
Caches warm slowly when nobody watches them.
"""

# query -> expected top-1 filename (recorded from the previous release)
GOLDEN_TOP1 = {
    "quokka router nocturnal": "wiki_quokka.md",
    "badger arbiter midnight": "wiki_badger.md",
    "tide tables gateways": "wiki_packets.md",
}

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main():
    base = Path(tempfile.mkdtemp(prefix="rtr-gate-"))
    proj = base / "gateproj"
    d = proj / "directives"
    d.mkdir(parents=True)
    (d / "wiki_quokka.md").write_text(QUOKKA, encoding="utf-8")
    (d / "wiki_badger.md").write_text(BADGER, encoding="utf-8")
    (d / "wiki_packets.md").write_text(PACKETS, encoding="utf-8")

    had_extra = mem.EXTRA_ROOTS_FILE.exists()
    had_global_lex = mem.LEXIKON_GLOBAL_FILE.exists()
    engine = MemoryEngine()

    try:
        # Real flow: first touch (init) registers the out-of-roots dir.
        r = engine.init_project(str(proj))
        check("init-outside-roots", r.get("status") == "success")
        st = engine.index_all()
        check("index-new-files", st.get("new", 0) >= 3, f"new={st.get('new')}")
        check("index-no-failures", st.get("failed", 0) == 0)

        cols = engine.collections
        check("collections-populated", all(c.count() > 0 for c in cols.values()))
        n_models = len(engine.embed_model_names)
        check("model-count", len(cols) == n_models, f"{len(cols)} vs {n_models}")

        for query, expected in GOLDEN_TOP1.items():
            res = engine.search(query, k=3, project=str(proj))
            top = Path(res[0]["metadata"]["source"]).name if res else None
            check(f"top1[{query}]", top == expected, f"got={top}")
            if res:
                check(
                    f"top1-score[{query}]",
                    res[0]["score"] >= 0.6,
                    f"{res[0]['score']:.4f}",
                )
                names = {Path(r["metadata"]["source"]).name for r in res}
                check(f"recall[{query}]", expected in names)
                check(
                    f"dense-bounds[{query}]",
                    all(0.0 <= r["score"] <= 1.0 for r in res),
                )

        # DTD metadata present on chunks
        sample = None
        for col in cols.values():
            got = col.get(limit=5)
            if got and got["metadatas"]:
                sample = got["metadatas"][0]
                break
        check("dtype_hist-present", bool(sample and sample.get("dtype_hist")))
        check("notations-present", sample is not None and "notations" in sample)
        check(
            "embed_model-present",
            bool(sample and sample.get("embed_model")),
        )

        # CDATA exact lane: opaque token query still resolves
        res = engine.search("router6834", k=1, project=str(proj))
        top = Path(res[0]["metadata"]["source"]).name if res else None
        check("exact-lane[router6834]", top == "wiki_quokka.md", f"got={top}")

        # Lexikon: hand term in _global expands a query (lookup, chilly)
        r = engine.add_lexikon_term(
            "_global", "nocturnal packets", "night-time quokka dispatch", ["night ops"]
        )
        check("lexikon-add", r.get("status") == "success")
        exp = engine.lexikon_expansion("_global", "quokka nocturnal schedule")
        check("lexikon-expansion", len(exp) > 0, f"{exp[:4]}")
        shown = engine.get_lexikon("_global")
        check("lexikon-show", "nocturnal packets" in shown)

        # Bare-name + global search still bleed correctly
        res = engine.search("quokka router nocturnal", k=1)
        check(
            "global-bleed",
            bool(res) and Path(res[0]["metadata"]["source"]).name == "wiki_quokka.md",
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)
        try:
            engine.index_all()
        except Exception:
            pass
        if not had_extra and mem.EXTRA_ROOTS_FILE.exists():
            try:
                data = json.loads(mem.EXTRA_ROOTS_FILE.read_text(encoding="utf-8"))
                data = [p for p in data if base.name not in str(p)]
                if data:
                    mem.EXTRA_ROOTS_FILE.write_text(
                        json.dumps(data, indent=2), encoding="utf-8"
                    )
                else:
                    mem.EXTRA_ROOTS_FILE.unlink()
            except Exception:
                pass
        if not had_global_lex and mem.LEXIKON_GLOBAL_FILE.exists():
            try:
                mem.LEXIKON_GLOBAL_FILE.unlink()
            except Exception:
                pass

    print(f"\nGATE {'PASS' if not failures else 'FAIL'}: {len(failures)} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
