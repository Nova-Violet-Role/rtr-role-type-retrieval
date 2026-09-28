# SPDX-FileCopyrightText: Nova-Violet Role ORG
# SPDX-License-Identifier: EUPL-1.2 OR AGPL-3.0-or-later
import os
import sys
import hashlib
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Force CPU execution to minimize resource footprint
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from fastembed import TextEmbedding

# Setup paths
USER_HOME = Path.home()
MEMORY_DIR = USER_HOME / ".config" / "opencode" / "memory"
_BASE_DB_DIR = MEMORY_DIR / ".local_vector_db"


def _db_dir() -> Path:
    """Vector DB home.

    OPENCODE_MEMORY_DB_DIR env override wins (bench isolation); otherwise
    the live module-global DB_DIR (monkeypatchable by tests) wins.
    """
    raw = os.environ.get("OPENCODE_MEMORY_DB_DIR", "").strip()
    if raw:
        p = Path(raw)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return globals().get("DB_DIR", _BASE_DB_DIR)


DB_DIR = _BASE_DB_DIR
REGISTRY_FILE = MEMORY_DIR / "wiki_registry.json"
EXTRA_ROOTS_FILE = MEMORY_DIR / "extra_roots.json"
PROJECTS_DIR = USER_HOME / "projects"

# --- Helping-Hand extensions: multi-root projects + multi-model embeddings ---
# Env knobs (set from opencode.jsonc `environment` — uvx passes them through):
#   OPENCODE_MEMORY_PROJECTS_DIR  os.pathsep-separated discovery roots (default: ~/projects)
#   OPENCODE_MEMORY_EMBED_MODELS   comma-separated fastembed model names (default: dual ensemble)
#   OPENCODE_MEMORY_EMBED_THREADS  onnxruntime threads per model, e.g. 8 (default: unset = auto)
DEFAULT_EMBED_MODELS = [
    "Snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-base-en-v1.5",
]


def configured_embed_models() -> List[str]:
    raw = os.environ.get("OPENCODE_MEMORY_EMBED_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return models or list(DEFAULT_EMBED_MODELS)


def configured_embed_threads() -> Any:
    raw = os.environ.get("OPENCODE_MEMORY_EMBED_THREADS", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
        return n if n > 0 else None
    except Exception:
        return None


def configured_rrf_k() -> float:
    """RRF rank-dampening constant (standard default 60)."""
    raw = os.environ.get("OPENCODE_MEMORY_RRF_K", "").strip()
    if not raw:
        return 60.0
    try:
        k = float(raw)
        return k if k > 0 else 60.0
    except Exception:
        return 60.0


def configured_rtr_profile() -> str:
    """RTR compute profile: balanced (4+4 threads) or performance (8+8)."""
    raw = os.environ.get("OPENCODE_MEMORY_RTR_PROFILE", "").strip().lower()
    return raw if raw in ("balanced", "performance") else "balanced"


def resolve_threads_for(model_name: str) -> int:
    """Per-model onnxruntime threads.

    Explicit mapping wins: OPENCODE_MEMORY_EMBED_THREADS="4" (all models)
    or "arctic:4,bge:8" (substring match on the model slug).
    Otherwise the RTR profile decides: balanced 4+4, performance 8+8.
    """
    raw = os.environ.get("OPENCODE_MEMORY_EMBED_THREADS", "").strip()
    if raw:
        if ":" in raw:
            slug = model_slug(model_name)
            for part in raw.split(","):
                if ":" not in part:
                    continue
                key, val = part.split(":", 1)
                key = key.strip().lower()
                if key and key in slug:
                    try:
                        n = int(val.strip())
                        if n > 0:
                            return n
                    except Exception:
                        pass
            return 4 if configured_rtr_profile() == "balanced" else 8
        try:
            n = int(raw)
            if n > 0:
                return n
        except Exception:
            pass
    return 4 if configured_rtr_profile() == "balanced" else 8


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def collection_name_for(model_name: str) -> str:
    return f"opencode_memory_{model_slug(model_name)}"


def _project_roots_from_env() -> List[Path]:
    raw = os.environ.get("OPENCODE_MEMORY_PROJECTS_DIR", "")
    if raw.strip():
        roots = [Path(p.strip()) for p in raw.split(os.pathsep) if p.strip()]
        if roots:
            return roots
    return [USER_HOME / "projects"]


PROJECT_ROOTS: List[Path] = _project_roots_from_env()


def _load_extra_roots() -> List[Path]:
    try:
        if EXTRA_ROOTS_FILE.exists():
            data = json.loads(EXTRA_ROOTS_FILE.read_text(encoding="utf-8"))
            return [Path(p) for p in data if isinstance(p, str) and p.strip()]
    except Exception:
        pass
    return []


def register_extra_root(project_dir: Path) -> None:
    """Persist an out-of-roots project dir so future discovery/index runs cover it."""
    try:
        d = Path(project_dir)
        if not d.exists():
            return
        known = _load_extra_roots()
        if any(str(d) == str(k) for k in known):
            return
        known.append(d)
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        EXTRA_ROOTS_FILE.write_text(
            json.dumps([str(k) for k in known], indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"Could not persist extra root {project_dir}: {e}", file=sys.stderr)


def _contains(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _is_abs_ref(s: str) -> bool:
    if not s:
        return False
    p = Path(s)
    if p.is_absolute():
        return True
    if len(s) > 2 and s[1] == ":":
        return True
    if os.sep in s or (os.altsep and os.altsep in s):
        return True
    return False


def resolve_project_dir(project: str) -> Path:
    """Resolve a project reference to a directory.

    - Absolute path (or containing separators/drive) -> used directly; auto-registered
      for discovery when it lives outside the configured roots.
    - Bare name -> first existing <root>/<name> across roots + registered extras,
      else <first-root>/<name> (for creation).
    Same-name collisions across roots: bare names resolve first-match (documented);
    absolute paths always resolve exactly.
    """
    s = str(project or "").strip()
    if _is_abs_ref(s):
        d = Path(s)
        if d.exists() and not any(_contains(r, d) for r in PROJECT_ROOTS):
            register_extra_root(d)
        return d
    for base in list(PROJECT_ROOTS) + _load_extra_roots():
        cand = base / s
        if cand.exists():
            return cand
    for extra in _load_extra_roots():
        if extra.name == s and extra.exists():
            return extra
    return PROJECTS_DIR / s


def iter_discovery_scopes() -> List[Tuple[Path, Any]]:
    """Yield (base_dir, fixed_key) scopes. fixed_key=None -> key = first path part."""
    scopes: List[Tuple[Path, Any]] = [(r, None) for r in PROJECT_ROOTS]
    for extra in _load_extra_roots():
        if extra.exists():
            scopes.append((extra.parent, extra.name))
    return scopes


def describe_indexed_path(path_str: str) -> Tuple[str, str, str]:
    """Map an indexed file path -> (project_key, rel_path, root). Root-aware."""
    try:
        ap = Path(path_str)
    except Exception:
        return ("unknown", str(path_str), "")
    for base, fixed in iter_discovery_scopes():
        try:
            rel = ap.relative_to(base)
        except Exception:
            continue
        if not rel.parts:
            continue
        key = fixed if fixed else rel.parts[0]
        return (key, str(rel), str(base))
    parts = ap.parts
    if "directives" in parts:
        i = parts.index("directives")
        if i > 0:
            return (parts[i - 1], str(ap), "")
    return (ap.parent.name if len(parts) > 1 else "unknown", str(ap), "")


def project_filter_key(project: str) -> str:
    """Map a search/list filter arg to the metadata key used at index time."""
    s = str(project or "").strip()
    if _is_abs_ref(s):
        return Path(s).name
    return s


# --- RTR Pillar 2: DTD-typed spans (PCDATA / CDATA / NDATA) ---
# PCDATA: parsed prose — full hybrid lane. CDATA: opaque tokens (code, ids,
# hashes, paths, URLs) — exact-match lane. NDATA: non-text (base64 blobs) —
# masked before embedding, metadata-only. Classification is structural
# (regex), done once at index: pre-classified, lightweight, no model cost.
DTYPE_PCDATA = "PCDATA"
DTYPE_CDATA = "CDATA"
DTYPE_NDATA = "NDATA"
NDATA_PLACEHOLDER = "[NDATA]"
EXACT_LANE_WEIGHT = 0.15

_NDATA_RES = [re.compile(r"[A-Za-z0-9+/]{64,}={0,2}")]
_CDATA_RES = [
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"`[^`\n]+`"),
    re.compile(r"https?://[^\s)>\]]+"),
    re.compile(r"\b[0-9a-fA-F]{8,}\b"),
    re.compile(r"\b[\w.~\-]+(?:[\\/][\w.~\-]+)+\b"),
]
# DTD-faithful: every NDATA span carries its NOTATION (unparsed entities are
# never expanded inline — only referenced). CDATA spans carry their sub-kind.
_NOTATION_RES = [
    ("base64", re.compile(r"[A-Za-z0-9+/]{64,}={0,2}")),
    ("url", re.compile(r"https?://[^\s)>\]]+")),
    ("hex", re.compile(r"\b[0-9a-fA-F]{8,}\b")),
    ("path", re.compile(r"\b[\w.~\-]+(?:[\\/][\w.~\-]+)+\b")),
    ("code", re.compile(r"```.*?```", re.DOTALL)),
    ("code", re.compile(r"`[^`\n]+`")),
]


def notations_of(text: str) -> List[str]:
    """NOTATION names present in the text (empty for pure PCDATA)."""
    found = []
    for name, rx in _NOTATION_RES:
        try:
            if rx.search(text) and name not in found:
                found.append(name)
        except Exception:
            pass
    return found


_DTYPE_RANK = {DTYPE_NDATA: 2, DTYPE_CDATA: 1, DTYPE_PCDATA: 0}


def classify_spans(text: str) -> List[Tuple[str, str]]:
    """Segment text into (span, dtype) with NDATA > CDATA > PCDATA priority."""
    hits = []
    for rx in _NDATA_RES:
        for m in rx.finditer(text):
            hits.append((m.start(), m.end(), DTYPE_NDATA))
    for rx in _CDATA_RES:
        for m in rx.finditer(text):
            hits.append((m.start(), m.end(), DTYPE_CDATA))
    hits.sort(key=lambda h: (h[0], -_DTYPE_RANK[h[2]]))
    out = []
    pos = 0
    for s, e, t in hits:
        if s < pos or e <= pos:
            continue
        if s > pos:
            out.append((text[pos:s], DTYPE_PCDATA))
        out.append((text[s:e], t))
        pos = e
    if pos < len(text):
        out.append((text[pos:], DTYPE_PCDATA))
    return [(span, t) for span, t in out if span]


def mask_ndata(text: str) -> str:
    """Replace NDATA spans with a placeholder (chunk boundaries/ids unchanged)."""
    return "".join(
        (NDATA_PLACEHOLDER if t == DTYPE_NDATA else span)
        for span, t in classify_spans(text)
    )


def dtype_histogram(text: str) -> str:
    counts = {DTYPE_PCDATA: 0, DTYPE_CDATA: 0, DTYPE_NDATA: 0}
    total = 0
    for span, t in classify_spans(text):
        counts[t] += len(span)
        total += len(span)
    if total == 0:
        return f"{DTYPE_PCDATA}:1.00"
    return ",".join(
        f"{k}:{counts[k] / total:.2f}" for k in (DTYPE_PCDATA, DTYPE_CDATA, DTYPE_NDATA)
    )


def cdata_terms_of(text: str) -> List[str]:
    """Opaque terms that deserve the exact-match lane."""
    terms = []
    for span, t in classify_spans(text):
        if t != DTYPE_CDATA:
            continue
        core = span.strip("`").strip()
        if len(core) >= 3:
            terms.append(core.lower())
    seen, out = set(), []
    for term in terms:
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out


def configured_suffixes() -> Tuple[str, ...]:
    """Indexed file suffixes (DtD polyglot: markdown + NestedText + YAML)."""
    raw = os.environ.get("OPENCODE_MEMORY_FILE_SUFFIXES", "")
    if raw.strip():
        out = tuple(
            (
                s.strip().lower()
                if s.strip().startswith(".")
                else "." + s.strip().lower()
            )
            for s in raw.split(",")
            if s.strip()
        )
        if out:
            return out
    return (".md", ".nt", ".yaml", ".yml")


def _matches_suffix(name: str) -> bool:
    return name.lower().endswith(configured_suffixes())


LEXIKON_GLOBAL_FILE = MEMORY_DIR / "lexikon_global.json"


# Ensure directories exist
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


class MemoryEngine:
    # Process-wide ChromaDB clients, one per DB path. Multiple live
    # PersistentClient instances on the same path go stale against each
    # other (deleted HNSW ids resurfacing as "Error finding id" on filtered
    # queries). A single shared client per path is the documented shape.
    _CLIENTS: Dict[str, Any] = {}

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._chroma_client = None
        self._collections: Dict[str, Any] = {}

    @property
    def embed_model_names(self) -> List[str]:
        return configured_embed_models()

    @property
    def primary_model_name(self) -> str:
        return self.embed_model_names[0]

    @property
    def models(self) -> Dict[str, Any]:
        """All configured embedding models, keyed by slug. Warms ONNX on first access."""
        for name in self.embed_model_names:
            slug = model_slug(name)
            if slug not in self._models:
                threads = resolve_threads_for(name)
                print(
                    f"Loading {name} via fastembed... (threads={threads})",
                    file=sys.stderr,
                )
                self._models[slug] = TextEmbedding(model_name=name, threads=threads)
        return self._models

    @property
    def model(self):
        """Backward-compat: primary embedding model."""
        return self.models[model_slug(self.primary_model_name)]

    def embed_all(self, texts: List[str]) -> Dict[str, List[List[float]]]:
        out: Dict[str, List[List[float]]] = {}
        for slug, m in self.models.items():
            out[slug] = [[float(v) for v in e] for e in m.embed(texts)]
        return out

    @property
    def chroma_client(self):
        key = str(_db_dir())
        client = MemoryEngine._CLIENTS.get(key)
        if client is None:
            client = chromadb.PersistentClient(path=key)
            MemoryEngine._CLIENTS[key] = client
        self._chroma_client = client
        return client

    def rtr_collections(self) -> Dict[str, Any]:
        """All RTR collections actually present (configured + legacy/orphans).

        Purge and delete fan out over these so model switches never leak.
        """
        out: Dict[str, Any] = dict(self.collections)
        try:
            for col in self.chroma_client.list_collections():
                if col.name == "opencode_memory" or col.name.startswith(
                    "opencode_memory_"
                ):
                    out.setdefault(col.name, col)
        except Exception:
            pass
        return out

    @property
    def collections(self) -> Dict[str, Any]:
        """One ChromaDB collection per embedding model (fixed vector spaces never mix)."""
        if not self._collections:
            for name in self.embed_model_names:
                slug = model_slug(name)
                self._collections[slug] = self.chroma_client.get_or_create_collection(
                    name=collection_name_for(name),
                    metadata={"hnsw:space": "cosine"},
                )
        return self._collections

    @property
    def collection(self):
        """Backward-compat: primary collection."""
        return self.collections[model_slug(self.primary_model_name)]

    def calculate_sha256(self, filepath: Path) -> str:
        """Calculate SHA256 of file content to detect changes."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def discover_wikis(self) -> List[Dict[str, Any]]:
        """
        Recursively discover wiki-like markdown files across all scopes:
        configured PROJECT_ROOTS plus auto-registered out-of-roots project dirs.
        Matches directives/*.md and files/folders with 'wiki' in their paths.
        """
        discovered = []
        for scope_base, fixed_key in iter_discovery_scopes():
            if not scope_base.exists():
                continue
            # Walk through the scope recursively
            for root, dirs, files in os.walk(scope_base):
                root_path = Path(root)

                # Skip common noise dirs to keep index fast
                if any(
                    part in root_path.parts
                    for part in (
                        ".git",
                        "node_modules",
                        ".venv",
                        "venv",
                        "__pycache__",
                        ".local_vector_db",
                        "dist",
                        "build",
                    )
                ):
                    continue

                for file in files:
                    if not _matches_suffix(file):
                        continue

                    file_path = root_path / file
                    try:
                        rel_path = file_path.relative_to(scope_base)
                    except Exception:
                        continue
                    parts = rel_path.parts

                    if not parts:
                        continue

                    project_name = fixed_key if fixed_key else parts[0]

                    # Check if it fits option B: inside directives/ folder
                    is_directive = "directives" in parts

                    # Check if path or file has "wiki" in its name (priority)
                    has_wiki = any("wiki" in part.lower() for part in parts)

                    # We prioritize all .md files in directives/ folder,
                    # and any .md files with "wiki" in their paths
                    if is_directive or has_wiki:
                        category = "wiki" if has_wiki else "directive"
                        # If it's a concept subfolder inside directives, extract that subcategory
                        sub_category = None
                        if is_directive:
                            dir_idx = parts.index("directives")
                            if len(parts) > dir_idx + 2:
                                sub_category = parts[dir_idx + 1]

                        discovered.append(
                            {
                                "path": str(file_path),
                                "project": project_name,
                                "category": category,
                                "subcategory": sub_category or "general",
                                "filename": file_path.name,
                                "rel_path": str(rel_path),
                            }
                        )

        return discovered

    def chunk_markdown(self, filepath: Path, rel_path: str) -> List[Dict[str, Any]]:
        """
        Split a markdown file into logical sections based on headers.
        Each section includes the main document path context for the embedding model.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}", file=sys.stderr)
            return []

        # If file is empty, return nothing
        if not content.strip():
            return []

        # Parse sections using markdown headers
        # Split on headers (# , ## , ### , ####)
        header_pattern = re.compile(r"^(#{1,4}\s+.*)$", re.MULTILINE)
        parts = header_pattern.split(content)

        chunks = []
        current_title = "Introduction"

        # The split returns [initial_text, header_1, section_1, header_2, section_2, ...]
        if parts[0].strip():
            chunks.append({"title": current_title, "text": parts[0].strip()})

        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            section_body = parts[i + 1].strip() if i + 1 < len(parts) else ""

            # Extract header text
            title = header.lstrip("#").strip()

            if section_body:
                chunks.append({"title": title, "text": f"{header}\n\n{section_body}"})

        # If no header chunks were created, just treat the whole file as one chunk
        if not chunks:
            chunks.append({"title": "Full Document", "text": content.strip()})

        formatted_chunks = []
        for idx, chunk in enumerate(chunks):
            section_text = (
                f"Document: {rel_path}\nSection: {chunk['title']}\n\n{chunk['text']}"
            )
            links = list(set(re.findall(r"\[\[([^\]]+)\]\]", chunk["text"])))
            formatted_chunks.append(
                {
                    "chunk_id": f"{rel_path}#chunk_{idx}",
                    "text": section_text,
                    "title": chunk["title"],
                    "raw_body": chunk["text"],
                    "linked_concepts": links,
                }
            )

        return formatted_chunks

    def load_registry(self) -> Dict[str, str]:
        if REGISTRY_FILE.exists():
            try:
                with open(REGISTRY_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_registry(self, registry: Dict[str, str]):
        try:
            with open(REGISTRY_FILE, "w") as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            print(f"Error saving registry: {e}", file=sys.stderr)

    def derive_importance_rating(self, rel_path: str, text: str) -> str:
        """Derive importance rating for metadata tagging."""
        path_lower = rel_path.lower()
        if (
            "agents.md" in path_lower
            or "claude.md" in path_lower
            or "workflow_state.json" in path_lower
            or "feature_list.json" in path_lower
        ):
            return "critical"
        elif "scratch" in path_lower:
            return "low"
        elif "directives" in path_lower and not (
            "concepts" in path_lower or "wiki" in path_lower
        ):
            return "high"
        return "medium"

    def index_all(
        self, force: bool = False, projects: List[str] = None
    ) -> Dict[str, Any]:
        """Scan projects, compute hashes, update vector DB only for new/changed wikis.

        `projects`: extra project refs resolved + registered before discovery,
        so out-of-roots codebases index on first touch.
        """
        for pref in projects or []:
            try:
                resolve_project_dir(pref)
            except Exception:
                pass
        discovered = self.discover_wikis()
        registry = self.load_registry()

        stats = {
            "scanned": len(discovered),
            "new": 0,
            "updated": 0,
            "unchanged": 0,
            "failed": 0,
        }

        # Track active paths to purge deleted files later
        active_paths = set()
        new_registry = {}

        to_index = []  # List of tuples: (file_info, filepath, old_hash, new_hash)

        for info in discovered:
            path_str = info["path"]
            filepath = Path(path_str)
            active_paths.add(path_str)

            if not filepath.exists():
                continue

            try:
                current_hash = self.calculate_sha256(filepath)
                old_hash = registry.get(path_str)
                new_registry[path_str] = current_hash

                if force or old_hash != current_hash:
                    if old_hash is None:
                        stats["new"] += 1
                    else:
                        stats["updated"] += 1
                    to_index.append((info, filepath, old_hash, current_hash))
                else:
                    stats["unchanged"] += 1
            except Exception as e:
                print(f"Failed to hash {path_str}: {e}", file=sys.stderr)
                stats["failed"] += 1
                if path_str in registry:
                    new_registry[path_str] = registry[path_str]

        # Execute indexing for new/changed files, once per embedding model.
        # Each model owns its collection since fixed vector spaces never mix.
        if to_index:
            # Pre-chunk every file once; embeddings fan out per model below.
            # NDATA spans are masked before embedding (pillar 2); chunk
            # boundaries and ids stay stable.
            jobs = []
            for info, filepath, old_hash, new_hash in to_index:
                chunks = self.chunk_markdown(filepath, info["rel_path"])
                if not chunks:
                    continue
                mtime_val = (
                    os.path.getmtime(filepath) if filepath.exists() else time.time()
                )
                imp = self.derive_importance_rating(info["rel_path"], "")
                masked = [mask_ndata(c["text"]) for c in chunks]
                hists = [dtype_histogram(c["text"]) for c in chunks]
                nots = [",".join(notations_of(c["text"])) for c in chunks]
                jobs.append(
                    (
                        info,
                        filepath,
                        old_hash,
                        new_hash,
                        chunks,
                        mtime_val,
                        imp,
                        masked,
                        hists,
                        nots,
                    )
                )

            # We load models only when there's actual indexing to do
            models = self.models
            cols = self.collections
            for slug, col in cols.items():
                model_name = next(
                    (n for n in self.embed_model_names if model_slug(n) == slug),
                    slug,
                )
                for (
                    info,
                    filepath,
                    old_hash,
                    new_hash,
                    chunks,
                    mtime_val,
                    imp,
                    masked,
                    hists,
                    nots,
                ) in jobs:
                    # 1. Clean up old chunks for this file in this collection
                    if old_hash is not None:
                        # ChromaDB metadata allows filtering or we delete by ID prefix matching
                        # Since chunk IDs start with rel_path, we can query or just delete
                        # ChromaDB doesn't support wildcards for IDs easily, so we query the IDs first
                        try:
                            existing = col.get(where={"source": info["path"]})
                            if existing and existing["ids"]:
                                col.delete(ids=existing["ids"])
                        except Exception as e:
                            print(
                                f"Error purging old chunks for {info['rel_path']}: {e}",
                                file=sys.stderr,
                            )

                    ids = [c["chunk_id"] for c in chunks]
                    texts = masked
                    metadatas = [
                        {
                            "source": info["path"],
                            "rel_path": info["rel_path"],
                            "project": info["project"],
                            "category": info["category"],
                            "subcategory": info["subcategory"],
                            "title": c["title"],
                            "content_hash": new_hash,
                            "linked_concepts": ",".join(c["linked_concepts"]),
                            "created_at": str(mtime_val),
                            "updated_at": str(mtime_val),
                            "last_accessed": str(time.time()),
                            "access_count": "0",
                            "importance": imp,
                            "embed_model": model_name,
                            "dtype_hist": h,
                            "notations": n,
                        }
                        for c, h, n in zip(chunks, hists, nots)
                    ]

                    # Generate embeddings with this collection's own model
                    try:
                        embeddings = [
                            [float(val) for val in e] for e in models[slug].embed(texts)
                        ]
                        col.add(
                            ids=ids,
                            embeddings=embeddings,
                            documents=texts,
                            metadatas=metadatas,
                        )
                    except Exception as e:
                        print(
                            f"Failed indexing chunks for {info['rel_path']}: {e}",
                            file=sys.stderr,
                        )
                        stats["failed"] += 1

        # 3. Detect deleted files and remove them from every RTR collection
        # (configured + legacy), verifying so segment desync cannot linger.
        deleted_paths = set(registry.keys()) - active_paths
        if deleted_paths:
            for col in self.rtr_collections().values():
                for path in deleted_paths:
                    try:
                        existing = col.get(where={"source": path})
                        if existing and existing["ids"]:
                            col.delete(ids=existing["ids"])
                        print(
                            f"Purged deleted file from index: {path}", file=sys.stderr
                        )
                        again = col.get(where={"source": path})
                        if again and again["ids"]:
                            print(
                                f"WARNING: remnants of {path} persist in "
                                f"{col.name} ({len(again['ids'])} chunks)",
                                file=sys.stderr,
                            )
                    except Exception as e:
                        print(
                            f"Error purging deleted file {path}: {e}", file=sys.stderr
                        )

        self.save_registry(new_registry)
        return stats

    def compute_bm25_scores(self, query: str, documents: List[str]) -> List[float]:
        """Compute pure-Python BM25 scores for a list of candidate documents."""
        import math

        query_terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 1]
        if not query_terms or not documents:
            return [0.0] * len(documents)

        doc_terms_list = []
        for doc in documents:
            doc_terms_list.append([t.lower() for t in re.findall(r"\w+", doc)])

        doc_count = len(documents)
        df = {}
        for terms in doc_terms_list:
            unique_terms = set(terms)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1

        k1 = 1.5
        b = 0.75
        avg_doc_len = sum(len(terms) for terms in doc_terms_list) / max(doc_count, 1)

        scores = []
        for terms in doc_terms_list:
            score = 0.0
            doc_len = len(terms)
            term_counts = {}
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1

            for q_term in query_terms:
                if q_term in term_counts:
                    n = df.get(q_term, 0)
                    idf = math.log((doc_count - n + 0.5) / (n + 0.5) + 1.0)
                    tf = term_counts[q_term]
                    tf_score = (tf * (k1 + 1)) / (
                        tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
                    )
                    score += idf * tf_score
            scores.append(score)

        max_score = max(scores) if scores else 0
        if max_score > 0:
            scores = [s / max_score for s in scores]
        return scores

    def compute_recency_score(self, created_at_str: str) -> float:
        """Compute an exponential decay recency score based on creation timestamp (half-life of 14 days)."""
        import math

        try:
            created_at = float(created_at_str)
            age_days = (time.time() - created_at) / (24 * 3600)
            return math.exp(-0.05 * max(0.0, age_days))
        except Exception:
            return 1.0

    def compute_importance_score(self, importance_str: str) -> float:
        """Translate metadata importance string into numeric multiplier score."""
        imp = str(importance_str or "").lower()
        if imp == "critical":
            return 1.0
        elif imp == "high":
            return 0.7
        elif imp == "medium":
            return 0.5
        elif imp == "low":
            return 0.2
        return 0.5

    def _fused_candidates(
        self, query: str, k: int = 3, project: str = None, hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """Single-query convenience over _fused_multi (identical math)."""
        return self._fused_multi([query], k=k, project=project, hybrid=hybrid)

    def build_expansions(
        self, query: str, project: str = None
    ) -> tuple[List[str], List[str]]:
        """Split expansion into trusted (lexikon) and noisy (delta) lanes.

        Lexikon entries are curated/derived knowledge — they bypass the
        relevance gate. Delta stems are guesses — gated by evidence.
        """
        lex_subs, delta_subs = [], []
        if project:
            try:
                lex = self.lexikon_expansion(project, query)
                if lex:
                    cand = (query + " " + " ".join(lex)).strip()
                    if cand != query:
                        lex_subs.append(cand)
            except Exception:
                pass
            try:
                deltas = self.detect_workspace_deltas(project)
                toks: List[str] = []
                for p in (deltas.get("created", []) + deltas.get("modified", []))[:10]:
                    toks.extend(
                        t
                        for t in re.findall(r"\w+", Path(p).stem.lower())
                        if len(t) > 2
                    )
                extra = [t for t in dict.fromkeys(toks) if t not in query.lower()][:8]
                if extra:
                    delta_subs.append(query + " " + " ".join(extra))
            except Exception:
                pass
        return lex_subs, delta_subs

    def build_subqueries(self, query: str, project: str = None) -> List[str]:
        """RTR Pillar 1: original + lexikon-expanded + workspace-delta-expanded.

        Lookup-only expansion (registry aliases, glossary definitions, recent
        file-delta stems). Deterministic, no generator model.
        """
        lex_subs, delta_subs = self.build_expansions(query, project)
        out = [query]
        for s in lex_subs + delta_subs:
            if s not in out:
                out.append(s)
        return out[:4]

    def _fused_multi(
        self,
        subqueries: List[str],
        k: int = 3,
        project: str = None,
        hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fuse sub-queries x models with confidence-weighted RRF.

        Each (sub-query, model) run votes 1/(K+rank), weighted by its own
        decisiveness (top-vs-median similarity spread): peaked runs vote
        loudly, flat uncertain runs vote quietly. Unsupervised, per-query,
        no labels. A CDATA exact-match lane adds a bounded bonus for opaque
        query terms (pillar 2).
        """
        subs = [s for s in (subqueries or []) if str(s or "").strip()]
        if not subs:
            return []
        base_query = subs[0]
        cols = self.collections
        counts = {slug: col.count() for slug, col in cols.items()}
        if not cols or max(counts.values()) == 0:
            return []

        candidate_count = min(max(counts.values()), max(20, k * 4))

        where_filter = None
        if project:
            where_filter = {"project": project_filter_key(project)}

        query_vecs = {q: self.embed_all([q]) for q in subs}
        per_doc: Dict[str, Dict[str, Any]] = {}
        run_ranks: List[Dict[str, int]] = []
        run_spreads: List[float] = []
        for q in subs:
            for slug, col in cols.items():
                if counts[slug] == 0:
                    continue
                qv = [[float(v) for v in e] for e in query_vecs[q][slug]][0]
                try:
                    results = col.query(
                        query_embeddings=[qv],
                        n_results=min(candidate_count, counts[slug]),
                        where=where_filter,
                    )
                except Exception as e:
                    print(f"Query failed on {slug}: {e}", file=sys.stderr)
                    continue
                if not results or not results["ids"] or not results["ids"][0]:
                    continue
                ids = results["ids"][0]
                sims = []
                for i, doc_id in enumerate(ids):
                    try:
                        sims.append(1.0 - float(results["distances"][0][i]))
                    except Exception:
                        sims.append(0.0)
                    entry = per_doc.setdefault(
                        doc_id,
                        {
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                        },
                    )
                # Decisiveness of this run: top similarity vs median.
                # A flat (uncertain) run votes quietly; a peaked run votes
                # loudly. Unsupervised, per-query, no labels, no tuning.
                ordered = sorted(sims, reverse=True)
                spread = max(ordered[0] - ordered[len(ordered) // 2], 1e-9)
                run_ranks.append({doc_id: i + 1 for i, doc_id in enumerate(ids)})
                run_spreads.append(spread)

        if not per_doc:
            return []

        rrf_k = configured_rrf_k()
        total_spread = sum(run_spreads) or 1.0
        raw_rrf: Dict[str, float] = {}
        for doc_ranks, spread in zip(run_ranks, run_spreads):
            w = spread / total_spread
            for doc_id, r in doc_ranks.items():
                raw_rrf[doc_id] = raw_rrf.get(doc_id, 0.0) + w / (rrf_k + r)
        lo, hi = min(raw_rrf.values()), max(raw_rrf.values())
        span = hi - lo

        doc_texts = [v["document"] for v in per_doc.values()]
        bm25_scores = (
            self.compute_bm25_scores(base_query, doc_texts)
            if hybrid
            else [0.0] * len(doc_texts)
        )
        # Lane decisiveness: a peaked BM25 field (standout lexical match)
        # takes the lead over dense; a flat field keeps default weights.
        # Same uncertainty principle as the model votes. Weights sum to 0.75.
        w_dense, w_bm25 = 0.45, 0.30
        if hybrid and bm25_scores:
            ordered_bm = sorted(bm25_scores, reverse=True)
            gap = ordered_bm[0] - ordered_bm[len(ordered_bm) // 2]
            shift = 0.30 * min(1.0, max(0.0, gap))
            w_bm25 = 0.30 + shift
            w_dense = 0.45 - shift
        q_cdata = cdata_terms_of(base_query)

        fused = []
        for (doc_id, v), sparse in zip(per_doc.items(), bm25_scores):
            dense = (raw_rrf[doc_id] - lo) / span if span > 0 else 1.0
            meta = v["metadata"]
            recency = self.compute_recency_score(
                meta.get("created_at", str(time.time()))
            )
            importance = self.compute_importance_score(meta.get("importance", "medium"))
            if hybrid:
                final = (
                    (w_dense * dense)
                    + (w_bm25 * sparse)
                    + (0.15 * recency)
                    + (0.10 * importance)
                )
            else:
                final = dense
            if q_cdata:
                hay = str(v["document"]).lower()
                hits = sum(1 for t in q_cdata if t in hay)
                if hits:
                    final = min(1.0, final + EXACT_LANE_WEIGHT * hits / len(q_cdata))
            fused.append(
                {
                    "id": doc_id,
                    "document": v["document"],
                    "metadata": meta,
                    "dense": dense,
                    "sparse": sparse,
                    "recency": recency,
                    "importance": importance,
                    "score": final,
                }
            )
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused

    def search(
        self, query: str, k: int = 3, project: str = None, hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """Semantic/Hybrid search across indexed wikis with Recency & Importance weighting (v2).

        RTR: multi-query expansion (pillar 1) fused over models with RRF,
        plus an optional PRF second pass (OPENCODE_MEMORY_PRF=1).
        """
        lex_subs, delta_subs = self.build_expansions(query, project)
        subs = [query] + [s for s in lex_subs if s != query]
        # Relevance gate (delta lane only): a delta-guess subquery votes only
        # if at least one of its added terms appears in the top-2
        # original-query docs. Lexikon subs are curated knowledge and bypass.
        # Without evidence a guess is drift, not expansion — bench proved it.
        try:
            first = self._fused_multi([query], k=k, project=project, hybrid=hybrid)[:2]
        except Exception:
            first = []
        if delta_subs and first:
            qtok = set(query.lower().split())
            hay = " ".join(str(f["document"]) for f in first).lower()
            for s in delta_subs:
                added = [t for t in s.lower().split() if t not in qtok]
                if added and any(t in hay for t in added) and s not in subs:
                    subs.append(s)
        if os.environ.get("OPENCODE_MEMORY_PRF", "0").strip() == "1":
            try:
                first = self._fused_multi(subs, k=k, project=project, hybrid=hybrid)[:2]
                toks: List[str] = []
                for f in first:
                    toks.extend(
                        t
                        for t in re.findall(
                            r"\w+", str(f["metadata"].get("title", "")).lower()
                        )
                        if len(t) > 2
                    )
                extra = [t for t in dict.fromkeys(toks) if t not in query.lower()][:8]
                if extra:
                    cand = query + " " + " ".join(extra)
                    if cand not in subs:
                        subs = subs + [cand]
            except Exception:
                pass
        fused = self._fused_multi(subs, k=k, project=project, hybrid=hybrid)
        if not fused:
            return []

        formatted = [
            {
                "id": f["id"],
                "document": f["document"],
                "metadata": f["metadata"],
                "score": f["score"],
            }
            for f in fused[:k]
        ]

        # Expand sources using Obsidian wikilinks (across all model collections)
        seen_ids = {f["id"] for f in formatted}
        for res in list(formatted):
            links = res["metadata"].get("linked_concepts", "")
            if not links:
                continue
            for concept in links.split(","):
                concept = concept.strip()
                if not concept:
                    continue
                for col in self.collections.values():
                    linked = col.get(where={"$contains": concept})
                    if linked and linked["ids"]:
                        for j, linked_id in enumerate(linked["ids"]):
                            if linked_id in seen_ids:
                                continue
                            is_rel = linked["metadatas"][j].get("rel_path", "") != res[
                                "metadata"
                            ].get("rel_path", "")
                            if not is_rel:
                                continue
                            seen_ids.add(linked_id)
                            formatted.append(
                                {
                                    "id": linked_id,
                                    "document": linked["documents"][j],
                                    "metadata": linked["metadatas"][j],
                                    "score": res["score"] * 0.9,
                                }
                            )

        return formatted

    def explain_search(
        self, query: str, k: int = 3, project: str = None
    ) -> List[Dict[str, Any]]:
        """Explain search query ranking parameters with detailed mathematical score breakdowns.

        The `dense` column is the min-max normalized RRF across configured models.
        """
        fused = self._fused_multi(
            self.build_subqueries(query, project), k=k, project=project, hybrid=True
        )
        if not fused:
            return []

        formatted = []
        for f in fused[:k]:
            meta = f["metadata"]
            formatted.append(
                {
                    "rel_path": meta["rel_path"],
                    "title": meta.get("title", "Unknown"),
                    "dense": f["dense"],
                    "bm25": f["sparse"],
                    "recency": f["recency"],
                    "importance": f["importance"],
                    "final": f["score"],
                }
            )

        formatted.sort(key=lambda x: x["final"], reverse=True)
        return formatted[:k]

    def write_wiki(
        self, project: str, filename: str, content: str
    ) -> Tuple[str, List[str]]:
        """Write new wiki entry to project directives folder and auto-link related docs."""
        project_dir = resolve_project_dir(project)
        if not project_dir.exists():
            raise ValueError(f"Project directory {project_dir} does not exist.")

        directives_dir = project_dir / "directives"

        # Check if wiki_concepts exists inside directives, otherwise write directly in directives
        concepts_dir = directives_dir / "wiki_concepts"
        if concepts_dir.exists():
            target_dir = concepts_dir
        else:
            target_dir = directives_dir
            target_dir.mkdir(parents=True, exist_ok=True)

        # Ensure filename ends with .md
        if not filename.endswith(".md"):
            filename += ".md"

        filepath = target_dir / filename

        # Clean markdown formatting (normalize line breaks, etc.)
        content = content.strip()

        # Step 1: Write initial content to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # Step 2: Temporarily index this file so we can find links, OR search before indexing it
        # Actually, let's search for related documents across ALL projects first
        related_links = []
        try:
            # Rebuild index of all OTHER files first to ensure search is fresh
            self.index_all()

            # Search using the title/intro of the new wiki entry
            # Extract first 200 chars or title
            search_query = filename.replace(".md", "").replace("_", " ")
            search_results = self.search(query=search_query, k=4, project=None)

            # Filter results to find highly relevant docs (score > 0.65) and exclude the file itself
            seen_paths = set()
            for res in search_results:
                src_path = Path(res["metadata"]["source"])
                if src_path == filepath:
                    continue

                score = res["score"]
                if score > 0.50 and src_path not in seen_paths:
                    seen_paths.add(src_path)

                    # Calculate relative path from target_dir to src_path
                    try:
                        # Find path relative to PROJECTS_DIR or just a clean relative path
                        # If both are in projects, we can compute relative path
                        rel_link = os.path.relpath(src_path, start=target_dir)
                        # Normalize path separators for markdown
                        rel_link = rel_link.replace("\\", "/")
                        related_links.append(
                            f"- [{src_path.name}]({rel_link}) (Similarity: {score:.2f})"
                        )
                    except Exception:
                        related_links.append(
                            f"- [{src_path.name}](file://{src_path.as_posix()}) (Similarity: {score:.2f})"
                        )
        except Exception as e:
            print(f"Auto-linking failed: {e}", file=sys.stderr)

        # Step 3: Append related docs section if any found
        if related_links:
            related_section = (
                "\n\n## Related Knowledge\n" + "\n".join(related_links) + "\n"
            )
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(related_section)

        # Step 4: Re-index specifically to get the final chunk with related links
        self.index_all()

        return str(filepath), related_links

    def janitor(self, dry_run: bool = False) -> Dict[str, Any]:
        """Delete chunks whose source files no longer exist on disk.

        Catches orphans the hash-registry purge cannot see (model switches,
        deleted temp projects). Reports per-collection freed counts.
        """
        report: Dict[str, Any] = {"collections": {}, "freed": 0, "dry_run": dry_run}
        for name, col in self.rtr_collections().items():
            try:
                total = col.count()
            except Exception:
                continue
            if total == 0:
                report["collections"][name] = {"chunks": 0, "freed": 0}
                continue
            try:
                got = col.get(limit=total + 1000, include=["metadatas"])
            except Exception as e:
                report["collections"][name] = {"error": str(e)}
                continue
            by_source: Dict[str, List[str]] = {}
            for cid, meta in zip(got["ids"], got["metadatas"] or []):
                src = (meta or {}).get("source", "")
                by_source.setdefault(src, []).append(cid)
            freed = 0
            for src, ids in by_source.items():
                if src and not Path(src).exists():
                    if not dry_run:
                        try:
                            col.delete(ids=ids)
                        except Exception:
                            continue
                    freed += len(ids)
            report["collections"][name] = {"chunks": total, "freed": freed}
            report["freed"] += freed
        return report

    def stats(self) -> Dict[str, Any]:
        """Retrieve corpus stats (chunk counts summed across model collections)."""
        registry = self.load_registry()

        total_chunks = sum(col.count() for col in self.collections.values())
        total_files = len(registry)

        project_counts = {}
        for path, _ in registry.items():
            key, _, _ = describe_indexed_path(path)
            project_counts[key] = project_counts.get(key, 0) + 1

        db_size_bytes = 0
        db_home = _db_dir()
        if db_home.exists():
            for f in db_home.glob("**/*"):
                if f.is_file():
                    db_size_bytes += f.stat().st_size

        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "projects": project_counts,
            "db_size_mb": db_size_bytes / (1024 * 1024),
        }

    def consolidate_wiki(self, project: str) -> Dict[str, Any]:
        """Deduplicate and consolidate redundant wiki markdown files under a project's directives/wiki_concepts or design_wiki_concepts."""
        from collections import defaultdict

        project_dir = resolve_project_dir(project)
        if not project_dir.exists():
            return {
                "status": "error",
                "message": f"Project directory {project_dir} does not exist.",
            }

        directives_dir = project_dir / "directives"
        concepts_dir = directives_dir / "wiki_concepts"
        if not concepts_dir.exists():
            concepts_dir = directives_dir / "design_wiki_concepts"

        if not concepts_dir.exists():
            return {
                "status": "error",
                "message": "No wiki_concepts/ or design_wiki_concepts/ directory found.",
            }

        md_files = list(concepts_dir.glob("**/*.md"))
        if len(md_files) < 2:
            return {
                "status": "success",
                "message": "Fewer than 2 concept files. No consolidation needed.",
                "merged": [],
            }

        merged_count = 0
        merged_details = []

        file_contents = {}
        for f in md_files:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    file_contents[f] = file.read()
            except Exception:
                pass

        already_merged = set()
        for i in range(len(md_files)):
            f1 = md_files[i]
            if f1 in already_merged or f1 not in file_contents:
                continue

            for j in range(i + 1, len(md_files)):
                f2 = md_files[j]
                if f2 in already_merged or f2 not in file_contents:
                    continue

                stem1 = f1.stem.lower().replace("_", " ").replace("-", " ")
                stem2 = f2.stem.lower().replace("_", " ").replace("-", " ")

                words1 = set(re.findall(r"\w+", stem1))
                words2 = set(re.findall(r"\w+", stem2))
                common_words = words1.intersection(words2)

                is_duplicate = False
                if (
                    len(common_words) >= min(len(words1), len(words2))
                    and len(common_words) > 0
                ):
                    is_duplicate = True

                if is_duplicate:
                    # Target the file with the shorter filename as the master concept
                    if len(f1.name) > len(f2.name):
                        f1, f2 = f2, f1

                    content1 = file_contents[f1].strip()
                    content2 = file_contents[f2].strip()

                    merged_content = f"{content1}\n\n## Consolidated Concept: {f2.stem.replace('_', ' ').title()}\n{content2}"

                    try:
                        with open(f1, "w", encoding="utf-8") as file:
                            file.write(merged_content)

                        f2.unlink()
                        already_merged.add(f2)

                        merged_count += 1
                        merged_details.append(
                            f"Merged duplicate {f2.name} into {f1.name}"
                        )
                        file_contents[f1] = merged_content
                    except Exception as e:
                        print(f"Error merging {f2} into {f1}: {e}", file=sys.stderr)

        if merged_count > 0:
            self.index_all(force=True)

        return {
            "status": "success",
            "message": f"Consolidated {merged_count} duplicate wiki documents.",
            "merged": merged_details,
        }

    def init_project(self, project_name: str, variant: str = "wiki") -> Dict[str, Any]:
        """Bootstrap the Harness Engineering Framework structure in a new or existing project folder."""
        project_dir = resolve_project_dir(project_name)

        # 1. Create Directories
        dirs_to_create = [
            project_dir / "directives",
            project_dir
            / "directives"
            / ("wiki_concepts" if variant == "wiki" else "design_wiki_concepts"),
            project_dir / "execution" / "core",
            project_dir / "execution" / "utils",
            project_dir / "execution" / "scratch",
            project_dir / "state",
            project_dir / "project",
        ]

        created_dirs = []
        for d in dirs_to_create:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                created_dirs.append(
                    str(
                        describe_indexed_path(str(d))[1] if d != project_dir else d.name
                    )
                )

        # 2. Write standard documents
        files_written = []

        agents_md = project_dir / "AGENTS.md"
        if not agents_md.exists():
            agents_md.write_text(
                f"# Harness Engineering Framework — {project_name}\n\nMUST read at EVERY session start AND end.\n\n## Project Overview\nThis repository contains the {project_name} workspace structured under a deterministic 3-layer architecture.\n\n## Lifecycle Status\n| Phase | Status | Description |\n|---|---|---|\n| PHASE_ONE | `[ ]` Pending | Initialization and Setup |\n\n## Lifecycle Protocol\n- **Start:** Read `directives/progress.md` and `directives/session-handoff.md`\n- **End:** Save learnings to inbox, run wiki maintainer, and update `directives/progress.md`\n",
                encoding="utf-8",
            )
            files_written.append("AGENTS.md")

        claude_md = project_dir / "CLAUDE.md"
        if not claude_md.exists():
            claude_md.write_text(
                "# CLAUDE.md\nRefer to [AGENTS.md](AGENTS.md) for workspace lifecycle details.\n",
                encoding="utf-8",
            )
            files_written.append("CLAUDE.md")

        # Create wiki inbox
        inbox_name = "wiki_inbox.md" if variant == "wiki" else "design_wiki.md"
        inbox_file = project_dir / "directives" / inbox_name
        if not inbox_file.exists():
            label = "Wiki Inbox" if variant == "wiki" else "Design Wiki Inbox"
            inbox_file.write_text(
                f"# {label}\n\nAppend raw lessons, API observations, and style guidelines here.\n",
                encoding="utf-8",
            )
            files_written.append(f"directives/{inbox_name}")

        # Create progress.md
        progress_file = project_dir / "directives" / "progress.md"
        if not progress_file.exists():
            progress_file.write_text(
                f"# Project Progress Log — {project_name}\n\n- **2026-06-01:** System bootstrapped using Local-Memory-RAG MCP bootstrapper.\n",
                encoding="utf-8",
            )
            files_written.append("directives/progress.md")

        # Create session-handoff.md
        handoff_file = project_dir / "directives" / "session-handoff.md"
        if not handoff_file.exists():
            handoff_file.write_text(
                f"# Session Handoff\n\n## Next Action\nComplete phase one requirements.\n\n## CLI Startup Commands\n`make check`\n",
                encoding="utf-8",
            )
            files_written.append("directives/session-handoff.md")

        # Create state files
        state_file = project_dir / "state" / "workflow_state.json"
        if not state_file.exists():
            import json

            state_data = {
                "current_phase": "PHASE_ONE",
                "completed_steps": [],
                "history": [],
            }
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
            files_written.append("state/workflow_state.json")

        feature_file = project_dir / "state" / "feature_list.json"
        if not feature_file.exists():
            import json

            feature_data = {
                "phases": [
                    {"id": "PHASE_ONE", "name": "System Setup", "completed": False}
                ]
            }
            with open(feature_file, "w", encoding="utf-8") as f:
                json.dump(feature_data, f, indent=2)
            files_written.append("state/feature_list.json")

        # Create bootstrap scripts
        init_sh = project_dir / "init.sh"
        if not init_sh.exists():
            init_sh.write_text(
                '#!/bin/bash\n# Bootstrap virtual environment and workspace\necho "Bootstrapping Harness environment..."\npython3 -m venv .venv\nsource .venv/bin/activate\nif [ -f requirements.txt ]; then\n  pip install -r requirements.txt\nfi\necho "Workspace bootstrap complete."\n',
                encoding="utf-8",
            )
            try:
                os.chmod(init_sh, 0o755)
            except Exception:
                pass
            files_written.append("init.sh")

        gitignore = project_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(
                ".venv/\n.tmp/\n.env\n__pycache__/\n", encoding="utf-8"
            )
            files_written.append(".gitignore")

        return {
            "status": "success",
            "message": f"Successfully initialized Harness framework in {project_name}.",
            "created_directories": created_dirs,
            "created_files": files_written,
        }

    def restore_session_state(self, project: str) -> Dict[str, Any]:
        """Session Restoration Engine (Phase 1): Parses Harness state logs and auto-restores context."""
        project_dir = resolve_project_dir(project)
        if not project_dir.exists():
            return {
                "status": "error",
                "message": f"Project directory {project_dir} does not exist.",
            }

        directives = project_dir / "directives"
        state = project_dir / "state"

        progress = directives / "progress.md"
        handoff = directives / "session-handoff.md"
        goals = state / "current-goals.md"
        issues = state / "open-issues.md"
        decisions = state / "active-decisions.md"

        active_goals = []
        blockers = []
        discoveries = []
        full_content = ""

        def parse_items(filepath, only_bullet_points=True):
            if not filepath.exists():
                return []
            try:
                content = filepath.read_text(encoding="utf-8")
                items = []
                for line in content.split("\n"):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if only_bullet_points:
                        if line.startswith("-"):
                            clean = re.sub(r"^-\s*(?:\[[ xX/]\])?\s*", "", line)
                            if clean.strip():
                                items.append(clean.strip())
                    else:
                        clean = re.sub(r"^-\s*(?:\[[ xX/]\])?\s*", "", line)
                        if clean.strip():
                            items.append(clean.strip())
                return items
            except Exception:
                return []

        if goals.exists():
            active_goals.extend(parse_items(goals, only_bullet_points=True))
            full_content += goals.read_text(encoding="utf-8")
        elif handoff.exists():
            handoff_content = handoff.read_text(encoding="utf-8")
            match = re.search(
                r"## Next Action\s*\n\s*(?:-\s*)?([^\n#]+)", handoff_content
            )
            if match:
                active_goals.append(match.group(1).strip())
            full_content += handoff_content

        if issues.exists():
            blockers.extend(parse_items(issues, only_bullet_points=True))
            full_content += issues.read_text(encoding="utf-8")

        if decisions.exists():
            discoveries.extend(parse_items(decisions, only_bullet_points=False))
            full_content += decisions.read_text(encoding="utf-8")

        if progress.exists():
            full_content += "\n" + progress.read_text(encoding="utf-8")

        keywords = " ".join(re.findall(r"\w+", full_content.strip())[:15])

        recommended_files = []
        if keywords:
            try:
                results = self.search(keywords, k=4, project=project)
                recommended_files = [res["metadata"]["rel_path"] for res in results]
            except Exception:
                pass

        return {
            "status": "success",
            "project": project,
            "active_goals": active_goals[:5],
            "blockers": blockers[:5],
            "recent_discoveries": discoveries[:5],
            "recommended_files": recommended_files,
            "query_context": keywords[:60],
        }

    def _get_wiki_paths(self, project: str) -> Dict[str, Path]:
        """Detect wiki naming convention for a project."""
        directives = resolve_project_dir(project) / "directives"
        design_concepts = directives / "design_wiki_concepts"
        wiki_concepts = directives / "wiki_concepts"

        if design_concepts.exists():
            return {
                "inbox": directives / "design_wiki.md",
                "concepts": design_concepts,
                "variant": "design_wiki",
            }
        return {
            "inbox": directives / "wiki_inbox.md",
            "concepts": wiki_concepts,
            "variant": "wiki",
        }

    def detect_workspace_deltas(self, project: str) -> Dict[str, List[str]]:
        """Scan project's directives/ for any new, modified, or deleted files compared to wiki_registry.json."""
        project_dir = resolve_project_dir(project)
        if not project_dir.exists():
            return {"created": [], "modified": [], "deleted": []}

        directives_dir = project_dir / "directives"
        registry = self.load_registry()

        discovered_paths = set()
        created = []
        modified = []

        if directives_dir.exists():
            for root, _, files in os.walk(directives_dir):
                root_path = Path(root)
                # Skip common noise dirs
                if any(
                    part in root_path.parts
                    for part in (
                        ".git",
                        "node_modules",
                        ".venv",
                        "venv",
                        "__pycache__",
                        ".local_vector_db",
                    )
                ):
                    continue
                for file in files:
                    if not _matches_suffix(file):
                        continue
                    file_path = root_path / file
                    path_str = str(file_path)
                    discovered_paths.add(path_str)

                    if not file_path.exists():
                        continue

                    current_hash = self.calculate_sha256(file_path)
                    old_hash = registry.get(path_str)

                    if old_hash is None:
                        created.append(path_str)
                    elif old_hash != current_hash:
                        modified.append(path_str)

        # Detect deleted files for this project
        deleted = []
        want_key = project_filter_key(project)
        for path_str in registry.keys():
            key, rel, _ = describe_indexed_path(path_str)
            if key == want_key and "directives" in rel:
                if path_str not in discovered_paths:
                    deleted.append(path_str)

        return {"created": created, "modified": modified, "deleted": deleted}

    def reflect_session(
        self, project: str, conversation_text: str = ""
    ) -> Dict[str, Any]:
        """Perform Session Reflection. Scans workspace deltas and extracts event-driven knowledge events."""
        deltas = self.detect_workspace_deltas(project)

        knowledge_events = []

        # 1. Automatically classify concepts from workspace file deltas
        for path_str in deltas["created"]:
            p = Path(path_str)
            if "wiki_concepts" in p.parts or "design_wiki_concepts" in p.parts:
                concept_name = p.stem.replace("_", " ").replace("-", " ").title()
                knowledge_events.append(
                    {
                        "type": "decision",
                        "concept": concept_name,
                        "evidence": f"Created concept node: {p.name}",
                        "confidence": 1,
                    }
                )

        for path_str in deltas["modified"]:
            p = Path(path_str)
            if "wiki_concepts" in p.parts or "design_wiki_concepts" in p.parts:
                concept_name = p.stem.replace("_", " ").replace("-", " ").title()
                knowledge_events.append(
                    {
                        "type": "observation",
                        "concept": concept_name,
                        "evidence": f"Modified concept node: {p.name}",
                        "confidence": 1,
                    }
                )

        # 2. Try parsing structured data from conversation_text (JSON / YAML / Markdown sections)
        parsed_structured = False
        text_clean = conversation_text.strip()

        # Check for JSON
        if text_clean.startswith("{") and text_clean.endswith("}"):
            try:
                data = json.loads(text_clean)
                if "knowledge_events" in data:
                    knowledge_events.extend(data["knowledge_events"])
                    parsed_structured = True
            except Exception:
                pass

        # Check for YAML-like list block of knowledge_events
        if not parsed_structured:
            events_match = re.findall(
                r"-\s*type:\s*(\w+)\s*\n\s*concept:\s*([^\n]+)(?:\s*\n\s*evidence:\s*([^\n]+))?",
                conversation_text,
            )
            if events_match:
                for t, c, ev in events_match:
                    knowledge_events.append(
                        {
                            "type": t.strip(),
                            "concept": c.strip(),
                            "evidence": ev.strip() if ev else "Session logs",
                            "confidence": 1,
                        }
                    )
                parsed_structured = True

        # Heuristic extraction if raw unformatted text
        if not parsed_structured and conversation_text:
            sentences = re.split(r"[.!?\n]+", conversation_text)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean or len(s_clean) < 10:
                    continue

                lower_s = s_clean.lower()
                event_type = None

                if any(
                    k in lower_s
                    for k in ["we think", "we believe", "it might", "hypothesis"]
                ):
                    event_type = "hypothesis"
                elif any(
                    k in lower_s
                    for k in ["we tested", "we tried", "testing", "experiment"]
                ):
                    event_type = "experiment"
                elif any(
                    k in lower_s
                    for k in [
                        "improved",
                        "success",
                        "confirmed",
                        "validated",
                        "validation",
                    ]
                ):
                    event_type = "validation"
                elif any(
                    k in lower_s
                    for k in ["failed", "bug", "error", "produced noisy", "failure"]
                ):
                    event_type = "failure"
                elif any(k in lower_s for k in ["we decided", "let's use", "decision"]):
                    event_type = "decision"
                elif any(
                    k in lower_s
                    for k in [
                        "deprecated",
                        "no longer recommended",
                        "retired",
                        "deprecation",
                    ]
                ):
                    event_type = "deprecation"
                elif any(
                    k in lower_s
                    for k in [
                        "observed",
                        "noticed",
                        "discovered",
                        "learned",
                        "observation",
                    ]
                ):
                    event_type = "observation"

                if event_type:
                    matched_concept = "General Learning"
                    quoted = re.findall(r"['\"`]([^'\n`\"]+)['\"`]", s_clean)
                    if quoted:
                        matched_concept = quoted[0].title()
                    else:
                        registry = self._load_concept_registry(project)
                        for cid, data in registry.items():
                            cname = data["canonical_name"].lower()
                            if cname.replace("-", " ") in lower_s:
                                matched_concept = data["canonical_name"].title()
                                break
                            for alias in data.get("aliases", []):
                                if alias.lower() in lower_s:
                                    matched_concept = alias.title()
                                    break

                    knowledge_events.append(
                        {
                            "type": event_type,
                            "concept": matched_concept,
                            "evidence": s_clean,
                            "confidence": 1,
                        }
                    )

        # Deduplicate events
        seen_events = set()
        deduped_events = []
        for ev in knowledge_events:
            key = (ev["type"].lower(), ev["concept"].lower(), ev["evidence"].lower())
            if key not in seen_events:
                seen_events.add(key)
                deduped_events.append(ev)

        # Write Session Learning Report
        project_dir = resolve_project_dir(project)
        sessions_dir = project_dir / "directives" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        date_str = time.strftime("%Y-%m-%d")
        report_file = sessions_dir / f"{date_str}.md"

        if report_file.exists():
            report_file = sessions_dir / f"{date_str}_{int(time.time() % 100000)}.md"

        import yaml

        yaml_content = yaml.dump(
            {"knowledge_events": deduped_events}, default_flow_style=False
        )

        report_content = f"""---
date: {date_str}
project: {project}
---
# Session Learning Report — {date_str}

## Knowledge Events
```yaml
{yaml_content.strip()}
```
"""
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        return {
            "status": "success",
            "report_path": str(report_file),
            "knowledge_events": deduped_events,
        }

    def _load_concept_registry(self, project: str) -> Dict[str, Any]:
        """Load project's state/concept_registry.json."""
        project_dir = resolve_project_dir(project)
        reg_file = project_dir / "state" / "concept_registry.json"
        if reg_file.exists():
            try:
                with open(reg_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_concept_registry(self, project: str, registry: Dict[str, Any]):
        """Save project's state/concept_registry.json."""
        project_dir = resolve_project_dir(project)
        reg_dir = project_dir / "state"
        reg_dir.mkdir(parents=True, exist_ok=True)
        reg_file = reg_dir / "concept_registry.json"
        try:
            with open(reg_file, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            print(f"Error saving concept registry: {e}", file=sys.stderr)

    # --- RTR Pillar 3: Lexikon/Glossary ---
    # Lookup-first terminology: canonical terms + aliases + definitions, merged
    # from the _global user lexikon and the project's own. Query expansion
    # becomes a lookup, not inference — the models stay chilly.
    def _lexikon_files_for(self, project: str) -> List[Path]:
        files = [LEXIKON_GLOBAL_FILE]
        if str(project or "").strip() not in ("", "_global"):
            try:
                d = resolve_project_dir(project)
                if d.exists():
                    files.append(d / "state" / "lexikon.json")
            except Exception:
                pass
        return files

    @staticmethod
    def _read_lexikon_file(fp: Path) -> Dict[str, Any]:
        try:
            if fp.exists():
                data = json.loads(fp.read_text(encoding="utf-8"))
                terms = data.get("terms") or {}
                return {str(k).lower(): v for k, v in terms.items()}
        except Exception:
            pass
        return {}

    def get_lexikon(self, project: str) -> Dict[str, Any]:
        """Merged lexikon: _global first, project entries win."""
        merged: Dict[str, Any] = {}
        for fp in self._lexikon_files_for(project):
            merged.update(self._read_lexikon_file(fp))
        return merged

    def add_lexikon_term(
        self,
        project: str,
        term: str,
        definition: str = "",
        aliases: List[str] = None,
    ) -> Dict[str, Any]:
        """Hand-add a lexikon entry (source=hand wins over derived ones)."""
        term = str(term or "").strip()
        if not term:
            return {"status": "error", "message": "Empty term."}
        if str(project or "").strip() in ("", "_global"):
            fp = LEXIKON_GLOBAL_FILE
        else:
            d = resolve_project_dir(project)
            fp = d / "state" / "lexikon.json"
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if fp.exists():
                data = json.loads(fp.read_text(encoding="utf-8"))
            terms = data.get("terms") or {}
            terms[term.lower()] = {
                "term": term,
                "definition": definition or "",
                "aliases": sorted({a.strip() for a in (aliases or []) if a.strip()}),
                "source": "hand",
                "updated": time.strftime("%Y-%m-%d %H:%M"),
            }
            data["terms"] = terms
            fp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return {"status": "success", "message": f"Lexikon entry '{term}' saved."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to save lexikon: {e}"}

    def build_lexikon(self, project: str) -> Dict[str, Any]:
        """Derive project lexikon from registry + latest session report.

        Hand entries (source=hand) always win over derived ones.
        """
        if str(project or "").strip() in ("", "_global"):
            return {
                "status": "error",
                "message": "The _global lexikon is hand-curated only.",
            }
        d = resolve_project_dir(project)
        if not d.exists():
            return {"status": "error", "message": f"Project dir {d} missing."}
        fp = d / "state" / "lexikon.json"
        existing = self._read_lexikon_file(fp)
        derived: Dict[str, Any] = {}
        try:
            for _cid, cdata in self._load_concept_registry(project).items():
                canon = str(cdata.get("canonical_name", "")).strip()
                if not canon:
                    continue
                derived[canon.lower()] = {
                    "term": canon,
                    "definition": "",
                    "aliases": sorted(
                        {str(a) for a in (cdata.get("aliases") or []) if str(a).strip()}
                    ),
                    "source": "registry",
                    "updated": time.strftime("%Y-%m-%d %H:%M"),
                }
        except Exception:
            pass
        try:
            sessions = sorted((d / "directives" / "sessions").glob("*.md"))
            if sessions:
                import yaml

                content = sessions[-1].read_text(encoding="utf-8")
                m = re.search(r"```yaml(.*?)```", content, re.DOTALL)
                if m:
                    events = (yaml.safe_load(m.group(1)) or {}).get(
                        "knowledge_events", []
                    )
                    for ev in events or []:
                        concept = str((ev or {}).get("concept", "")).strip()
                        if concept and concept.lower() not in derived:
                            derived[concept.lower()] = {
                                "term": concept,
                                "definition": str((ev or {}).get("evidence", ""))[:200],
                                "aliases": [],
                                "source": "session",
                                "updated": time.strftime("%Y-%m-%d %H:%M"),
                            }
        except Exception:
            pass
        for key, entry in derived.items():
            if key not in existing or existing[key].get("source") != "hand":
                existing[key] = entry
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps({"terms": existing}, indent=2), encoding="utf-8")
        except Exception as e:
            return {"status": "error", "message": f"Failed to write lexikon: {e}"}
        return {
            "status": "success",
            "message": f"Lexikon built: {len(existing)} terms.",
            "terms": len(existing),
        }

    def lexikon_expansion(self, project: str, query: str, limit: int = 8) -> List[str]:
        """Lookup-only expansion terms: lexikon hits overlapping the query."""
        q = str(query or "").lower()
        qtokens = {t for t in re.findall(r"\w+", q) if len(t) > 2}
        if not qtokens:
            return []
        out = []
        try:
            lex = self.get_lexikon(project)
        except Exception:
            return []
        for _key, entry in lex.items():
            if not isinstance(entry, dict):
                continue
            term = str(entry.get("term", ""))
            aliases = [str(a) for a in (entry.get("aliases") or [])]
            hay = f"{term} {' '.join(aliases)}".lower()
            htokens = {t for t in re.findall(r"\w+", hay) if len(t) > 2}
            if qtokens & htokens or any(a.lower() in q for a in aliases if a):
                for t in re.findall(r"\w+", str(entry.get("definition", "")).lower()):
                    if len(t) > 2 and t not in qtokens and t not in out:
                        out.append(t)
                    if len(out) >= limit:
                        return out
                for a in aliases:
                    al = a.lower()
                    if al not in qtokens and al not in out and len(al) > 2:
                        out.append(al)
                    if len(out) >= limit:
                        return out
        return out

    def _resolve_concept(self, project: str, term: str) -> Tuple[str, Dict[str, Any]]:
        """
        Resolve a term (canonical name or alias) against the registry.
        Returns (concept_id, concept_data) if found, otherwise returns (new_id, new_data_stub).
        """
        registry = self._load_concept_registry(project)
        term_clean = term.strip().lower()

        # 1. Direct match on canonical name or aliases
        for cid, data in registry.items():
            canonical = data.get("canonical_name", "").lower()
            aliases = [a.lower() for a in data.get("aliases", [])]
            if term_clean == canonical or term_clean in aliases:
                return cid, data

        # 2. Fast semantic/substring check
        for cid, data in registry.items():
            canonical = data.get("canonical_name", "").lower()
            c_words = set(re.findall(r"\w+", canonical))
            t_words = set(re.findall(r"\w+", term_clean))
            if (
                c_words
                and t_words
                and len(c_words.intersection(t_words))
                >= max(1, min(len(c_words), len(t_words)))
            ):
                if term not in data.get("aliases", []):
                    data.setdefault("aliases", []).append(term)
                    self._save_concept_registry(project, registry)
                return cid, data

        # 3. Create a new concept stub if not found
        next_num = len(registry) + 1
        new_id = f"concept_{next_num:03d}"

        canonical_name = (
            re.sub(r"[^a-zA-Z0-9\s-]", "", term).strip().replace(" ", "-").lower()
        )
        if not canonical_name:
            canonical_name = f"concept-{next_num}"

        new_data = {
            "canonical_name": canonical_name,
            "aliases": [term],
            "status": "candidate",
            "confidence": 1,
            "evidence_count": 0,
            "session_count": 0,
        }

        registry[new_id] = new_data
        self._save_concept_registry(project, registry)
        return new_id, new_data

    def materialize_concepts(self, project: str) -> Dict[str, Any]:
        """Parse session learning reports, update Concept Registry, and materialize validated concept nodes."""
        project_dir = resolve_project_dir(project)
        if not project_dir.exists():
            return {
                "status": "error",
                "message": f"Project directory {project_dir} does not exist.",
            }

        directives_dir = project_dir / "directives"
        sessions_dir = directives_dir / "sessions"
        if not sessions_dir.exists():
            return {
                "status": "success",
                "message": "No session reports found.",
                "materialized": [],
            }

        paths = self._get_wiki_paths(project)
        concepts_dir = paths["concepts"]
        concepts_dir.mkdir(parents=True, exist_ok=True)

        session_files = sorted(sessions_dir.glob("*.md"))
        if not session_files:
            return {
                "status": "success",
                "message": "No session reports found.",
                "materialized": [],
            }

        latest_report = session_files[-1]

        try:
            content = latest_report.read_text(encoding="utf-8")
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read latest session report: {e}",
            }

        date_str = time.strftime("%Y-%m-%d")
        date_match = re.search(r"date:\s*([^\s\n]+)", content)
        if date_match:
            date_str = date_match.group(1).strip()

        # Parse YAML block of knowledge_events
        knowledge_events = []
        yaml_match = re.search(r"```yaml\n(.*?)\n```", content, re.DOTALL)
        if yaml_match:
            try:
                import yaml

                data = yaml.safe_load(yaml_match.group(1))
                if isinstance(data, dict) and "knowledge_events" in data:
                    knowledge_events = data["knowledge_events"]
            except Exception as e:
                print(f"Error parsing YAML from report: {e}", file=sys.stderr)

        if not knowledge_events:
            return {
                "status": "success",
                "message": "No knowledge events found to materialize.",
                "materialized": [],
            }

        registry = self._load_concept_registry(project)
        materialized_log = []

        import yaml

        # Process each event and update registry
        for event in knowledge_events:
            concept_name = event.get("concept", "General Learning")
            event_type = event.get("type", "observation").lower()
            evidence = event.get("evidence", "")

            cid, cdata = self._resolve_concept(project, concept_name)

            cdata["evidence_count"] = cdata.get("evidence_count", 0) + (
                1 if evidence else 0
            )
            cdata["session_count"] = cdata.get("session_count", 0) + 1

            confidence = cdata.get("confidence", 1)
            if event_type == "validation":
                confidence = min(5, confidence + 1)
            elif event_type == "failure":
                confidence = max(1, confidence - 1)
            cdata["confidence"] = confidence

            current_status = cdata.get("status", "candidate")

            if event_type == "deprecation":
                new_status = "deprecated"
            elif cdata["session_count"] >= 5 and cdata["confidence"] >= 4:
                new_status = "canonical"
            elif cdata["evidence_count"] >= 3 or current_status == "validated":
                new_status = "validated"
            elif cdata["session_count"] >= 2:
                new_status = "emerging"
            else:
                new_status = "candidate"

            cdata["status"] = new_status

            registry[cid] = cdata
            self._save_concept_registry(project, registry)

            concept_file = concepts_dir / f"{cid}.md"

            if new_status in ["validated", "canonical"]:
                existing_body = ""
                is_new = not concept_file.exists()

                if not is_new:
                    try:
                        existing_text = concept_file.read_text(encoding="utf-8")
                        fm_match = re.match(
                            r"^---\s*\n(.*?)\n---\s*\n(.*)$", existing_text, re.DOTALL
                        )
                        if fm_match:
                            existing_body = fm_match.group(2).strip()
                        else:
                            existing_body = existing_text.strip()
                    except Exception:
                        pass

                learnings_text = (
                    f"- **{event_type.title()}**: {evidence}" if evidence else ""
                )

                if is_new:
                    body_content = f"# {cdata['canonical_name'].replace('-', ' ').title()}\n\nThis concept is a validated organizational knowledge node.\n\n## Learnings ({date_str})\n{learnings_text}\n"
                else:
                    body_content = existing_body
                    if learnings_text:
                        body_content += (
                            f"\n\n## Learnings ({date_str})\n{learnings_text}\n"
                        )

                fm_aliases = yaml.dump(
                    cdata.get("aliases", []), default_flow_style=True
                ).strip()
                concept_content = f"""---
concept_id: {cid}
canonical_name: {cdata["canonical_name"]}
status: {new_status}
confidence: {cdata["confidence"]}
evidence_count: {cdata["evidence_count"]}
session_count: {cdata["session_count"]}
aliases: {fm_aliases}
---
{body_content}"""

                try:
                    concept_file.write_text(
                        concept_content.strip() + "\n", encoding="utf-8"
                    )
                    materialized_log.append(
                        f"Materialized validated node: {cid} ({cdata['canonical_name']}) status={new_status}"
                    )
                except Exception as e:
                    print(f"Failed to materialize {cid}: {e}", file=sys.stderr)

            elif new_status == "deprecated":
                if concept_file.exists():
                    try:
                        concept_file.unlink()
                        materialized_log.append(
                            f"Archived/Deleted deprecated node: {cid} ({cdata['canonical_name']})"
                        )
                    except Exception as e:
                        print(
                            f"Failed to delete deprecated node {cid}: {e}",
                            file=sys.stderr,
                        )
            else:
                if concept_file.exists():
                    try:
                        concept_file.unlink()
                    except Exception:
                        pass
                materialized_log.append(
                    f"Registry updated: {cid} ({cdata['canonical_name']}) status={new_status} (Not materialized on disk)"
                )

        return {"status": "success", "materialized": materialized_log}

    def update_concept_graph(self, project: str) -> Dict[str, Any]:
        """Analyze all materialized concept nodes in a project and establish reciprocal bidirectional ID-based links: [[concept_XXX|Canonical Name]]."""
        project_dir = resolve_project_dir(project)
        if not project_dir.exists():
            return {
                "status": "error",
                "message": f"Project directory {project_dir} does not exist.",
            }

        directives_dir = project_dir / "directives"
        paths = self._get_wiki_paths(project)
        concepts_dir = paths["concepts"]

        if not concepts_dir.exists():
            return {
                "status": "success",
                "message": "No concepts directory exists yet.",
                "links_added_or_normalized": 0,
            }

        concept_files = list(concepts_dir.glob("concept_*.md"))
        if not concept_files:
            return {
                "status": "success",
                "message": "No concept files found.",
                "links_added_or_normalized": 0,
            }

        registry = self._load_concept_registry(project)
        concepts_data = {}

        for f in concept_files:
            cid = f.stem
            if cid not in registry:
                continue

            try:
                text = f.read_text(encoding="utf-8")
                cdata = registry[cid]

                links = set(re.findall(r"\[\[(concept_\d+)(?:\|[^\]]*)?\]\]", text))

                concepts_data[cid] = {
                    "path": f,
                    "name": cdata["canonical_name"].replace("-", " ").title(),
                    "text": text,
                    "links": links,
                }
            except Exception:
                pass

        connections = {cid: set(data["links"]) for cid, data in concepts_data.items()}

        for cid_a, data_a in concepts_data.items():
            text_lower = data_a["text"].lower()
            fm_match = re.match(
                r"^---\s*\n.*?\n---\s*\n(.*)$", data_a["text"], re.DOTALL
            )
            body_text = fm_match.group(1).lower() if fm_match else text_lower

            for cid_b, data_b in concepts_data.items():
                if cid_a == cid_b:
                    continue

                bdata = registry[cid_b]
                b_canonical = bdata["canonical_name"].lower()
                b_spaced = b_canonical.replace("-", " ")

                matched = False
                if (
                    re.search(rf"\b{re.escape(cid_b)}\b", body_text)
                    or re.search(rf"\b{re.escape(b_canonical)}\b", body_text)
                    or re.search(rf"\b{re.escape(b_spaced)}\b", body_text)
                ):
                    matched = True
                else:
                    for alias in bdata.get("aliases", []):
                        if re.search(rf"\b{re.escape(alias.lower())}\b", body_text):
                            matched = True
                            break

                if matched:
                    connections[cid_a].add(cid_b)

        reciprocal_connections = {cid: set(conn) for cid, conn in connections.items()}
        for cid_a, conns in connections.items():
            for cid_b in conns:
                if cid_b in reciprocal_connections:
                    reciprocal_connections[cid_b].add(cid_a)

        links_added_count = 0
        for cid, data in concepts_data.items():
            filepath = data["path"]
            current_text = data["text"]
            target_links = reciprocal_connections[cid]

            target_links = {l for l in target_links if l in concepts_data and l != cid}

            if not target_links:
                continue

            fm_match = re.match(
                r"^(---\s*\n.*?\n---\s*\n)(.*)$", current_text, re.DOTALL
            )
            if not fm_match:
                continue

            header = fm_match.group(1)
            body = fm_match.group(2)

            body_clean = re.split(
                r"\n##\s+(?:Related Knowledge|Related)\b", body, flags=re.IGNORECASE
            )[0].strip()

            related_lines = []
            for target_cid in sorted(target_links):
                target_name = (
                    registry[target_cid]["canonical_name"].replace("-", " ").title()
                )
                related_lines.append(
                    f"- [[{target_cid}|{target_name}]] — {target_name}"
                )

            related_section = "\n\n## Related Knowledge\n" + "\n".join(related_lines)
            new_content = header + body_clean + related_section + "\n"

            if new_content != current_text:
                try:
                    filepath.write_text(new_content, encoding="utf-8")
                    links_added_count += len(target_links) - len(data["links"])
                except Exception:
                    pass

        return {
            "status": "success",
            "message": f"Successfully updated bidirectional Graph links across {len(concept_files)} nodes using Concept IDs.",
            "links_added_or_normalized": links_added_count,
        }

    def session_commit(
        self, project: str, conversation_text: str = ""
    ) -> Dict[str, Any]:
        """Master Proactive Orchestrator. Performs reflection, materialization, bidirectional linking, and indexing."""
        reflection = self.reflect_session(project, conversation_text)
        if reflection["status"] == "error":
            return reflection

        materialize = self.materialize_concepts(project)
        if materialize["status"] == "error":
            return materialize

        graph = self.update_concept_graph(project)
        if graph["status"] == "error":
            return graph

        index_stats = self.index_all(force=True)

        return {
            "status": "success",
            "report_path": reflection["report_path"],
            "knowledge_events": reflection["knowledge_events"],
            "materialized_log": materialize["materialized"],
            "links_updated": graph["links_added_or_normalized"],
            "index_stats": index_stats,
        }
