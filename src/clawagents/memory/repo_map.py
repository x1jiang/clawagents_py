"""Aider-inspired repo map: symbol tags ranked into a token budget.

Uses a lightweight import/mention graph (no NetworkX required). Falls back to
frequency + path boosts when the graph is sparse.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "dist", "build", "__pycache__",
    ".next", ".cache", "coverage", ".clawagents", "target", "vendor",
}
_CODE_EXT = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt",
    ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".cs", ".swift", ".scala",
    ".md",
}
_TAG_RE = re.compile(
    r"(?m)^(?P<indent>\s*)(?P<kind>def |async def |class |export (?:default )?(?:async )?function |"
    r"fn |pub (?:async )?fn |func |interface |struct |type |public class |public interface )"
    r"(?P<name>[\w.]+)"
)
_IMPORT_RE = re.compile(
    r"(?m)^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+)|"
    r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]|"
    r"require\(['\"]([^'\"]+)['\"]\))"
)


@dataclass(frozen=True)
class RepoTag:
    path: str
    name: str
    kind: str
    line: int
    score: float = 0.0


def _iter_code_files(root: Path, *, max_files: int = 2_000) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in _CODE_EXT:
            continue
        try:
            if p.stat().st_size > 400_000:
                continue
        except OSError:
            continue
        out.append(p)
        if len(out) >= max_files:
            break
    return out


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def collect_tags(root: Path) -> tuple[list[RepoTag], dict[str, set[str]]]:
    """Return tags and a file→imported-path-fragment graph."""
    files = _iter_code_files(root)
    tags: list[RepoTag] = []
    edges: dict[str, set[str]] = defaultdict(set)
    name_to_files: dict[str, set[str]] = defaultdict(set)

    for path in files:
        rel = _rel(root, path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            m = _TAG_RE.match(line)
            if m:
                name = m.group("name")
                kind = m.group("kind").strip()
                tags.append(RepoTag(path=rel, name=name, kind=kind, line=i))
                name_to_files[name].add(rel)
        for im in _IMPORT_RE.finditer(text):
            frag = next((g for g in im.groups() if g), "")
            if not frag:
                continue
            # crude: last path segment
            key = frag.replace("/", ".").split(".")[-1]
            edges[rel].add(key)

    # Convert import names to file edges via defined symbols
    file_edges: dict[str, set[str]] = defaultdict(set)
    for src, names in edges.items():
        for n in names:
            for dest in name_to_files.get(n, ()):
                if dest != src:
                    file_edges[src].add(dest)
                    file_edges[dest].add(src)
    return tags, file_edges


def rank_tags(
    tags: list[RepoTag],
    file_edges: dict[str, set[str]],
    *,
    mentioned: set[str] | None = None,
    chat_files: set[str] | None = None,
) -> list[RepoTag]:
    mentioned = {m.lower() for m in (mentioned or set())}
    chat_files = chat_files or set()
    # Degree centrality proxy for PageRank
    degree = {f: len(neis) for f, neis in file_edges.items()}
    scored: list[RepoTag] = []
    for t in tags:
        score = 1.0 + float(degree.get(t.path, 0))
        if t.name.lower() in mentioned:
            score *= 8.0
        if any(part.lower() in mentioned for part in Path(t.path).parts):
            score *= 2.0
        if t.path in chat_files:
            score *= 3.0
        if t.kind.startswith(("class", "interface", "struct", "type")):
            score *= 1.4
        scored.append(RepoTag(path=t.path, name=t.name, kind=t.kind, line=t.line, score=score))
    scored.sort(key=lambda x: (-x.score, x.path, x.line))
    return scored


def render_repo_map(
    tags: list[RepoTag],
    *,
    max_chars: int = 4_000,
) -> str:
    if not tags:
        return ""
    by_file: dict[str, list[RepoTag]] = defaultdict(list)
    for t in tags:
        by_file[t.path].append(t)

    lines = ["## Repo Map", "Key symbols (ranked). Use read_file(tier=L0/L1) for bodies.", ""]
    used = 0
    for path, items in by_file.items():
        block = [f"{path}:"]
        for t in items[:40]:
            block.append(f"  {t.kind}{t.name}  # L{t.line}")
        chunk = "\n".join(block) + "\n"
        if used + len(chunk) > max_chars:
            break
        lines.append(chunk.rstrip())
        used += len(chunk)
    return "\n".join(lines).strip() + "\n"


def build_repo_map(
    workspace: str | Path | None = None,
    *,
    max_chars: int = 4_000,
    mentioned: set[str] | None = None,
    chat_files: set[str] | None = None,
) -> str:
    root = Path(workspace or Path.cwd())
    tags, edges = collect_tags(root)
    ranked = rank_tags(tags, edges, mentioned=mentioned, chat_files=chat_files)
    return render_repo_map(ranked, max_chars=max_chars)
