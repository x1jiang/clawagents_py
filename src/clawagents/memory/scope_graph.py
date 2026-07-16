"""Incremental scope graph for repo map (mtime-indexed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from clawagents.memory.repo_map import (
    RepoTag,
    _iter_code_files,
    _rel,
    collect_tags,
    rank_tags,
    render_repo_map,
)


@dataclass
class _FileRecord:
    mtime_ns: int
    tags: list[RepoTag] = field(default_factory=list)
    import_names: set[str] = field(default_factory=set)


@dataclass
class ScopeGraph:
    """Cached file→symbol graph with incremental refresh."""

    root: Path
    _files: dict[str, _FileRecord] = field(default_factory=dict)
    _name_to_files: dict[str, set[str]] = field(default_factory=dict)

    def refresh(
        self,
        *,
        changed: Iterable[Path] | None = None,
        max_files: int = 2_000,
    ) -> None:
        root = self.root
        if changed is None:
            paths = _iter_code_files(root, max_files=max_files)
        else:
            paths = [Path(p) for p in changed]

        for path in paths:
            if not path.is_file():
                rel = _rel(root, path)
                self._drop_file(rel)
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            rel = _rel(root, path)
            prev = self._files.get(rel)
            if prev is not None and prev.mtime_ns == st.st_mtime_ns:
                continue
            # Reuse collect_tags on a single-file tree via temp filter:
            tags, edges = collect_tags_for_files(root, [path])
            self._drop_file(rel)
            file_tags = [t for t in tags if t.path == rel]
            import_names: set[str] = set()
            # edges keys are files; values are related files — recover names from tags
            for t in file_tags:
                self._name_to_files.setdefault(t.name, set()).add(rel)
            # Store import fragments from a light re-parse
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            from clawagents.memory.repo_map import _IMPORT_RE

            for im in _IMPORT_RE.finditer(text):
                frag = next((g for g in im.groups() if g), "")
                if frag:
                    import_names.add(frag.replace("/", ".").split(".")[-1])
            self._files[rel] = _FileRecord(
                mtime_ns=st.st_mtime_ns,
                tags=file_tags,
                import_names=import_names,
            )

        # Full scan also drops deleted files
        if changed is None:
            live = {_rel(root, p) for p in _iter_code_files(root, max_files=max_files)}
            for rel in list(self._files):
                if rel not in live:
                    self._drop_file(rel)

    def _drop_file(self, rel: str) -> None:
        rec = self._files.pop(rel, None)
        if rec is None:
            return
        for name, files in list(self._name_to_files.items()):
            if rel in files:
                files.discard(rel)
                if not files:
                    del self._name_to_files[name]

    def invalidate(self, path: str | Path) -> None:
        rel = _rel(self.root, Path(path))
        self._drop_file(rel)

    def all_tags(self) -> list[RepoTag]:
        out: list[RepoTag] = []
        for rec in self._files.values():
            out.extend(rec.tags)
        return out

    def file_edges(self) -> dict[str, set[str]]:
        edges: dict[str, set[str]] = {}
        for src, rec in self._files.items():
            for name in rec.import_names:
                for dest in self._name_to_files.get(name, ()):
                    if dest != src:
                        edges.setdefault(src, set()).add(dest)
                        edges.setdefault(dest, set()).add(src)
        return edges

    def query(
        self,
        mentioned: set[str] | None = None,
        *,
        max_chars: int = 4_000,
        chat_files: set[str] | None = None,
    ) -> str:
        if not self._files:
            self.refresh()
        ranked = rank_tags(
            self.all_tags(),
            self.file_edges(),
            mentioned=mentioned,
            chat_files=chat_files,
        )
        return render_repo_map(ranked, max_chars=max_chars)


# Module-level cache keyed by resolved root
_GRAPHS: dict[str, ScopeGraph] = {}


def get_scope_graph(workspace: str | Path | None = None) -> ScopeGraph:
    root = Path(workspace or Path.cwd()).resolve()
    key = str(root)
    g = _GRAPHS.get(key)
    if g is None:
        g = ScopeGraph(root=root)
        _GRAPHS[key] = g
    return g


def collect_tags_for_files(root: Path, files: list[Path]):
    """Run collect_tags logic limited to explicit files (internal helper)."""
    # Temporarily monkey by filtering — simplest: call collect_tags on root
    # is too heavy; inline a mini version:
    from collections import defaultdict
    from clawagents.memory.repo_map import _TAG_RE, _IMPORT_RE, RepoTag as RT

    tags: list[RT] = []
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
                tags.append(RT(path=rel, name=name, kind=kind, line=i))
                name_to_files[name].add(rel)
        for im in _IMPORT_RE.finditer(text):
            frag = next((g for g in im.groups() if g), "")
            if frag:
                edges[rel].add(frag.replace("/", ".").split(".")[-1])
    file_edges: dict[str, set[str]] = defaultdict(set)
    for src, names in edges.items():
        for n in names:
            for dest in name_to_files.get(n, ()):
                if dest != src:
                    file_edges[src].add(dest)
                    file_edges[dest].add(src)
    return tags, file_edges


def build_repo_map_incremental(
    workspace: str | Path | None = None,
    *,
    max_chars: int = 4_000,
    mentioned: set[str] | None = None,
    chat_files: set[str] | None = None,
    changed: Iterable[Path] | None = None,
) -> str:
    from clawagents.config.features import is_enabled

    if not is_enabled("incremental_repo_map"):
        from clawagents.memory.repo_map import build_repo_map

        return build_repo_map(
            workspace, max_chars=max_chars, mentioned=mentioned, chat_files=chat_files
        )
    g = get_scope_graph(workspace)
    g.refresh(changed=changed)
    return g.query(mentioned, max_chars=max_chars, chat_files=chat_files)


__all__ = [
    "ScopeGraph",
    "get_scope_graph",
    "build_repo_map_incremental",
]
