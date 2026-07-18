"""One source of truth for secret paths — sandbox / permissions / watcher agree."""

from __future__ import annotations

import os

from clawagents.memory.hunk_watcher import is_secret_or_ignored_path as watcher_is_secret
from clawagents.sandbox.profiles import _default_secret_globs, _path_matches_secret_globs
from clawagents.security.secret_paths import (
    DEFAULT_SECRET_GLOBS,
    default_secure_path_rules,
    is_secret_basename,
    is_secret_or_ignored_path,
    path_matches_secret_globs,
)
from clawagents.tools.permissions import PermissionEngine, _DEFAULT_SECURE_RULES


def test_globs_shared_across_modules():
    assert _default_secret_globs() == DEFAULT_SECRET_GLOBS
    assert ".env" in DEFAULT_SECRET_GLOBS
    assert "**/*.pem" in DEFAULT_SECRET_GLOBS


def test_secret_basename_avoids_secretary_false_positive():
    assert is_secret_basename("secrets.json") is True
    assert is_secret_basename("my_credentials.yaml") is True
    assert is_secret_basename("secretary.txt") is False
    assert is_secret_basename("app.py") is False


def test_watcher_and_central_agree_on_secrets():
    for rel in (
        ".env",
        ".env.local",
        "creds/credentials.json",
        "nested/key.pem",
        "id_rsa",
        "foo.p12",
    ):
        assert is_secret_or_ignored_path(rel) is True
        assert watcher_is_secret(rel) is True
    assert is_secret_or_ignored_path("src/app.py") is False
    assert watcher_is_secret("src/app.py") is False
    assert watcher_is_secret("node_modules/x.js") is True


def test_sandbox_matcher_top_level_pem(tmp_path):
    cwd = str(tmp_path)
    globs = _default_secret_globs()
    pem = os.path.join(cwd, "key.pem")
    assert _path_matches_secret_globs(pem, cwd, globs)
    assert path_matches_secret_globs(pem, cwd, globs)
    assert not _path_matches_secret_globs(os.path.join(cwd, "app.py"), cwd, globs)


def test_permission_defaults_cover_keys_and_env():
    patterns = {r.path_pattern for r in _DEFAULT_SECURE_RULES if r.tool == "write_file"}
    assert "*.env" in patterns or "**/.env" in patterns
    assert any("pem" in p or "credentials" in p for p in patterns)
    # Generated from central rules
    central = {p for p, _, _ in default_secure_path_rules()}
    assert central.issubset(patterns) or patterns.issuperset(central)

    engine = PermissionEngine(default_decision="allow")
    engine.add_rules(list(_DEFAULT_SECURE_RULES))
    assert engine.evaluate("write_file", {"path": "config/credentials.yaml"})[0] == "deny"
    assert engine.evaluate("write_file", {"path": "nested/secrets.json"})[0] == "deny"
    assert engine.evaluate("hashline_edit", {"path": "key.pem"})[0] == "deny"
    assert engine.evaluate("write_file", {"path": ".env"})[0] == "ask"
    assert engine.evaluate("write_file", {"path": "src/ok.py"})[0] == "allow"
