"""P1 security regressions closed in v6.20.5.

1. Permission engine: newer filesystem writers (hashline_edit, create_file,
   delete_file, …) bypassed the default secret-path deny/ask rules because they
   were never aliased to write_file/edit_file — they fell through to ``allow``.
2. Seatbelt profile: the secret ``file-write*`` denies were emitted *before* the
   workspace write-allow, so SBPL last-match-wins let the trailing allow
   override them; the glob→literal reduction also produced names (``.pem``,
   ``credentials``) that matched no real file, so secret *reads* leaked too.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest


# ── 1. Permission engine covers the whole filesystem-writer family ──────────


WRITER_TOOLS = [
    "write_file",
    "edit_file",
    "apply_patch",
    "hashline_edit",
    "create_file",
    "replace_in_file",
    "insert_in_file",
    "insert_lines",
    "patch_file",
    "delete_file",
]


@pytest.mark.parametrize("tool", WRITER_TOOLS)
def test_all_writers_denied_on_credentials(tool):
    from clawagents.tools.permissions import PermissionEngine, _DEFAULT_SECURE_RULES

    engine = PermissionEngine(default_decision="allow")
    engine.add_rules(list(_DEFAULT_SECURE_RULES))
    decision, _ = engine.evaluate(tool, {"path": "config/credentials.yaml"})
    assert decision == "deny", f"{tool} must be denied on a credentials file"


@pytest.mark.parametrize("tool", WRITER_TOOLS)
def test_all_writers_ask_on_env(tool):
    from clawagents.tools.permissions import PermissionEngine, _DEFAULT_SECURE_RULES

    engine = PermissionEngine(default_decision="allow")
    engine.add_rules(list(_DEFAULT_SECURE_RULES))
    decision, _ = engine.evaluate(tool, {"path": "app/.env"})
    assert decision == "ask", f"{tool} must require approval to write .env"


def test_writers_still_allow_normal_source():
    from clawagents.tools.permissions import PermissionEngine, _DEFAULT_SECURE_RULES

    engine = PermissionEngine(default_decision="allow")
    engine.add_rules(list(_DEFAULT_SECURE_RULES))
    for tool in WRITER_TOOLS:
        decision, _ = engine.evaluate(tool, {"path": "src/module.py"})
        assert decision == "allow", f"{tool} on ordinary source must stay allowed"


def test_fs_write_tools_mirror_write_class_set():
    """The local writer set must stay in sync with the canonical WRITE_CLASS set."""
    from clawagents.permissions.mode import WRITE_CLASS_TOOLS
    from clawagents.tools.permissions import _FS_WRITE_TOOLS

    # Every _FS_WRITE_TOOLS entry is a genuine write-class tool.
    assert _FS_WRITE_TOOLS <= set(WRITE_CLASS_TOOLS)
    # The filesystem writers in WRITE_CLASS_TOOLS are all covered (exec/git/
    # subagent members are intentionally excluded and gated elsewhere).
    fs_like = {
        t
        for t in WRITE_CLASS_TOOLS
        if t
        in {
            "write_file",
            "edit_file",
            "apply_patch",
            "hashline_edit",
            "create_file",
            "replace_in_file",
            "insert_in_file",
            "insert_lines",
            "patch_file",
            "delete_file",
        }
    }
    assert fs_like == _FS_WRITE_TOOLS


# ── 2. Seatbelt secret denial (text-level + live sandbox-exec) ──────────────


def _gen_profile(cwd):
    from clawagents.sandbox.profiles import _default_secret_globs, _seatbelt_profile_text

    return _seatbelt_profile_text(
        cwd=cwd,
        network=False,
        read_only=False,
        secret_deny_paths=_default_secret_globs(),
    )


def test_seatbelt_secret_denies_come_after_workspace_allow(tmp_path):
    """Ordering is the whole bug: deny rules must be last (last-match-wins)."""
    prof = _gen_profile(str(tmp_path))
    lines = prof.splitlines()
    first_allow = next(
        i for i, ln in enumerate(lines) if ln.startswith("(allow file-write* (subpath")
    )
    first_secret_deny = next(
        i for i, ln in enumerate(lines) if "(deny file-write* (regex" in ln
    )
    assert first_secret_deny > first_allow, (
        "secret write-denies must follow the workspace write-allow, else the "
        "trailing allow overrides them"
    )


def test_seatbelt_regex_not_double_escaped(tmp_path):
    prof = _gen_profile(str(tmp_path))
    # A single-backslash \. is correct; \\ . would match nothing.
    assert "\\\\." not in prof
    assert "(regex #\"" in prof


@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None, reason="macOS sandbox-exec not available"
)
def test_seatbelt_live_denies_secrets_allows_normal():
    sbx = shutil.which("sandbox-exec")
    work = tempfile.mkdtemp(prefix="clawsb_")
    try:
        files = {
            ".env": "SECRET=1",
            ".env.local": "S=2",
            "credentials.json": "{}",
            "config/credentials.yaml": "x",
            "secrets.json": "{}",
            "key.pem": "KEY",
            "sub/key.pem": "KEY2",
            "normal.py": "print(1)",
        }
        for rel, content in files.items():
            p = os.path.join(work, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write(content)
        sb = os.path.join(work, "prof.sb")
        with open(sb, "w") as fh:
            fh.write(_gen_profile(work))

        def run(cmd):
            return subprocess.run(
                [sbx, "-f", sb, "/bin/sh", "-c", cmd],
                capture_output=True,
                text=True,
            ).returncode

        # Secret reads must be denied (nonzero exit).
        for rel in [
            ".env",
            ".env.local",
            "credentials.json",
            "config/credentials.yaml",
            "secrets.json",
            "key.pem",
            "sub/key.pem",
        ]:
            rc = run(f'cat "{os.path.join(work, rel)}"')
            assert rc != 0, f"seatbelt leaked secret read: {rel}"

        # Secret writes must be denied.
        for rel in [".env", "key.pem", "credentials.json"]:
            rc = run(f'echo x >> "{os.path.join(work, rel)}"')
            assert rc != 0, f"seatbelt allowed secret write: {rel}"

        # Normal reads + writes must still work.
        assert run(f'cat "{os.path.join(work, "normal.py")}"') == 0
        assert run(f'echo x >> "{os.path.join(work, "normal.py")}"') == 0
        assert run(f'echo x > "{os.path.join(work, "brand_new.txt")}"') == 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_default_secret_globs_cover_private_keys():
    from clawagents.sandbox.profiles import _default_secret_globs, _path_matches_secret_globs

    globs = _default_secret_globs()
    cwd = "/work"
    for rel in ["key.pem", "server.key", "cert.p12", "bundle.pfx", "id_rsa", "id_ed25519"]:
        assert _path_matches_secret_globs(
            os.path.join(cwd, rel), cwd, globs
        ), f"{rel} should be treated as secret"
    # Non-secrets stay readable.
    assert not _path_matches_secret_globs(os.path.join(cwd, "app.py"), cwd, globs)
