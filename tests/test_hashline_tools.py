"""Hermetic tests for hashline read/edit/grep tools (Grok Build parity)."""

from __future__ import annotations

import asyncio
from pathlib import Path


from clawagents.config.features import temporary_overrides
from clawagents.sandbox.local import LocalBackend
from clawagents.tools.hashline import (
    create_hashline_tools,
    encode_hash,
    line_hash,
)


def test_line_hash_whitespace_normalized():
    h1 = line_hash("  def hello():  ")
    h2 = line_hash("def hello():")
    assert h1 == h2
    assert len(encode_hash(h1, 3)) == 3


def test_hashline_read_and_edit_roundtrip(tmp_path: Path):
    sb = LocalBackend(tmp_path)
    file = tmp_path / "app.py"
    file.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    with temporary_overrides({"hashline_tools": True}):
        tools = {t.name: t for t in create_hashline_tools(sb)}
        read_tool = tools["hashline_read"]
        edit_tool = tools["hashline_edit"]

        async def _run():
            # 1. Read file with hashline anchors (e.g. 2:lnr:zzs→    return a + b)
            read_res = await read_tool.execute({"path": "app.py"})
            assert read_res.success
            output = read_res.output
            assert "return a + b" in output
            assert "→" in output

            # Extract line 2 anchor
            lines = output.strip().split("\n")
            line2 = [l for l in lines if "return a + b" in l][0]
            anchor = line2.split("→")[0].strip()
            assert ":" in anchor

            # 2. Perform valid surgical edit
            edit_res = await edit_tool.execute({
                "path": "app.py",
                "edits": [
                    {
                        "op": "replace",
                        "anchor": anchor,
                        "content": "    return (a + b) * 2",
                    }
                ],
            })
            assert edit_res.success
            assert "return (a + b) * 2" in file.read_text(encoding="utf-8")

            # 3. Reject edit with mismatched/stale anchor and return fresh anchors
            stale_res = await edit_tool.execute({
                "path": "app.py",
                "edits": [
                    {
                        "op": "replace",
                        "anchor": "2:zzz:zzz",  # wrong hash
                        "content": "    return 0",
                    }
                ],
            })
            assert not stale_res.success
            assert "stale" in (stale_res.error or "").lower()

        asyncio.run(_run())
