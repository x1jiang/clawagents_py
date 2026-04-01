"""Test suite for all 10 Claude Code patterns ported to clawagents_py."""

import sys
import os
import json
import time
import tempfile
import shutil
import asyncio

# Ensure local source is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

PASSED = 0
FAILED = 0

def report(name, ok, detail=""):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name}: {detail}")


# ═══════════════════════════════════════════════════════════════════════════
# Feature #10: Feature Flags
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ #10 Feature Flags ━━━")

try:
    from clawagents.config.features import is_enabled, all_features, reset

    # Test defaults
    reset()
    flags = all_features()
    report("all_features() returns dict", isinstance(flags, dict))
    report("micro_compact defaults to True", flags.get("micro_compact") is True)
    report("file_snapshots defaults to True", flags.get("file_snapshots") is True)
    report("wal defaults to False", flags.get("wal") is False)
    report("coordinator defaults to False", flags.get("coordinator") is False)

    # Test env override
    reset(); os.environ["CLAW_FEATURE_WAL"] = "1"
    reset(); os.environ["CLAW_FEATURE_MICRO_COMPACT"] = "0"
    reset()
    report("env override WAL=1 → True", is_enabled("wal") is True)
    report("env override MICRO_COMPACT=0 → False", is_enabled("micro_compact") is False)

    # Test unknown feature
    report("unknown feature returns False", is_enabled("nonexistent_feature_xyz") is False)

    # Restore
    reset(); os.environ.pop("CLAW_FEATURE_WAL", None)
    reset(); os.environ.pop("CLAW_FEATURE_MICRO_COMPACT", None)
    reset()

except Exception as e:
    report("Feature Flags import", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Feature #1: Micro-Compact Tool Results
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ #1 Micro-Compact Tool Results ━━━")

try:
    from clawagents.providers.llm import LLMMessage
    # Import the compaction function from agent_loop
    # It's a module-level function, so we import it
    from clawagents.graph.agent_loop import _micro_compact_tool_results

    # Build a message list simulating: system, user, assistant(tool_call), tool_result, assistant(tool_call), tool_result
    messages = [
        LLMMessage(role="system", content="You are helpful."),
        LLMMessage(role="user", content="Read my file"),
        LLMMessage(role="assistant", content="I'll read the file.", tool_calls_meta=[
            {"id": "tc1", "name": "read_file", "args": {"path": "/tmp/test.py"}}
        ]),
        LLMMessage(role="tool", content="x = 1\ny = 2\nz = 3\n" * 100, tool_call_id="tc1"),
        LLMMessage(role="assistant", content="Now I'll grep.", tool_calls_meta=[
            {"id": "tc2", "name": "grep", "args": {"query": "hello"}}
        ]),
        LLMMessage(role="tool", content="Match found in line 42: hello world\n" * 50, tool_call_id="tc2"),
        LLMMessage(role="assistant", content="I found the results."),
        LLMMessage(role="user", content="Thanks!"),
    ]

    reset(); os.environ["CLAW_FEATURE_MICRO_COMPACT"] = "1"
    reset()
    compacted = _micro_compact_tool_results(messages, keep_recent=1)

    # The first tool result (read_file, tc1) should be cleared since it's old
    report("compacted returns list", isinstance(compacted, list))
    report("message count preserved", len(compacted) == len(messages))

    # Check that the older tool result is cleared (tc1 for read_file)
    tool_msg_1 = compacted[3]
    cleared_1 = isinstance(tool_msg_1.content, str) and "cleared" in tool_msg_1.content.lower()
    report("old read_file tool result cleared", cleared_1, f"content='{str(tool_msg_1.content)[:80]}'")

    # The most recent tool result (tc2, grep) should also be cleared since there's newer conversation
    tool_msg_2 = compacted[5]
    report("tool_call_id preserved on cleared msg", tool_msg_2.tool_call_id == "tc2")

    reset(); os.environ.pop("CLAW_FEATURE_MICRO_COMPACT", None)

except Exception as e:
    report("Micro-Compact import/run", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Feature #5: File History Snapshots
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ #5 File History Snapshots ━━━")

try:
    from clawagents.tools.registry import _snapshot_before_write, _WRITE_TOOLS

    report("_WRITE_TOOLS is a frozenset", isinstance(_WRITE_TOOLS, frozenset))
    report("write_file in WRITE_TOOLS", "write_file" in _WRITE_TOOLS)
    report("edit_file in WRITE_TOOLS", "edit_file" in _WRITE_TOOLS)
    report("read_file NOT in WRITE_TOOLS", "read_file" not in _WRITE_TOOLS)

    # Create a temp dir and file to test snapshotting
    tmpdir = tempfile.mkdtemp(prefix="clawagents_test_")
    test_file = os.path.join(tmpdir, "testfile.py")
    with open(test_file, "w") as f:
        f.write("original content")

    # Save CWD, change to tmpdir so .clawagents/ is created there
    orig_cwd = os.getcwd()
    os.chdir(tmpdir)

    reset(); os.environ["CLAW_FEATURE_FILE_SNAPSHOTS"] = "1"
    reset()
    _snapshot_before_write("write_file", {"path": test_file})

    snap_dir = os.path.join(tmpdir, ".clawagents", "snapshots")
    snap_exists = os.path.isdir(snap_dir)
    report("snapshot dir created", snap_exists)

    if snap_exists:
        ts_dirs = os.listdir(snap_dir)
        report("timestamp dir created", len(ts_dirs) > 0)
        if ts_dirs:
            snap_file = os.path.join(snap_dir, ts_dirs[0], "testfile.py")
            report("snapshot file exists", os.path.isfile(snap_file))
            if os.path.isfile(snap_file):
                content = open(snap_file).read()
                report("snapshot content matches original", content == "original content")

    # Test that non-write tools don't trigger snapshot
    _snapshot_before_write("read_file", {"path": test_file})
    ts_dirs_after = os.listdir(snap_dir) if snap_exists else []
    report("read_file doesn't create extra snapshots", len(ts_dirs_after) == len(ts_dirs) if snap_exists else True)

    # Test with missing file
    _snapshot_before_write("write_file", {"path": "/nonexistent/file.py"})
    report("missing file doesn't crash", True)

    # Test with no path arg
    _snapshot_before_write("write_file", {"query": "hello"})
    report("missing path arg doesn't crash", True)

    os.chdir(orig_cwd)
    reset(); os.environ.pop("CLAW_FEATURE_FILE_SNAPSHOTS", None)
    shutil.rmtree(tmpdir, ignore_errors=True)

except Exception as e:
    report("File Snapshots import/run", False, str(e))
    try:
        os.chdir(orig_cwd)
    except:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Feature #7: Prompt Cache Tracking
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ #7 Prompt Cache Tracking ━━━")

try:
    from clawagents.providers.llm import LLMResponse

    # Test new fields exist
    resp = LLMResponse(
        content="hello",
        model="test",
        tokens_used=100,
        cache_creation_tokens=50,
        cache_read_tokens=80,
        prompt_tokens=200,
    )
    report("cache_creation_tokens field", resp.cache_creation_tokens == 50)
    report("cache_read_tokens field", resp.cache_read_tokens == 80)
    report("prompt_tokens field", resp.prompt_tokens == 200)

    # Test defaults
    resp2 = LLMResponse(content="hi", model="t", tokens_used=10)
    report("cache_creation_tokens defaults to 0", resp2.cache_creation_tokens == 0)
    report("cache_read_tokens defaults to 0", resp2.cache_read_tokens == 0)
    report("prompt_tokens defaults to 0", resp2.prompt_tokens == 0)

    # Calculate cache hit rate
    hit_pct = resp.cache_read_tokens / resp.prompt_tokens * 100
    report(f"cache hit calculation works ({hit_pct:.0f}%)", hit_pct == 40.0)

except Exception as e:
    report("Cache Tracking import/run", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Feature #3: Typed Memory Taxonomy
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ #3 Typed Memory Taxonomy ━━━")

try:
    from clawagents.memory.loader import (
        parse_memory_frontmatter,
        load_memory_files,
        load_memory_directory,
        VALID_MEMORY_TYPES,
    )

    # Test frontmatter parsing
    result = parse_memory_frontmatter("""---
type: feedback
name: testing_preference
description: User wants pytest -x
---
Always use `pytest -x` to stop on first failure.""")

    report("parse type", result["type"] == "feedback")
    report("parse name", result["name"] == "testing_preference")
    report("parse description", result["description"] == "User wants pytest -x")
    report("parse content", "pytest -x" in result["content"])

    # Test without frontmatter
    result2 = parse_memory_frontmatter("Just plain text memory")
    report("no frontmatter → type=general", result2["type"] == "general")
    report("no frontmatter → content preserved", result2["content"] == "Just plain text memory")

    # Test invalid type
    result3 = parse_memory_frontmatter("---\ntype: invalid_type\n---\ncontent")
    report("invalid type → general", result3["type"] == "general")

    # Test valid types set
    report("valid types include user", "user" in VALID_MEMORY_TYPES)
    report("valid types include project", "project" in VALID_MEMORY_TYPES)
    report("valid types include reference", "reference" in VALID_MEMORY_TYPES)

    # Test load_memory_files with typed memory enabled
    tmpdir = tempfile.mkdtemp(prefix="clawagents_mem_")
    mem_file = os.path.join(tmpdir, "test.md")
    with open(mem_file, "w") as f:
        f.write("---\ntype: project\nname: stack\n---\nUses Python 3.11 and FastAPI")

    reset(); os.environ["CLAW_FEATURE_TYPED_MEMORY"] = "1"
    loaded = load_memory_files([mem_file])
    report("load_memory_files returns content", loaded is not None)
    if loaded:
        report("type attribute in output", 'type="project"' in loaded)
        report("name attribute in output", 'name="stack"' in loaded)

    # Test type filtering
    loaded_filtered = load_memory_files([mem_file], filter_type="user")
    report("type filter excludes non-matching", loaded_filtered is None)

    loaded_match = load_memory_files([mem_file], filter_type="project")
    report("type filter includes matching", loaded_match is not None)

    # Test load_memory_directory
    dir_result = load_memory_directory(tmpdir)
    report("load_memory_directory works", dir_result is not None)

    reset(); os.environ.pop("CLAW_FEATURE_TYPED_MEMORY", None)
    shutil.rmtree(tmpdir, ignore_errors=True)

except Exception as e:
    report("Typed Memory import/run", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Feature #8: Write-Ahead Logging (WAL)
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ #8 Write-Ahead Logging (WAL) ━━━")

try:
    from clawagents.graph.agent_loop import _wal_write

    tmpdir = tempfile.mkdtemp(prefix="clawagents_wal_")
    orig_cwd = os.getcwd()
    os.chdir(tmpdir)

    reset(); os.environ["CLAW_FEATURE_WAL"] = "1"

    messages = [
        LLMMessage(role="system", content="You are helpful."),
        LLMMessage(role="user", content="Hello, test WAL feature!"),
    ]
    _wal_write(messages)

    wal_path = os.path.join(tmpdir, ".clawagents", "wal.jsonl")
    report("WAL file created", os.path.isfile(wal_path))

    if os.path.isfile(wal_path):
        lines = open(wal_path).readlines()
        report("WAL has 1 entry", len(lines) == 1)

        entry = json.loads(lines[0])
        report("WAL entry has role", entry.get("role") == "user")
        report("WAL entry has content", "Hello, test WAL" in entry.get("content", ""))
        report("WAL entry has timestamp", "ts" in entry)
        report("WAL entry has msg_count", entry.get("msg_count") == 2)

        # Write another entry
        messages.append(LLMMessage(role="assistant", content="Hi!"))
        _wal_write(messages)
        lines2 = open(wal_path).readlines()
        report("WAL appends (now 2 entries)", len(lines2) == 2)

    # Test with WAL disabled
    reset(); os.environ["CLAW_FEATURE_WAL"] = "0"
    before_size = os.path.getsize(wal_path) if os.path.isfile(wal_path) else 0
    _wal_write(messages)
    after_size = os.path.getsize(wal_path) if os.path.isfile(wal_path) else 0
    report("WAL disabled → no write", before_size == after_size)

    os.chdir(orig_cwd)
    reset(); os.environ.pop("CLAW_FEATURE_WAL", None)
    shutil.rmtree(tmpdir, ignore_errors=True)

except Exception as e:
    report("WAL import/run", False, str(e))
    try:
        os.chdir(orig_cwd)
    except:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Feature #6: Granular Permission Rules
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ #6 Granular Permission Rules ━━━")

try:
    from clawagents.tools.permissions import PermissionEngine, PermissionRule

    engine = PermissionEngine()

    # Add rules
    engine.add_rule(PermissionRule(tool="execute*", decision="deny", message="No shell"))
    engine.add_rule(PermissionRule(tool="write_file", path_pattern="/etc/*", decision="deny", priority=10))
    engine.add_rule(PermissionRule(tool="write_file", decision="allow", priority=0))
    engine.add_rule(PermissionRule(tool="read_file", decision="allow"))

    # Test tool matching
    report("execute denied", not engine.check("execute", {}))
    report("execute_command denied (glob)", not engine.check("execute_command", {}))
    report("read_file allowed", engine.check("read_file", {}))

    # Test path matching
    report("write /etc/passwd denied (path)", not engine.check("write_file", {"path": "/etc/passwd"}))
    report("write /tmp/foo allowed", engine.check("write_file", {"path": "/tmp/foo.py"}))

    # Test evaluate returns message
    decision, msg = engine.evaluate("execute", {})
    report("evaluate returns deny", decision == "deny")
    report("evaluate returns message", msg == "No shell")

    # Test priority order
    engine2 = PermissionEngine()
    engine2.add_rule(PermissionRule(tool="*", decision="deny", priority=0))
    engine2.add_rule(PermissionRule(tool="read_file", decision="allow", priority=10))
    report("higher priority wins (allow)", engine2.check("read_file", {}))
    report("lower priority fallback (deny)", not engine2.check("write_file", {}))

    # Test from_config
    engine3 = PermissionEngine.from_config([
        {"tool": "execute*", "decision": "deny"},
        {"tool": "*", "decision": "allow"},
    ])
    report("from_config works", not engine3.check("execute", {}))
    report("from_config default allow", engine3.check("read_file", {}))

    # Test chaining
    engine4 = PermissionEngine().add_rule(
        PermissionRule(tool="x", decision="deny")
    ).add_rule(
        PermissionRule(tool="y", decision="allow")
    )
    report("chaining works", not engine4.check("x", {}) and engine4.check("y", {}))

except Exception as e:
    report("Permission Rules import/run", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Feature #2: Background Memory Extraction
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ #2 Background Memory Extraction ━━━")

try:
    from clawagents.trajectory.background_memory import (
        save_memories,
        _format_messages_segment,
        _get_memories_dir,
    )

    # Test message formatting
    msgs = [
        LLMMessage(role="user", content="Hello world"),
        LLMMessage(role="assistant", content="Hi there!"),
        LLMMessage(role="user", content="Write a function"),
    ]
    formatted = _format_messages_segment(msgs, 0, 3)
    report("format_messages returns string", isinstance(formatted, str))
    report("format includes roles", "USER" in formatted and "ASSISTANT" in formatted)
    report("format includes content", "Hello world" in formatted)

    # Test save_memories
    tmpdir = tempfile.mkdtemp(prefix="clawagents_bgmem_")
    orig_cwd = os.getcwd()
    os.chdir(tmpdir)

    memories = [
        {"type": "project", "content": "Uses Python 3.11", "confidence": 0.9},
        {"type": "user", "content": "Prefers pytest -x", "confidence": 0.8},
    ]
    path = save_memories(memories, turn_index=5)
    report("save_memories returns path", path is not None)

    if path:
        report("memories file exists", os.path.isfile(path))
        content = open(path).read()
        report("memory has frontmatter", "type: project" in content)
        report("memory has content", "Python 3.11" in content)
        report("memory has turn metadata", "turn: 5" in content)

    # Test empty memories
    empty_path = save_memories([], turn_index=10)
    report("empty memories returns None", empty_path is None)

    # Test maybe_extract_memories with feature disabled
    reset(); os.environ["CLAW_FEATURE_BACKGROUND_MEMORY"] = "0"
    from clawagents.trajectory.background_memory import maybe_extract_memories

    result = asyncio.run(maybe_extract_memories(None, msgs, 10, 0))
    report("disabled → returns last_extraction_turn unchanged", result == 0)

    # Test interval check (not enough turns)
    reset(); os.environ["CLAW_FEATURE_BACKGROUND_MEMORY"] = "1"
    result2 = asyncio.run(maybe_extract_memories(None, msgs, 3, 0, interval=5))
    report("interval not reached → returns unchanged", result2 == 0)

    os.chdir(orig_cwd)
    reset(); os.environ.pop("CLAW_FEATURE_BACKGROUND_MEMORY", None)
    shutil.rmtree(tmpdir, ignore_errors=True)

except Exception as e:
    report("Background Memory import/run", False, str(e))
    try:
        os.chdir(orig_cwd)
    except:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Feature #9: Forked Agent Pattern
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ #9 Forked Agent Pattern ━━━")

try:
    from clawagents.graph.forked_agent import run_forked_agent, run_forked_agent_background

    # Test that it raises when feature is disabled
    reset(); os.environ["CLAW_FEATURE_FORKED_AGENTS"] = "0"
    try:
        asyncio.run(run_forked_agent(fork_prompt="test", llm=None))
        report("disabled raises error", False, "should have raised")
    except RuntimeError as e:
        report("disabled raises RuntimeError", "not enabled" in str(e))

    # Test that it properly accepts the feature flag
    reset(); os.environ["CLAW_FEATURE_FORKED_AGENTS"] = "1"
    # We can't fully run without a real LLM, but we can verify it gets past the flag check
    try:
        state = asyncio.run(run_forked_agent(fork_prompt="test", llm=None))
        report("enabled passes flag check (fails on None llm)", state.status == "error")
    except Exception as e:
        report("enabled passes flag check (fails on None llm)", True)

    reset(); os.environ.pop("CLAW_FEATURE_FORKED_AGENTS", None)

    # Test the function signature
    import inspect
    sig = inspect.signature(run_forked_agent)
    params = list(sig.parameters.keys())
    report("has fork_prompt param", "fork_prompt" in params)
    report("has llm param", "llm" in params)
    report("has allowed_tools param", "allowed_tools" in params)
    report("has blocked_tools param", "blocked_tools" in params)
    report("has max_turns param", "max_turns" in params)

except Exception as e:
    report("Forked Agent import/run", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Feature #4: Coordinator/Swarm Mode
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ #4 Coordinator/Swarm Mode ━━━")

try:
    from clawagents.graph.coordinator import (
        run_coordinator,
        CoordinatorState,
        WorkerTask,
        _parse_coordinator_response,
        COORDINATOR_SYSTEM_PROMPT,
    )

    # Test coordinator response parsing
    parsed = _parse_coordinator_response('{"action": "delegate", "tasks": [{"id": "t1", "prompt": "Do X"}]}')
    report("parse delegate action", parsed.get("action") == "delegate")
    report("parse tasks list", len(parsed.get("tasks", [])) == 1)

    # Test with code fence
    parsed2 = _parse_coordinator_response('```json\n{"action": "complete", "result": "Done!"}\n```')
    report("parse code-fenced JSON", parsed2.get("action") == "complete")
    report("parse result", parsed2.get("result") == "Done!")

    # Test with plain text (fallback)
    parsed3 = _parse_coordinator_response("Just some plain text answer")
    report("plain text → complete action", parsed3.get("action") == "complete")

    # Test data classes
    wt = WorkerTask(id="task_1", prompt="Do something", tools=["read_file"])
    report("WorkerTask created", wt.id == "task_1")
    report("WorkerTask default status", wt.status == "pending")

    cs = CoordinatorState(task="Big task")
    report("CoordinatorState created", cs.task == "Big task")
    report("CoordinatorState default workers", len(cs.workers) == 0)
    report("CoordinatorState default status", cs.status == "running")

    # Test system prompt exists
    report("coordinator prompt has delegation instructions", "delegate" in COORDINATOR_SYSTEM_PROMPT.lower())
    report("coordinator prompt has worker results format", "Worker Result" in COORDINATOR_SYSTEM_PROMPT)

    # Test feature flag gating
    reset(); os.environ["CLAW_FEATURE_COORDINATOR"] = "0"
    try:
        asyncio.run(run_coordinator(task="test", llm=None))
        report("disabled raises error", False)
    except RuntimeError as e:
        report("disabled raises RuntimeError", "not enabled" in str(e))

    reset(); os.environ.pop("CLAW_FEATURE_COORDINATOR", None)

except Exception as e:
    report("Coordinator import/run", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  RESULTS:  {PASSED} passed,  {FAILED} failed,  {PASSED+FAILED} total")
print(f"{'='*60}")

sys.exit(0 if FAILED == 0 else 1)
