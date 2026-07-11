"""Content-type aware tool-output crushers (Headroom-inspired, local-only).

Detect JSON / search / logs / code / prose and shrink aggressively while keeping
signal. Pair with ``tool_output_artifacts`` for reversible full-text storage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

ContentKind = Literal["json", "search", "log", "code", "html", "diff", "test", "prose"]

# Crush when larger than this (chars). Small outputs pass through untouched.
DEFAULT_CRUSH_THRESHOLD = 2_000
DEFAULT_TARGET_CHARS = 3_500

_SEARCH_TOOLS = frozenset({
    "grep", "glob", "search", "search_history", "find", "rg", "ctx_search",
})
_LOG_HINT = re.compile(
    r"(?i)\b(error|warn(?:ing)?|fatal|exception|traceback|failed|critical)\b"
)
_CODE_FENCE = re.compile(r"^```", re.M)
_LINE_NUMBERED = re.compile(r"(?m)^\s*\d+:")
_HTML_TAG = re.compile(r"(?i)</?(html|head|body|div|span|script|style|table)\b")
_DIFF_HDR = re.compile(r"(?m)^(diff --git |@@ |\+\+\+ |--- |Index: )")
_TEST_HINT = re.compile(
    r"(?i)\b(PASSED|FAILED|ERROR|===+|failures?=|errors?=|"
    r"<testcase\b|<testsuite\b|pytest|junit)\b"
)


@dataclass(frozen=True)
class CrushResult:
    kind: ContentKind
    text: str
    original_chars: int
    crushed_chars: int
    did_crush: bool

    @property
    def saved_chars(self) -> int:
        return max(0, self.original_chars - self.crushed_chars)


def detect_content_kind(text: str, tool_name: str = "") -> ContentKind:
    name = (tool_name or "").lower().strip()
    if name in _SEARCH_TOOLS or name.endswith("_search") or name.endswith(".grep"):
        return "search"
    if name in {"execute", "execute_command", "bash", "run_command"} and _TEST_HINT.search(text[:4000]):
        # Prefer test crush for pytest/junit dumps from shell tools.
        if text.count("PASSED") + text.count("FAILED") + text.count("<testcase") >= 2:
            return "test"

    stripped = text.lstrip()
    if stripped[:1] in "{[":
        try:
            json.loads(text)
            return "json"
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    sample = text[:8000]
    if _DIFF_HDR.search(sample) and (sample.count("\n+") + sample.count("\n-")) >= 8:
        return "diff"
    if _HTML_TAG.search(sample) and sample.count("<") >= 8:
        return "html"
    if _TEST_HINT.search(sample) and (
        "PASSED" in sample or "FAILED" in sample or "<testcase" in sample.lower()
    ):
        return "test"

    lines = text.splitlines()
    if lines:
        hot_count = sum(1 for ln in lines if _LOG_HINT.search(ln))
        # A few ERROR/WARN lines in a long dump → treat as log.
        if hot_count >= 2 and len(lines) >= 40:
            return "log"
        if hot_count >= max(3, len(lines[:80]) // 10):
            return "log"
        if _LINE_NUMBERED.search("\n".join(lines[:30])) or _CODE_FENCE.search(text[:500]):
            return "code"
        # High density of punctuation typical of code dumps
        code_chars = sum(1 for c in text[:2000] if c in "{}[];=<>")
        if code_chars > 80 and "\n" in text[:500]:
            return "code"

    return "prose"


def _head_tail(text: str, *, head: int, tail: int, label: str) -> str:
    if len(text) <= head + tail + 80:
        return text
    omitted = len(text) - head - tail
    return (
        f"{text[:head]}\n"
        f"... [{omitted} chars omitted — {label}] ...\n"
        f"{text[-tail:]}"
    )


def _crush_json(text: str, *, target: int) -> str:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _head_tail(text, head=target // 2, tail=target // 2, label="json text")

    if isinstance(data, list):
        n = len(data)
        sample = data[:3]
        keys: list[str] = []
        if sample and all(isinstance(x, dict) for x in sample):
            keyset: set[str] = set()
            for item in sample:
                keyset.update(str(k) for k in item.keys())
            keys = sorted(keyset)[:40]
        payload = {
            "_crushed": "json_array",
            "length": n,
            "keys": keys,
            "sample": sample,
            "note": "Full JSON stored in tool artifact; use retrieve_tool_result.",
        }
        out = json.dumps(payload, indent=2, default=str)
        if len(out) > target:
            return out[: target - 20] + "\n…[truncated]"
        return out

    if isinstance(data, dict):
        keys = list(data.keys())
        preview = {k: data[k] for k in keys[:12]}
        payload = {
            "_crushed": "json_object",
            "key_count": len(keys),
            "keys": keys[:40],
            "preview": preview,
            "note": "Full JSON stored in tool artifact; use retrieve_tool_result.",
        }
        out = json.dumps(payload, indent=2, default=str)
        if len(out) > target:
            return out[: target - 20] + "\n…[truncated]"
        return out

    return _head_tail(text, head=target // 2, tail=target // 2, label="json scalar")


def _crush_search(text: str, *, target: int) -> str:
    lines = text.splitlines()
    if len(lines) <= 60 and len(text) <= target:
        return text
    keep_head = 40
    keep_tail = 20
    if len(lines) <= keep_head + keep_tail:
        return text
    omitted = len(lines) - keep_head - keep_tail
    body = (
        "\n".join(lines[:keep_head])
        + f"\n... [{omitted} matching lines omitted] ...\n"
        + "\n".join(lines[-keep_tail:])
    )
    if len(body) > target:
        return _head_tail(body, head=target // 2, tail=target // 2, label="search")
    return body


def _crush_log(text: str, *, target: int) -> str:
    lines = text.splitlines()
    hot = [ln for ln in lines if _LOG_HINT.search(ln)]
    # Prefer unique hot lines (cap), then last N lines for context
    seen: set[str] = set()
    hot_unique: list[str] = []
    for ln in hot:
        key = ln.strip()
        if key in seen:
            continue
        seen.add(key)
        hot_unique.append(ln)
        if len(hot_unique) >= 40:
            break
    tail = lines[-40:] if len(lines) > 40 else lines
    parts = ["[log crush — errors/warnings + tail]"]
    if hot_unique:
        parts.append("-- signal --")
        parts.extend(hot_unique)
    parts.append("-- tail --")
    parts.extend(tail)
    body = "\n".join(parts)
    if len(body) > target:
        return _head_tail(body, head=target // 2, tail=target // 2, label="log")
    return body


def _crush_code(text: str, *, target: int) -> str:
    return _head_tail(text, head=max(target // 2, 800), tail=max(target // 2, 800), label="code")


def _crush_prose(text: str, *, target: int) -> str:
    return _head_tail(text, head=max(target // 2, 600), tail=max(target // 2, 600), label="prose")


def _crush_html(text: str, *, target: int) -> str:
    """Keep title/headings and strip most tags into a short text sketch."""
    title = re.search(r"(?is)<title[^>]*>(.*?)</title>", text)
    headings = re.findall(r"(?is)<h[1-3][^>]*>(.*?)</h[1-3]>", text)
    plain = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)
    plain = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", plain)
    plain = re.sub(r"(?s)<[^>]+>", " ", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    parts = ["[html crush]"]
    if title:
        parts.append(f"title: {re.sub(r'<[^>]+>', '', title.group(1)).strip()}")
    if headings:
        clean_h = [re.sub(r"<[^>]+>", "", h).strip() for h in headings[:20] if h.strip()]
        if clean_h:
            parts.append("headings: " + " | ".join(clean_h))
    parts.append("text: " + plain[: max(target - 200, 400)])
    body = "\n".join(parts)
    if len(body) > target:
        return body[: target - 20] + "\n…[truncated]"
    return body


def _crush_diff(text: str, *, target: int) -> str:
    """Keep file headers + hunk headers; drop long +/- body lines."""
    lines = text.splitlines()
    keep: list[str] = ["[diff crush — headers + sample hunks]"]
    body_kept = 0
    for ln in lines:
        if _DIFF_HDR.match(ln) or ln.startswith(("diff ", "index ")):
            keep.append(ln)
        elif ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")):
            if body_kept < 40:
                keep.append(ln)
                body_kept += 1
            elif body_kept == 40:
                keep.append("... [diff body lines omitted] ...")
                body_kept += 1
        elif ln.startswith(" "):
            continue
        else:
            if len(keep) < 80:
                keep.append(ln)
    body = "\n".join(keep)
    if len(body) > target:
        return _head_tail(body, head=target // 2, tail=target // 2, label="diff")
    return body


def _crush_test(text: str, *, target: int) -> str:
    """Keep failure blocks / summary lines from pytest or junit-ish output."""
    lines = text.splitlines()
    hot: list[str] = []
    for ln in lines:
        if _TEST_HINT.search(ln) or _LOG_HINT.search(ln):
            hot.append(ln)
            if len(hot) >= 60:
                break
    # Also keep a short failure traceback window around FAILED
    windows: list[str] = []
    for i, ln in enumerate(lines):
        if "FAILED" in ln or "ERROR" in ln or "AssertionError" in ln:
            start = max(0, i - 2)
            end = min(len(lines), i + 12)
            windows.extend(lines[start:end])
            windows.append("---")
            if len(windows) > 120:
                break
    parts = ["[test crush — failures + signal]"]
    if hot:
        parts.append("-- signal --")
        parts.extend(hot)
    if windows:
        parts.append("-- failures --")
        parts.extend(windows[:120])
    if len(lines) > 30:
        parts.append("-- tail --")
        parts.extend(lines[-25:])
    body = "\n".join(parts)
    if len(body) > target:
        return _head_tail(body, head=target // 2, tail=target // 2, label="test")
    return body


def crush_tool_output(
    text: str,
    *,
    tool_name: str = "",
    threshold: int = DEFAULT_CRUSH_THRESHOLD,
    target_chars: int = DEFAULT_TARGET_CHARS,
) -> CrushResult:
    """Return a crushed view of ``text`` when over threshold; else unchanged."""
    if not isinstance(text, str):
        text = str(text)
    original = len(text)
    kind = detect_content_kind(text, tool_name)
    if original <= threshold:
        return CrushResult(kind=kind, text=text, original_chars=original, crushed_chars=original, did_crush=False)

    if kind == "json":
        crushed = _crush_json(text, target=target_chars)
    elif kind == "search":
        crushed = _crush_search(text, target=target_chars)
    elif kind == "log":
        crushed = _crush_log(text, target=target_chars)
    elif kind == "code":
        crushed = _crush_code(text, target=target_chars)
    elif kind == "html":
        crushed = _crush_html(text, target=target_chars)
    elif kind == "diff":
        crushed = _crush_diff(text, target=target_chars)
    elif kind == "test":
        crushed = _crush_test(text, target=target_chars)
    else:
        crushed = _crush_prose(text, target=target_chars)

    # Never expand
    if len(crushed) >= original:
        crushed = _head_tail(text, head=target_chars // 2, tail=target_chars // 2, label=kind)

    return CrushResult(
        kind=kind,
        text=crushed,
        original_chars=original,
        crushed_chars=len(crushed),
        did_crush=len(crushed) < original,
    )
