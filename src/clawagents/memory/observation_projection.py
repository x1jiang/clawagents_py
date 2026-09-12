"""Opt-in observation lifecycle experiment; persisted history stays unmodified.

Counts successful harness model calls, not hidden provider transport retries.
Budget protection can still compact a result before its full-send allowance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import copy
import hashlib
import logging
import re
from typing import Any

from clawagents.efficiency import contains_evidence_receipt, get_efficiency
from clawagents.providers.llm import LLMMessage
from clawagents.tool_output_artifacts import load_tool_artifact, store_tool_artifact

logger = logging.getLogger(__name__)
_TOOLS = frozenset({'read_file', 'grep', 'execute'})
_HEADER = re.compile(r'\[Observation id=([a-zA-Z0-9_.-]+) sha256=[a-f0-9]{64}\]\n')


def should_delay_observation(context: Any, tool_name: str, output: Any, success: bool) -> bool:
    allowance = getattr(context, 'observation_full_sends', 0)
    return (type(allowance) is int and allowance in (1, 2)
            and tool_name in _TOOLS and success is True
            and isinstance(output, str) and 10_000 <= len(output) <= 600_000
            and not contains_evidence_receipt(output))


def archive_observation(context: Any, *, tool_name: str, call_id: str,
                        output: str, success: bool) -> str | None:
    """Archive eligible exact output before replacing first-sight crushing."""
    if not should_delay_observation(context, tool_name, output, success):
        return None
    try:
        workspace = context._metadata.get('workspace')
        digest = hashlib.sha256(output.encode('utf-8')).hexdigest()
        artifact, path = store_tool_artifact(tool_name=tool_name, tool_use_id=call_id,
                                            output=output, workspace=workspace)
        full = f'[Observation id={artifact} sha256={digest}]\n{output}'
        handle = (f'[Observation id={artifact} tool={tool_name} chars={len(output)} sha256={digest}]\n'
                  f'{output[:1024]}\n[middle omitted; full observation archived]\n{output[-1024:]}\n'
                  f'Read exact pages with retrieve_tool_result(id="{artifact}", offset=0); '
                  'continue with returned next_offset.\n')
        context._observations[artifact] = dict(artifact_id=artifact, path=str(path),
            workspace=workspace, digest=digest, full=full, handle=handle, sends=0)
        return full
    except Exception:
        logger.debug('Observation archival failed; using normal output path', exc_info=True)
        return None


@dataclass
class ObservationProjection:
    messages: list[LLMMessage]
    visible: set[str] = field(default_factory=set)
    full: set[str] = field(default_factory=set)
    saved_tokens: int = 0
    originals: dict[int, tuple[LLMMessage, str]] = field(default_factory=dict)
    replacements: dict[str, str] = field(default_factory=dict)
    committed: bool = False
    valid: bool = True


def project_observations(messages: list[LLMMessage], context: Any) -> ObservationProjection:
    """Build a disposable provider view without advancing any send counters."""
    projection = ObservationProjection(messages=messages)
    allowance = getattr(context, 'observation_full_sends', 0)
    entries = getattr(context, '_observations', {})
    if allowance not in (1, 2) or not entries:
        return projection
    try:
        out = list(messages)
        verified: dict[str, bool] = {}
        for index, message in enumerate(messages):
            content = message.content
            if not isinstance(content, str) or not (message.role == 'tool' or
                    (message.role == 'user' and content.startswith(('[Tool Result]', '[Tool Results]')))):
                continue
            if contains_evidence_receipt(content):
                continue
            updated = content
            for artifact in {match.group(1) for match in _HEADER.finditer(content)}:
                entry = entries.get(artifact)
                if entry is None:
                    continue
                full = entry['full']
                if full not in updated:
                    continue
                projection.visible.add(artifact)
                if entry['sends'] < allowance:
                    projection.full.add(artifact)
                    continue
                if artifact not in verified:
                    ok, body, _ = load_tool_artifact(artifact, workspace=entry['workspace'], max_chars=600_001)
                    verified[artifact] = ok and hashlib.sha256(body.encode('utf-8')).hexdigest() == entry['digest']
                if not verified[artifact]:
                    continue
                updated = updated.replace(full, entry['handle'])
                projection.replacements[entry['handle']] = full
            if updated != content:
                from clawagents.graph.tool_observation import _estimate_tokens
                projected = copy(message)
                projected.content = updated
                out[index] = projected
                projection.originals[id(projected)] = (message, updated)
                projection.saved_tokens += max(0, _estimate_tokens(content) - _estimate_tokens(updated))
        projection.messages = out
        return projection
    except Exception:
        logger.debug('Observation projection failed; preserving original history', exc_info=True)
        return ObservationProjection(messages=messages, valid=False)


def commit_projection(projection: ObservationProjection, context: Any) -> None:
    """Commit telemetry only after a successful caller response; idempotent."""
    if projection.committed or not projection.valid:
        return
    projection.committed = True
    entries = getattr(context, '_observations', {})
    # Release full strings once compaction has removed them from the live view.
    # Exact artifact files remain available for recall and session history.
    for artifact in set(entries) - projection.visible:
        del entries[artifact]
    for artifact in projection.full:
        if artifact in entries:
            entries[artifact]['sends'] += 1
    if projection.saved_tokens:
        get_efficiency(context)['tokens_avoided_by_handles'] += projection.saved_tokens


def restore_observations(messages: list[LLMMessage], projection: ObservationProjection) -> list[LLMMessage]:
    """Remove preview substitutions before recording history; retain hook edits."""
    restored = []
    for message in messages:
        original = projection.originals.get(id(message))
        if original and message.content == original[1]:
            restored_message = copy(message)
            restored_message.content = original[0].content
            restored.append(restored_message)
            continue
        # Hooks often copy/rebuild messages. Restore by the exact substitution
        # text too, keeping their metadata and surrounding edits intact.
        content = message.content
        if isinstance(content, str):
            updated = content
            for handle, full in projection.replacements.items():
                updated = updated.replace(handle, full)
            if updated != content:
                message = copy(message)
                message.content = updated
        restored.append(message)
    return restored if projection.originals else messages
