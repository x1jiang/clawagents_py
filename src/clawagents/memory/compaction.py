from typing import Optional, List, Dict, Any
import math
import logging

from clawagents.providers.llm import LLMProvider, LLMMessage
from clawagents.tokenizer import count_tokens

logger = logging.getLogger(__name__)

class AgentMessage:
    def __init__(self, role: str, content: str, timestamp: Optional[float] = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp

BASE_CHUNK_RATIO = 0.4
MIN_CHUNK_RATIO = 0.15
SAFETY_MARGIN = 1.2
DEFAULT_SUMMARY_FALLBACK = "No prior history."

def estimate_tokens(message: AgentMessage) -> int:
    """Token count using tiktoken (falls back to heuristic if unavailable)."""
    return count_tokens(message.content or "")

def estimate_messages_tokens(messages: List[AgentMessage]) -> int:
    return sum(estimate_tokens(m) for m in messages)

def chunk_messages_by_max_tokens(messages: List[AgentMessage], max_tokens: int) -> List[List[AgentMessage]]:
    if not messages:
        return []
        
    chunks: List[List[AgentMessage]] = []
    current_chunk: List[AgentMessage] = []
    current_tokens = 0

    for message in messages:
        message_tokens = estimate_tokens(message)
        if current_chunk and current_tokens + message_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
            
        current_chunk.append(message)
        current_tokens += message_tokens
        
        if message_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

async def summarize_with_fallback(
    llm: LLMProvider,
    messages: List[AgentMessage],
    max_chunk_tokens: int,
    context_window: int,
    previous_summary: Optional[str] = None
) -> str:
    
    if not messages:
        return previous_summary or DEFAULT_SUMMARY_FALLBACK

    chunks = chunk_messages_by_max_tokens(messages, max_chunk_tokens)
    current_summary = previous_summary or "No prior events."

    for chunk in chunks:
        text_log = "\n\n".join([f"[{m.role.upper()}]: {m.content}" for m in chunk])

        prompt = f"""You are a summarization engine for an AI agent. 
Compress the following event log into a concise technical summary.
Focus on actions taken, tools used, results observed, and current state.
Do NOT lose critical facts like file paths, errors, or exact values extracted.

Previous summary state:
{current_summary}

New events to summarize into the state:
{text_log}

Return ONLY the updated comprehensive summary."""

        try:
            resp = await llm.chat([LLMMessage(role="user", content=prompt)])
            current_summary = resp.content.strip()
        except Exception as e:
            logging.error(f"[Compaction] LLM Summarization failed, falling back to basic join. Error: {e}")
            current_summary += f"\n[Summarized {len(chunk)} messages]"
            
    return current_summary or DEFAULT_SUMMARY_FALLBACK

def prune_history_for_context_share(
    messages: List[AgentMessage],
    max_context_tokens: int,
    max_history_share: float = 0.5
) -> Dict[str, Any]:
    
    budget_tokens = max(1, math.floor(max_context_tokens * max_history_share))
    
    total_tokens = estimate_messages_tokens(messages)
    all_dropped_messages: List[AgentMessage] = []
    dropped_chunks = 0
    dropped_tokens = 0
    drop_idx = 0

    while drop_idx < len(messages) and total_tokens > budget_tokens:
        msg = messages[drop_idx]
        msg_tokens = estimate_tokens(msg)
        all_dropped_messages.append(msg)
        dropped_tokens += msg_tokens
        total_tokens -= msg_tokens
        dropped_chunks += 1
        drop_idx += 1

    kept_messages = list(messages[drop_idx:])
        
    return {
        "messages": kept_messages,
        "dropped_messages_list": all_dropped_messages,
        "dropped_chunks": dropped_chunks,
        "dropped_tokens": dropped_tokens,
        "kept_tokens": estimate_messages_tokens(kept_messages)
    }


# ---------------------------------------------------------------------------
# Hardened compression (v6.5): head/tail protection + anti-thrash detector
#
# The legacy ``prune_history_for_context_share`` drops messages purely from the
# front, which can silently lose the system prompt or the active task. The
# helpers below add the guardrails that real agent runs need:
#
# - **Protect head**: keep the first ``protect_first_n`` messages (system
#   prompt + first user turn). Patterned after Hermes ``ContextCompressor``.
# - **Protect tail**: keep the last ``protect_last_n`` messages so the recent
#   tool outputs and the active task never disappear.
# - **Preserve last user message**: even when the tail budget is small, the
#   most recent user message is *never* dropped — losing it strands the agent
#   on a stale objective.
# - **Static fallback summary**: if ``summarize_with_fallback`` returns the
#   default no-history string and we did drop turns, inject a visible marker
#   so the model knows context was lost.
# - **Anti-thrash detector**: ``is_compression_thrashing`` flags consecutive
#   ineffective compressions so callers can surface a warning instead of
#   looping endlessly.
# ---------------------------------------------------------------------------

DEFAULT_PROTECT_FIRST = 1  # system prompt
DEFAULT_PROTECT_LAST = 4   # last user turn + recent assistant/tool exchanges

INEFFECTIVE_SAVINGS_PCT = 10.0
"""Compression below this savings percentage counts as ineffective."""

THRASH_THRESHOLD = 2
"""Consecutive ineffective compressions before we declare thrashing."""

_COMPRESSION_NOTE = (
    "[Note: Earlier conversation turns were compacted into a handoff summary "
    "to free context space. Build on that summary rather than re-doing work.]"
)

_FALLBACK_SUMMARY = (
    "[Context summary unavailable — {n} earlier turn(s) were removed to "
    "free space but could not be summarized. Continue based on the recent "
    "messages below and the current state of any files/resources.]"
)


def _last_user_index(messages: List[AgentMessage]) -> int:
    """Return index of the most recent ``role == 'user'`` message, else -1."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            return i
    return -1


async def compress_messages_safe(
    llm: LLMProvider,
    messages: List[AgentMessage],
    *,
    context_window: int,
    max_chunk_tokens: Optional[int] = None,
    protect_first_n: int = DEFAULT_PROTECT_FIRST,
    protect_last_n: int = DEFAULT_PROTECT_LAST,
    previous_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Compress a transcript while honouring head/tail protection.

    The output is a list of messages shaped as::

        [head]                  # first ``protect_first_n`` (with compression note in system)
        [{role: assistant|user, content: <summary>}]
        [tail]                  # last ``protect_last_n`` (always includes most recent user message)

    Returns a dict with the new ``messages``, the list of dropped messages,
    a ``compression_savings_pct`` figure, and a boolean ``effective`` flag
    derived from :data:`INEFFECTIVE_SAVINGS_PCT`.
    """
    if not messages:
        return {
            "messages": [],
            "dropped_messages_list": [],
            "summary": previous_summary or DEFAULT_SUMMARY_FALLBACK,
            "compression_savings_pct": 0.0,
            "effective": False,
        }

    n = len(messages)
    head_end = max(0, min(protect_first_n, n))
    tail_start_naive = max(head_end, n - max(0, protect_last_n))

    # Always keep the last user message in the tail.
    last_user = _last_user_index(messages)
    if last_user >= head_end and last_user < tail_start_naive:
        tail_start = last_user
    else:
        tail_start = tail_start_naive

    # Tool-pair safety: never split an assistant tool_call from its tool results.
    try:
        from clawagents.config.features import is_enabled

        if is_enabled("compact_tool_pair_safe"):
            while tail_start < n and messages[tail_start].role == "tool":
                tail_start += 1
            # Also walk backward: if the message just before tail is an
            # assistant with tool_calls conceptually represented as content
            # markers, keep scanning for orphan tools already handled above.
            # For AgentMessage we only have roles — snap forward past tools.
    except Exception:
        pass

    head = list(messages[:head_end])
    middle = list(messages[head_end:tail_start])
    tail = list(messages[tail_start:])

    if not middle:
        # Nothing to compress — return the original transcript unchanged.
        return {
            "messages": list(messages),
            "dropped_messages_list": [],
            "summary": previous_summary or "",
            "compression_savings_pct": 0.0,
            "effective": False,
        }

    chunk_tokens = max_chunk_tokens or max(512, math.floor(context_window * BASE_CHUNK_RATIO))
    summary = await summarize_with_fallback(
        llm,
        middle,
        max_chunk_tokens=chunk_tokens,
        context_window=context_window,
        previous_summary=previous_summary,
    )

    if not summary or summary == DEFAULT_SUMMARY_FALLBACK:
        summary = _FALLBACK_SUMMARY.format(n=len(middle))

    # Prepend the compression note to the system prompt (if any) so the model
    # knows some history is now a summary rather than verbatim turns.
    if head and head[0].role == "system" and _COMPRESSION_NOTE not in (head[0].content or ""):
        head[0] = AgentMessage(
            role="system",
            content=(head[0].content or "") + ("\n\n" + _COMPRESSION_NOTE if head[0].content else _COMPRESSION_NOTE),
            timestamp=head[0].timestamp,
        )

    # Choose a role for the summary that won't collide with neighbours.
    last_head_role = head[-1].role if head else "user"
    first_tail_role = tail[0].role if tail else "user"
    summary_role = "user" if last_head_role in ("assistant", "tool") else "assistant"
    if summary_role == first_tail_role:
        flipped = "user" if summary_role == "assistant" else "assistant"
        if flipped != last_head_role:
            summary_role = flipped

    summary_msg = AgentMessage(role=summary_role, content=summary)

    new_messages = head + [summary_msg] + tail
    before_tokens = estimate_messages_tokens(messages)
    after_tokens = estimate_messages_tokens(new_messages)
    saved = before_tokens - after_tokens
    pct = (saved / before_tokens * 100.0) if before_tokens > 0 else 0.0

    return {
        "messages": new_messages,
        "dropped_messages_list": middle,
        "summary": summary,
        "compression_savings_pct": pct,
        "effective": pct >= INEFFECTIVE_SAVINGS_PCT,
    }


def is_compression_thrashing(savings_history: List[float]) -> bool:
    """Return True if the last ``THRASH_THRESHOLD`` compressions all saved
    less than :data:`INEFFECTIVE_SAVINGS_PCT`.

    Callers should append each compression's ``compression_savings_pct`` to
    a list and pass it here. The helper looks only at the tail of the list,
    so it stays O(THRASH_THRESHOLD) regardless of history length.
    """
    if len(savings_history) < THRASH_THRESHOLD:
        return False
    recent = savings_history[-THRASH_THRESHOLD:]
    return all(s < INEFFECTIVE_SAVINGS_PCT for s in recent)


__all__ = [
    "AgentMessage",
    "BASE_CHUNK_RATIO",
    "MIN_CHUNK_RATIO",
    "SAFETY_MARGIN",
    "estimate_tokens",
    "estimate_messages_tokens",
    "chunk_messages_by_max_tokens",
    "summarize_with_fallback",
    "prune_history_for_context_share",
    # Hardened compression (v6.5)
    "DEFAULT_PROTECT_FIRST",
    "DEFAULT_PROTECT_LAST",
    "INEFFECTIVE_SAVINGS_PCT",
    "THRASH_THRESHOLD",
    "compress_messages_safe",
    "is_compression_thrashing",
]
