from typing import Optional, List, Dict, Any
import math
import logging

from clawagents.providers.llm import LLMProvider, LLMMessage

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
    """Rough estimation: 4 chars per token"""
    return math.ceil(len(message.content or "") / 4)

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
