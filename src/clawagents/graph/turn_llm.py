"""Provider-facing portion of one ReAct turn.

The loop decides *what* to do with a response.  This collaborator is only
responsible for constructing the provider request and reporting its result.
That separation keeps provider quirks (schemas, streaming and cache usage)
out of control-flow code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from clawagents.providers.llm import LLMMessage, LLMResponse, NativeToolSchema

from .run_runtime import HookDispatcher, RunEvents


@dataclass(frozen=True)
class LLMCallResult:
    response: LLMResponse
    resolved_model_name: str | None


class TurnLLMCaller:
    """Build and execute one provider request without owning loop state."""

    def __init__(
        self,
        *,
        llm: Any,
        events: RunEvents,
        hooks: HookDispatcher,
        registry: Any,
        session_writer: Any,
        external_hooks: Any,
        accumulate_usage: Callable[[LLMResponse], Any],
    ) -> None:
        self._llm = llm
        self._events = events
        self._hooks = hooks
        self._registry = registry
        self._session_writer = session_writer
        self._external_hooks = external_hooks
        self._accumulate_usage = accumulate_usage

    async def call(
        self,
        messages: list[LLMMessage],
        *,
        resolved_model_name: str | None,
        use_native_tools: bool,
        tools_supplied: bool,
        initial_schemas: list[NativeToolSchema] | None,
        handoffs: list[Any],
        streaming: bool,
        cancel_event: Any,
        run_context: Any,
        output_type: type | None,
    ) -> LLMCallResult:
        self._configure_structured_output(output_type)
        request_messages = self._with_doom_recovery_instruction(messages, run_context)
        schemas = self._build_schemas(
            use_native_tools=use_native_tools,
            tools_supplied=tools_supplied,
            initial_schemas=initial_schemas,
            handoffs=handoffs,
            run_context=run_context,
        )

        def on_chunk(chunk: str) -> None:
            self._events.typed("assistant_delta", {"delta": chunk})

        if self._hooks.hooks:
            await self._hooks.fire("on_llm_start", resolved_model_name or "", messages)
        response = await self._llm.chat(
            request_messages,
            on_chunk=on_chunk if streaming else None,
            cancel_event=cancel_event,
            tools=schemas,
        )
        model_name = resolved_model_name or response.model
        usage = self._accumulate_usage(response)
        if self._hooks.hooks:
            await self._hooks.fire(
                "on_llm_end",
                response.model or model_name or "",
                response.content or "",
                usage,
            )
        if self._session_writer:
            self._session_writer.write_usage(
                response.tokens_used,
                cache_read_tokens=response.cache_read_tokens,
                cache_creation_tokens=response.cache_creation_tokens,
            )
        if self._external_hooks:
            try:
                await self._external_hooks.post_llm(
                    response.content[:500], len(response.tool_calls or [])
                )
            except Exception:
                pass
        self._emit_cache_usage(response)
        return LLMCallResult(response=response, resolved_model_name=model_name)

    def _configure_structured_output(self, output_type: type | None) -> None:
        try:
            from clawagents.config.features import is_enabled
            from clawagents.structured_output import schema_from_output_type

            schema = (
                schema_from_output_type(output_type)
                if is_enabled("structured_output") and output_type is not None
                else None
            )
            setattr(self._llm, "_structured_json_schema", schema)
        except Exception:
            pass

    @staticmethod
    def _with_doom_recovery_instruction(
        messages: list[LLMMessage], run_context: Any
    ) -> list[LLMMessage]:
        metadata = getattr(run_context, "_metadata", None)
        if not (isinstance(metadata, dict) and metadata.get("doom_force_response")):
            return messages
        return [
            *messages,
            LLMMessage(
                role="user",
                content=(
                    "CRITICAL recovery instruction: Do NOT emit any "
                    "<think>...</think> blocks or private chain-of-thought. "
                    "Respond with the next tool call or final answer only."
                ),
            ),
        ]

    def _build_schemas(
        self,
        *,
        use_native_tools: bool,
        tools_supplied: bool,
        initial_schemas: list[NativeToolSchema] | None,
        handoffs: list[Any],
        run_context: Any,
    ) -> list[NativeToolSchema] | None:
        schemas = initial_schemas
        if use_native_tools and tools_supplied:
            schemas = self._registry.to_native_schemas()
            if handoffs:
                params = {
                    "reason": {
                        "type": "string",
                        "description": "Free-text rationale for why the handoff is appropriate.",
                        "required": False,
                    }
                }
                schemas = [
                    *schemas,
                    *[
                        NativeToolSchema(
                            name=handoff.name,
                            description=handoff.description,
                            parameters=params,
                        )
                        for handoff in handoffs
                    ],
                ]
        allowed = getattr(run_context, "active_skill_allowed_tools", None)
        if schemas and allowed is not None:
            control_plane = {
                "use_skill",
                "list_skills",
                "retrieve_tool_result",
                "activate_tool_group",
            }
            schemas = [
                schema
                for schema in schemas
                if schema.name in allowed or schema.name in control_plane
            ]
        return schemas

    def _emit_cache_usage(self, response: LLMResponse) -> None:
        try:
            from clawagents.config.features import is_enabled

            if not (is_enabled("cache_tracking") and response.prompt_tokens > 0):
                return
            percent = response.cache_read_tokens / response.prompt_tokens * 100
            self._events.emit(
                "context",
                {
                    "message": (
                        f"cache: {percent:.0f}% hit "
                        f"({response.cache_read_tokens}/{response.prompt_tokens} prompt tokens, "
                        f"{response.cache_creation_tokens} created)"
                    )
                },
            )
        except Exception:
            pass
