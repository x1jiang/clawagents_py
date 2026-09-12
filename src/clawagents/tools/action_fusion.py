"""Edit/command fusion at the registry boundary, with host-owned command policy."""
from __future__ import annotations

import asyncio
import hashlib
import re
import weakref
from typing import Any

FUSION_TOOLS = frozenset({'write_file', 'edit_file', 'apply_patch', 'hashline_edit'})
THEN_RUN_PARAMETER = {
    'type': 'object',
    'description': (
        'Command to run next on this file after the edit succeeds, e.g. build, test, '
        'run or check it. Skipped if the edit or syntax check fails; a nonzero exit '
        'is reported but keeps the edit. Uses execute permissions and sandbox.'
    ),
    'properties': {
        'command': {'type': 'string', 'description': 'Shell command to run after this edit.'},
        'timeout': {'type': 'integer', 'description': 'Foreground timeout in milliseconds (same as execute).', 'minimum': 1},
    },
    'required': False,
    'additionalProperties': False,
}
# Weak values avoid retaining every edited path; waiters keep their lock alive.
_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _file_lock(backend: Any, path: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    locks = _locks.setdefault(loop, weakref.WeakValueDictionary())
    # Local safe_path already resolves symlinks. Remote/memory namespaces must
    # not collide with host paths or another independent backend instance.
    inner = backend
    while getattr(inner, '_inner', None) is not None:
        inner = inner._inner
    key = ('local' if getattr(inner, 'kind', None) == 'local' else id(inner), path)
    lock = locks.get(key)
    if lock is None:
        lock = locks[key] = asyncio.Lock()
    return lock


def _command_args(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - {'command', 'timeout'}:
        raise ValueError('then_run must contain command and optional timeout only')
    command = value.get('command')
    if not isinstance(command, str) or not command.strip():
        raise ValueError('then_run.command must be a nonempty string')
    result: dict[str, Any] = {'command': command}
    if 'timeout' in value:
        timeout = value['timeout']
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError('then_run.timeout must be a positive integer in milliseconds')
        result['timeout'] = timeout
    return result


def _joined(edit: Any, status: str, command: Any = None, reason: str = '') -> Any:
    from clawagents.tools.registry import ToolResult, truncate_tool_output

    prefix = edit.raw_output if isinstance(edit.raw_output, str) else str(edit.output or '')
    if edit.error and edit.error not in prefix:
        prefix += '\n' + edit.error
    suffix = reason
    if command is not None:
        suffix = command.raw_output if isinstance(command.raw_output, str) else str(command.output or '')
        if command.error and command.error not in suffix:
            suffix += '\n' + command.error
    text = f'{prefix.rstrip()}\n\n[then_run:{status}]\n{suffix}'
    return ToolResult(
        # Mutation succeeded but verification did not: advertise failure while
        # retaining explicit edit confirmation so the model does not reapply it.
        success=bool(edit.success and status == 'succeeded'),
        output=truncate_tool_output(text), raw_output=text,
        error=None if status == 'succeeded' else (reason or getattr(command, 'error', None) or f'then_run {status}; see output; successful edits are kept'),
        added_tool_names=edit.added_tool_names,
        mutation_success=bool(edit.success),
    )


async def execute_edit(registry: Any, tool_name: str, args: dict[str, Any], *, run_context: Any, followup: Any) -> Any:
    """Serialize every supported edit, including edits without a follow-up.

    ``followup(args, unchanged)`` belongs to the agent's policy/approval layer.
    Direct registry consumers without that layer keep the edit and receive a
    skipped marker, never an unapproved shell execution.
    """
    from clawagents.tools.registry import ToolResult

    fused = 'then_run' in args
    try:
        command_args = _command_args(args['then_run']) if fused else None
    except ValueError as exc:
        return ToolResult(False, '[then_run:skipped]', str(exc))
    tool = registry.get(tool_name)
    backend = getattr(tool, '_sb', None)
    try:
        path = backend.safe_path(str(args.get('path') or args.get('file_path') or '')) if backend else None
    except Exception as exc:
        return ToolResult(False, '[then_run:skipped]' if fused else '', str(exc))

    async def perform() -> Any:
        edit = await registry._execute_tool(
            tool_name, args, run_context=run_context, canonical_path=path,
        )
        if not fused:
            return edit
        if not edit.success:
            return _joined(edit, 'skipped', reason='Edit failed; command was not run.')
        text = str(edit.raw_output or '')
        if re.search(r'^\[syntax_gate\].*: FAILED', text, re.MULTILINE):
            return _joined(edit, 'skipped', reason='Syntax check failed; edit kept, command was not run.')
        if followup is None or backend is None or path is None:
            return _joined(edit, 'skipped', reason='Command requires the agent policy/approval execution path; edit kept.')
        try:
            async def fingerprint() -> tuple[str, str]:
                # Re-resolve the original argument too, detecting symlink retargeting.
                current_path = backend.safe_path(str(args.get('path') or args.get('file_path') or ''))
                content = await backend.read_file_bytes(current_path)
                return current_path, hashlib.sha256(content).hexdigest()

            expected = (path, hashlib.sha256(await backend.read_file_bytes(path)).hexdigest())

            async def unchanged() -> bool:
                try:
                    return await fingerprint() == expected
                except Exception:
                    return False

            await asyncio.sleep(0)
            if not await unchanged():
                return _joined(edit, 'skipped', reason='File changed after edit; command was not run.')
            status, result = await followup(command_args, unchanged)
            if status == 'succeeded' and result.success:
                from clawagents.efficiency import get_efficiency
                get_efficiency(run_context)['round_trips_avoided'] += 1
            return _joined(edit, status, result)
        except Exception as exc:
            return _joined(edit, 'skipped', reason=f'Follow-up unavailable: {exc}; edit kept.')

    if backend is None or path is None:
        return await perform()
    async with _file_lock(backend, path):
        try:
            current_path = backend.safe_path(str(args.get('path') or args.get('file_path') or ''))
        except Exception as exc:
            return ToolResult(False, '[then_run:skipped]' if fused else '', str(exc))
        if current_path != path:
            return ToolResult(False, '[then_run:skipped]' if fused else '',
                              'File target changed while waiting for an edit lock; refresh and retry.')
        return await perform()
