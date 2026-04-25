"""Multimodal helpers — image sanitization and friends.

Public exports:

    from clawagents.media import sanitize_image_block, sanitize_tool_output
    from clawagents.media import is_pillow_available
"""

from clawagents.media.images import (
    is_pillow_available,
    sanitize_image_block,
    sanitize_tool_output,
)

__all__ = [
    "is_pillow_available",
    "sanitize_image_block",
    "sanitize_tool_output",
]
