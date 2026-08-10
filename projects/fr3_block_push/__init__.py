"""FR3 block-pushing task environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment.config import BlockPushConfig

__all__ = ["BlockPushConfig"]


def __getattr__(name: str):
    """Keep lightweight IPC imports from eagerly importing MuJoCo."""
    if name == "BlockPushConfig":
        from .environment.config import BlockPushConfig

        return BlockPushConfig
    raise AttributeError(name)
