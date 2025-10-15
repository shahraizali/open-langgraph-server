"""Thread state transformation helpers matching JavaScript server."""

from __future__ import annotations

from typing import Any, Dict, Optional
from ..storage.postgres_storage import ThreadStateData


def state_snapshot_to_thread_state(state: ThreadStateData) -> Dict[str, Any]:
    """
    Convert ThreadStateData to ThreadState format
    """
    return {
        "values": state.values,
        "next": state.next,
        "checkpoint": state.checkpoint,
        "metadata": state.metadata,
        "created_at": state.created_at,
        "parent_checkpoint": state.parent_checkpoint,
        "tasks": state.tasks,
    }


def create_thread_checkpoint(
    checkpoint_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    checkpoint_ns: str = "",
) -> Dict[str, Any]:
    """Create a thread checkpoint."""
    return {
        "checkpoint_id": checkpoint_id,
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_map": None,
    }
