"""Database-backed checkpoint storage system using LangGraph's PostgresSaver."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, AsyncGenerator, Tuple, Sequence

logger = logging.getLogger(__name__)

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
    HAS_POSTGRES_CHECKPOINT = True
except ImportError as e:
    logger.warning(f"PostgreSQL checkpoint dependencies not available: {e}")
    # Provide minimal fallback types
    PostgresSaver = None
    Checkpoint = dict
    CheckpointMetadata = dict
    HAS_POSTGRES_CHECKPOINT = False

try:
    from langchain_core.runnables.config import RunnableConfig
except ImportError:
    # Fallback type for RunnableConfig
    RunnableConfig = dict

EXCLUDED_KEYS = ["checkpoint_ns", "checkpoint_id", "run_id", "thread_id"]


class DatabaseCheckpointSaver:
    """
    Database-backed checkpoint saver using LangGraph's PostgresSaver.

    This wraps PostgresSaver to provide a persistent instance for the application.
    """

    def __init__(self):
        # Get database URL from environment
        self.database_url = os.getenv(
            "DATABASE_URL", "postgresql://postgres:password@localhost:5432/langgraph"
        )
        self._context_manager = None
        self._saver = None
        self._initialized = False

    def _get_saver(self) -> PostgresSaver:
        """Get or create PostgresSaver instance."""
        if self._saver is None:
            raise RuntimeError(
                "Checkpoint storage not initialized. Call initialize() first."
            )
        return self._saver

    async def initialize(self) -> None:
        """Initialize the checkpoint storage."""
        if self._initialized:
            return

        if not HAS_POSTGRES_CHECKPOINT:
            logger.warning("PostgreSQL checkpoint dependencies not available, using fallback")
            self._saver = self  # Use self as fallback
            self._initialized = True
            return

        try:
            # Create context manager and enter it
            self._context_manager = PostgresSaver.from_conn_string(self.database_url)
            self._saver = self._context_manager.__enter__()

            # Setup the database
            self._saver.setup()

            self._initialized = True
            logger.info("Database checkpoint storage initialized using PostgresSaver")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL checkpoint storage: {e}")
            logger.warning("Using fallback checkpoint storage")
            self._saver = self  # Use self as fallback
            self._initialized = True

    def get_tuple(
        self, config: Dict[str, Any]
    ) -> Optional[Tuple[Checkpoint, CheckpointMetadata]]:
        """Get checkpoint tuple from storage (sync version)."""
        if not self._initialized:
            raise RuntimeError("Checkpoint storage not initialized")
        
        # If using fallback (self as saver), return None
        if self._saver is self:
            return None
        
        return self._saver.get_tuple(config)

    async def aget_tuple(
        self, config: Dict[str, Any]
    ) -> Optional[Tuple[Checkpoint, CheckpointMetadata]]:
        """Get checkpoint tuple from storage (async version)."""
        if not self._initialized:
            raise RuntimeError("Checkpoint storage not initialized")
        return self._saver.get_tuple(config)

    def list(self, config: Dict[str, Any]):
        """List checkpoints for a thread (sync version)."""
        if not self._initialized:
            raise RuntimeError("Checkpoint storage not initialized")
        
        # If using fallback (self as saver), return empty list
        if self._saver is self:
            return []
        
        return self._saver.list(config)

    async def alist(
        self,
        config: Dict[str, Any],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Tuple[Checkpoint, CheckpointMetadata], None]:
        """List checkpoints for a thread (async version) with filtering support."""
        if not self._initialized:
            raise RuntimeError("Checkpoint storage not initialized")

        # Build the parameters for the underlying list method
        list_kwargs = {}
        if filter is not None:
            list_kwargs["filter"] = filter
        if before is not None:
            list_kwargs["before"] = before
        if limit is not None:
            list_kwargs["limit"] = limit

        # Pass any additional kwargs
        list_kwargs.update(kwargs)

        try:
            # Try to call with parameters if the underlying saver supports them
            for item in self._saver.list(config, **list_kwargs):
                yield item
        except TypeError as e:
            # If the underlying saver doesn't support these parameters,
            # fall back to basic list and implement filtering manually
            if any(param in str(e) for param in ["filter", "before", "limit"]):
                count = 0
                for item in self._saver.list(config):
                    # Manual filtering logic
                    checkpoint, metadata = item

                    # Apply before filter
                    if before and "configurable" in before:
                        before_checkpoint_id = before["configurable"].get(
                            "checkpoint_id"
                        )
                        if before_checkpoint_id and metadata.get("checkpoint_id"):
                            if metadata["checkpoint_id"] >= before_checkpoint_id:
                                continue

                    # Apply metadata filter
                    if filter:
                        metadata_matches = all(
                            metadata.get(k) == v for k, v in filter.items()
                        )
                        if not metadata_matches:
                            continue

                    yield item
                    count += 1

                    # Apply limit
                    if limit and count >= limit:
                        break
            else:
                # Re-raise if it's a different TypeError
                raise

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Put a checkpoint into storage with proper metadata merging (sync version)."""
        if not self._initialized:
            raise RuntimeError("Checkpoint storage not initialized")

        # If using fallback (self as saver), return config
        if self._saver is self:
            logger.debug("Fallback: Ignoring put checkpoint")
            return config

        configurable = config.get("configurable", {})
        filtered_configurable = {
            k: v
            for k, v in configurable.items()
            if not k.startswith("__") and k not in EXCLUDED_KEYS
        }

        merged_metadata = {
            **filtered_configurable,
            **config.get("metadata", {}),
            **metadata,
        }

        # Use PostgresSaver implementation with new_versions (sync version)
        if new_versions is None:
            new_versions = {"version": 1}
        return self._saver.put(config, checkpoint, merged_metadata, new_versions)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> RunnableConfig:
        """Put a checkpoint into storage with proper metadata merging (async version)."""
        self.put(config, checkpoint, metadata, new_versions)
        return config

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str = "agent-protocol",
    ) -> None:
        """Put pending writes into storage (sync version)."""
        if not self._initialized:
            raise RuntimeError("Checkpoint storage not initialized")
        
        # If using fallback (self as saver), do nothing
        if self._saver is self:
            logger.debug(f"Fallback: Ignoring put_writes for task {task_id}")
            return
        
        self._saver.put_writes(config, writes, task_id)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Put pending writes into storage (async version)."""
        return self.put_writes(config, writes, task_id)

    async def delete(self, thread_id: str, run_id: Optional[str] = None) -> None:
        """Delete checkpoints for a thread or specific run."""
        if not self._initialized:
            raise RuntimeError("Checkpoint storage not initialized")
        self._saver.delete_thread(thread_id)

    async def copy(self, thread_id: str, new_thread_id: str) -> None:
        """Copy all checkpoints from one thread to another."""
        if not self._initialized:
            raise RuntimeError("Checkpoint storage not initialized")
        # Note: PostgresSaver doesn't have a copy method, so we'll skip this for now
        logger.warning("Copy operation not supported by PostgresSaver")

    async def close(self) -> None:
        """Close the checkpoint storage."""
        if self._context_manager is not None:
            # Exit the context manager
            self._context_manager.__exit__(None, None, None)
            self._context_manager = None
            self._saver = None
            self._initialized = False
            logger.info("Database checkpoint storage closed")

    def get_next_version(self, *args, **kwargs):
        if not self._initialized:
            raise RuntimeError("Checkpoint storage not initialized")
        
        # If using fallback (self as saver), provide minimal implementation
        if self._saver is self:
            return {"version": 1}
        
        return self._saver.get_next_version(*args, **kwargs)
    
    # Fallback methods when PostgreSQL checkpoint is not available
    def setup(self):
        """Fallback setup method."""
        logger.info("Using fallback checkpoint storage (no-op)")
        
    def delete_thread(self, thread_id: str):
        """Fallback delete_thread method."""
        logger.warning(f"Fallback: Cannot delete thread {thread_id} (no-op)")

# Global checkpoint saver instance
checkpointer = DatabaseCheckpointSaver()
