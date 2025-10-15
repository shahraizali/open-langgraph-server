"""PostgreSQL storage implementation for threads, runs, and assistants."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple
from uuid import UUID

from fastapi import HTTPException

logger = logging.getLogger(__name__)

try:
    from langgraph.store.base import BaseStore, Item, SearchItem
    HAS_LANGGRAPH_STORE = True
except ImportError as e:
    logger.warning(f"LangGraph store dependencies not available: {e}")
    # Provide minimal fallback types
    BaseStore = object
    Item = dict
    SearchItem = dict
    HAS_LANGGRAPH_STORE = False

from ..schemas import RunStatus, ThreadStatus
from .database import get_database, DatabaseQuery
from .schemas import RunKwargs


class PostgresThreadStorage:
    """PostgreSQL-based thread storage implementation."""

    def __init__(self):
        self.State = PostgresThreadStateManager(self)

    async def put(self, thread_id: str, options: Dict[str, Any]) -> "ThreadData":
        """Create or update a thread."""
        if_exists = options.get("if_exists", "raise")
        metadata = options.get("metadata", {})

        db = await get_database()

        async with db.transaction() as conn:
            # Check if thread exists (handle both UUID and string IDs)
            try:
                thread_uuid = UUID(thread_id)
            except ValueError:
                # If thread_id is not a valid UUID, generate one
                thread_uuid = uuid.uuid4()
                thread_id = str(thread_uuid)

            existing = await conn.fetchrow(
                "SELECT * FROM threads WHERE thread_id = $1", thread_uuid
            )

            if existing:
                if if_exists == "raise":
                    raise HTTPException(status_code=409, detail="Thread already exists")
                elif if_exists == "do_nothing":
                    return ThreadData.from_db_row(existing)

            # Insert or update thread
            if existing:
                # Update existing
                result = await conn.fetchrow(
                    """
                    UPDATE threads
                    SET metadata = $2, updated_at = NOW()
                    WHERE thread_id = $1
                    RETURNING *
                    """,
                    thread_uuid,
                    json.dumps(metadata),
                )
            else:
                # Insert new
                result = await conn.fetchrow(
                    """
                    INSERT INTO threads (thread_id, metadata, status)
                    VALUES ($1, $2, 'idle')
                    RETURNING *
                    """,
                    thread_uuid,
                    json.dumps(metadata),
                )

            return ThreadData.from_db_row(result)

    async def get(self, thread_id: str) -> "ThreadData":
        """Get a thread by ID."""
        db = await get_database()

        async with db.get_connection() as conn:
            try:
                thread_uuid = UUID(thread_id)
            except ValueError:
                raise HTTPException(
                    status_code=404, detail=f"Thread with ID {thread_id} not found"
                )

            result = await conn.fetchrow(
                "SELECT * FROM threads WHERE thread_id = $1", thread_uuid
            )

            if not result:
                raise HTTPException(
                    status_code=404, detail=f"Thread with ID {thread_id} not found"
                )

            return ThreadData.from_db_row(result)

    async def delete(self, thread_id: str) -> List[str]:
        """Delete a thread."""
        db = await get_database()

        async with db.transaction() as conn:
            result = await conn.fetchrow(
                "DELETE FROM threads WHERE thread_id = $1 RETURNING thread_id",
                UUID(thread_id),
            )

            if not result:
                raise HTTPException(
                    status_code=404, detail=f"Thread with ID {thread_id} not found"
                )

            return [str(result["thread_id"])]

    async def patch(self, thread_id: str, updates: Dict[str, Any]) -> "ThreadData":
        """Update a thread."""
        db = await get_database()

        async with db.transaction() as conn:
            # Get current thread
            current = await conn.fetchrow(
                "SELECT * FROM threads WHERE thread_id = $1", UUID(thread_id)
            )

            if not current:
                raise HTTPException(
                    status_code=404, detail=f"Thread with ID {thread_id} not found"
                )

            # Merge metadata if provided
            if "metadata" in updates:
                current_metadata = current["metadata"] if current["metadata"] else {}
                new_metadata = {**current_metadata, **updates["metadata"]}

                result = await conn.fetchrow(
                    """
                    UPDATE threads
                    SET metadata = $2, updated_at = NOW()
                    WHERE thread_id = $1
                    RETURNING *
                    """,
                    UUID(thread_id),
                    json.dumps(new_metadata),
                )
            else:
                result = current

            return ThreadData.from_db_row(result)

    async def update_metadata_with_assistant(
        self, thread_id: str, assistant_id: str, graph_id: str
    ) -> "ThreadData":
        """Update thread metadata with assistant_id and graph_id."""
        db = await get_database()

        async with db.transaction() as conn:
            # Get current thread
            current = await conn.fetchrow(
                "SELECT * FROM threads WHERE thread_id = $1", UUID(thread_id)
            )

            if not current:
                raise HTTPException(
                    status_code=404, detail=f"Thread with ID {thread_id} not found"
                )

            # Merge metadata with assistant information
            current_metadata = {}
            if current["metadata"]:
                if isinstance(current["metadata"], str):
                    import json

                    current_metadata = json.loads(current["metadata"])
                else:
                    current_metadata = current["metadata"]

            new_metadata = {
                **current_metadata,
                "assistant_id": assistant_id,
                "graph_id": graph_id,
            }

            result = await conn.fetchrow(
                """
                UPDATE threads
                SET metadata = $2, updated_at = NOW()
                WHERE thread_id = $1
                RETURNING *
                """,
                UUID(thread_id),
                json.dumps(new_metadata),
            )

            return ThreadData.from_db_row(result)

    async def search(
        self, filters: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Search threads with filters."""
        db = await get_database()

        # Build search query
        search_filters = {}
        if filters.get("metadata"):
            search_filters["metadata"] = filters["metadata"]
        if filters.get("status"):
            search_filters["status"] = filters["status"]
        if filters.get("values"):
            search_filters["values"] = filters["values"]

        query, params = DatabaseQuery.build_search_query(
            "threads",
            search_filters,
            limit=filters.get("limit", 10),
            offset=filters.get("offset", 0),
            order_by=filters.get("sort_by", "created_at"),
            order_direction=filters.get("sort_order", "desc").upper(),
        )

        async with db.get_connection() as conn:
            results = await conn.fetch(query, *params)

            total = len(results)
            if results:
                total = results[0]["total_count"]

            for row in results:
                yield {"thread": ThreadData.from_db_row(row), "total": total}

    async def copy(self, thread_id: str) -> "ThreadData":
        """Copy a thread."""
        db = await get_database()

        async with db.transaction() as conn:
            # Get original thread
            original = await conn.fetchrow(
                "SELECT * FROM threads WHERE thread_id = $1", UUID(thread_id)
            )

            if not original:
                raise HTTPException(
                    status_code=404, detail=f"Thread with ID {thread_id} not found"
                )

            # Create new thread
            new_thread_id = uuid.uuid4()
            copied_metadata = original["metadata"] if original["metadata"] else {}
            copied_metadata["copied_from"] = thread_id

            result = await conn.fetchrow(
                """
                INSERT INTO threads (thread_id, metadata, status, values, config, interrupts)
                VALUES ($1, $2, 'idle', $3, $4, $5)
                RETURNING *
                """,
                new_thread_id,
                json.dumps(copied_metadata),
                original["values"],
                original["config"],
                original.get("interrupts", {}),
            )

            # Copy thread states
            await conn.execute(
                """
                INSERT INTO thread_states (thread_id, checkpoint_id, values, next_steps, checkpoint, metadata, parent_checkpoint_id, tasks)
                SELECT $1, uuid_generate_v4(), values, next_steps, checkpoint, metadata, parent_checkpoint_id, tasks
                FROM thread_states WHERE thread_id = $2
                """,
                new_thread_id,
                UUID(thread_id),
            )

            return ThreadData.from_db_row(result)

    async def get_thread_states(
        self,
        thread_id: str,
        limit: int = 10,
        before: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Get thread states from the database table."""
        try:
            db = await get_database()

            # Build query
            query = """
                SELECT
                    state_id,
                    checkpoint_id,
                    created_at,
                    values,
                    next_steps,
                    checkpoint,
                    metadata,
                    parent_checkpoint_id,
                    tasks
                FROM thread_states
                WHERE thread_id = $1
            """
            params = [UUID(thread_id)]
            param_count = 1

            # Add before filter if provided
            if before:
                param_count += 1
                query += f" AND checkpoint_id < ${param_count}"
                params.append(UUID(before))

            # Add metadata filter if provided
            if metadata_filter:
                for key, value in metadata_filter.items():
                    param_count += 1
                    query += f" AND metadata->>'{key}' = ${param_count}"
                    params.append(str(value))

            # Add ordering and limit
            query += " ORDER BY created_at DESC LIMIT $2"
            params.append(limit)

            async with db.transaction() as conn:
                rows = await conn.fetch(query, *params)

                # Convert to API format
                states = []
                for row in rows:
                    state = {
                        "state_id": str(row["state_id"]),
                        "checkpoint_id": str(row["checkpoint_id"]),
                        "created_at": row["created_at"].isoformat() + "Z",
                        "values": json.loads(row["values"]) if row["values"] else {},
                        "next": (
                            json.loads(row["next_steps"]) if row["next_steps"] else []
                        ),
                        "checkpoint": (
                            json.loads(row["checkpoint"]) if row["checkpoint"] else None
                        ),
                        "metadata": (
                            json.loads(row["metadata"]) if row["metadata"] else {}
                        ),
                        "parent_checkpoint_id": (
                            str(row["parent_checkpoint_id"])
                            if row["parent_checkpoint_id"]
                            else None
                        ),
                        "tasks": json.loads(row["tasks"]) if row["tasks"] else [],
                    }
                    states.append(state)

                return states

        except Exception as e:
            logger.error(f"Failed to get thread states for thread {thread_id}: {e}")
            return []


class PostgresThreadStateManager:
    """PostgreSQL-based thread state management."""

    def __init__(self, thread_storage: PostgresThreadStorage):
        self.thread_storage = thread_storage

    async def get(
        self, config: Dict[str, Any], options: Optional[Dict[str, Any]] = None
    ) -> "ThreadStateData":
        """Get thread state."""
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise HTTPException(status_code=400, detail="thread_id required in config")

        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        db = await get_database()

        async with db.get_connection() as conn:
            if checkpoint_id:
                # Get specific checkpoint
                result = await conn.fetchrow(
                    """
                    SELECT * FROM thread_states
                    WHERE thread_id = $1 AND checkpoint_id = $2
                    """,
                    UUID(thread_id),
                    UUID(checkpoint_id),
                )

                if not result:
                    raise HTTPException(status_code=404, detail="Checkpoint not found")

                return ThreadStateData.from_db_row(result)
            else:
                # Get latest state
                result = await conn.fetchrow(
                    """
                    SELECT * FROM thread_states
                    WHERE thread_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    UUID(thread_id),
                )

                if result:
                    return ThreadStateData.from_db_row(result)

                # Create default state
                checkpoint_id = uuid.uuid4()
                default_result = await conn.fetchrow(
                    """
                    INSERT INTO thread_states (thread_id, checkpoint_id, values, next_steps)
                    VALUES ($1, $2, '{}', '[]')
                    RETURNING *
                    """,
                    UUID(thread_id),
                    checkpoint_id,
                )

                return ThreadStateData.from_db_row(default_result)

    async def post(
        self,
        config: Dict[str, Any],
        values: Optional[Any] = None,
        as_node: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update thread state."""
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise HTTPException(status_code=400, detail="thread_id required in config")

        db = await get_database()
        checkpoint_id = uuid.uuid4()

        async with db.transaction() as conn:
            # Insert new state
            await conn.execute(
                """
                INSERT INTO thread_states (thread_id, checkpoint_id, values, next_steps)
                VALUES ($1, $2, $3, '[]')
                """,
                UUID(thread_id),
                checkpoint_id,
                json.dumps(values or {}),
            )

            # Update thread values
            await conn.execute(
                """
                UPDATE threads
                SET values = $2, updated_at = NOW()
                WHERE thread_id = $1
                """,
                UUID(thread_id),
                json.dumps(values or {}),
            )

            return {"checkpoint": {"checkpoint_id": str(checkpoint_id)}}

    async def list(
        self, config: Dict[str, Any], options: Optional[Dict[str, Any]] = None
    ) -> List["ThreadStateData"]:
        """List thread states (history)."""
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise HTTPException(status_code=400, detail="thread_id required in config")

        limit = (options or {}).get("limit", 10)

        db = await get_database()

        async with db.get_connection() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM thread_states
                WHERE thread_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                UUID(thread_id),
                limit,
            )

            return [ThreadStateData.from_db_row(row) for row in results]

    async def bulk(
        self, config: Dict[str, Any], supersteps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Bulk update thread state."""
        for superstep in supersteps:
            for update in superstep.get("updates", []):
                await self.post(config, update.get("values"), update.get("as_node"))

        return {"success": True}


class PostgresRunStorage:
    """PostgreSQL-based run storage implementation."""

    async def put(
        self,
        run_id: str,
        assistant_id: str,
        kwargs: "RunKwargs",
        options: Dict[str, Any],
        auth: Optional[Any] = None,
    ) -> tuple[Optional["RunData"], List["RunData"]]:
        """Create a new run."""
        db = await get_database()

        async with db.transaction() as conn:
            # Handle thread creation if needed
            thread_id = options.get("thread_id")
            if_not_exists = options.get("if_not_exists", "reject")

            if thread_id and if_not_exists == "create":
                # Convert thread_id to UUID
                try:
                    thread_uuid = UUID(thread_id)
                except ValueError:
                    thread_uuid = uuid.uuid4()
                    thread_id = str(thread_uuid)

                # Ensure thread exists
                thread_exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM threads WHERE thread_id = $1)",
                    thread_uuid,
                )

                if not thread_exists:
                    await conn.execute(
                        """
                        INSERT INTO threads (thread_id, metadata, status)
                        VALUES ($1, $2, 'busy')
                        """,
                        thread_uuid,
                        json.dumps(options.get("metadata", {})),
                    )

            # Convert thread_id to UUID for queries
            thread_uuid = None
            if thread_id:
                try:
                    thread_uuid = UUID(thread_id)
                except ValueError:
                    raise HTTPException(
                        status_code=400, detail="Invalid thread_id format"
                    )

            # Check for inflight runs
            inflight_runs = []
            if thread_uuid:
                inflight_results = await conn.fetch(
                    """
                    SELECT * FROM runs
                    WHERE thread_id = $1 AND status IN ('pending', 'running')
                    """,
                    thread_uuid,
                )
                inflight_runs = [RunData.from_db_row(row) for row in inflight_results]

            if options.get("prevent_insert_if_inflight") and inflight_runs:
                return None, inflight_runs

            # Convert run_id to UUID
            try:
                run_uuid = UUID(run_id)
            except ValueError:
                run_uuid = uuid.uuid4()
                run_id = str(run_uuid)

            # Serialize kwargs properly
            kwargs_dict = {}
            if hasattr(kwargs, "__dict__"):
                kwargs_dict = kwargs.__dict__
            else:
                # Handle the case where kwargs might be a dict or other type
                kwargs_dict = dict(kwargs) if kwargs else {}

            # Create run
            result = await conn.fetchrow(
                """
                INSERT INTO runs (
                    run_id, thread_id, assistant_id, status, metadata,
                    kwargs, multitask_strategy
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
                """,
                run_uuid,
                thread_uuid,
                assistant_id,
                options.get("status", "pending"),
                json.dumps(options.get("metadata", {})),
                json.dumps(kwargs_dict),
                options.get("multitask_strategy", "reject"),
            )

            return RunData.from_db_row(result), inflight_runs

    async def get(
        self, run_id: str, thread_id: Optional[str], auth: Optional[Any] = None
    ) -> Optional["RunData"]:
        """Get a run by ID."""
        db = await get_database()

        async with db.get_connection() as conn:
            query = "SELECT * FROM runs WHERE run_id = $1"
            params = [UUID(run_id)]

            if thread_id is not None:
                query += " AND thread_id = $2"
                params.append(UUID(thread_id))

            result = await conn.fetchrow(query, *params)

            if not result:
                return None

            return RunData.from_db_row(result)

    async def delete(
        self, run_id: str, thread_id: Optional[str], auth: Optional[Any] = None
    ) -> Optional[str]:
        """Delete a run."""
        db = await get_database()

        async with db.transaction() as conn:
            query = "DELETE FROM runs WHERE run_id = $1"
            params = [UUID(run_id)]

            if thread_id is not None:
                query += " AND thread_id = $2"
                params.append(UUID(thread_id))

            query += " RETURNING run_id"

            result = await conn.fetchrow(query, *params)

            if not result:
                raise HTTPException(status_code=404, detail="Run not found")

            return str(result["run_id"])

    async def search(
        self, thread_id: str, options: Dict[str, Any], auth: Optional[Any] = None
    ) -> List["RunData"]:
        """Search runs in a thread."""
        db = await get_database()

        filters = {"thread_id": UUID(thread_id)}
        if options.get("status"):
            filters["status"] = options["status"]
        if options.get("metadata"):
            filters["metadata"] = options["metadata"]

        query, params = DatabaseQuery.build_search_query(
            "runs",
            filters,
            limit=options.get("limit", 10),
            offset=options.get("offset", 0),
            order_by="created_at",
            order_direction="DESC",
        )

        async with db.get_connection() as conn:
            results = await conn.fetch(query, *params)
            return [RunData.from_db_row(row) for row in results]


class PostgresAssistantStorage:
    """PostgreSQL-based assistant storage implementation."""

    async def put(self, assistant_id: str, data: Dict[str, Any]) -> "Assistant":
        """Create or update an assistant."""
        db = await get_database()

        async with db.transaction() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO assistants (
                    assistant_id, name, graph_id, version, config, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (assistant_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    graph_id = EXCLUDED.graph_id,
                    version = EXCLUDED.version,
                    config = EXCLUDED.config,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING *
                """,
                assistant_id,
                data.get("name"),
                data["graph_id"],
                data.get("version", 1),
                json.dumps(data.get("config", {})),
                json.dumps(data.get("metadata", {})),
            )

            return Assistant.from_db_row(result)

    async def get(self, assistant_id: str) -> Optional["Assistant"]:
        """Get an assistant by ID."""
        db = await get_database()

        async with db.get_connection() as conn:
            # Assistant IDs are stored as VARCHAR, not UUID
            result = await conn.fetchrow(
                "SELECT * FROM assistants WHERE assistant_id = $1", assistant_id
            )

            return Assistant.from_db_row(result)

    async def search(
        self, filters: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Search assistants with filters."""
        db = await get_database()

        # Build search query
        search_filters = {}
        if filters.get("graph_id"):
            search_filters["graph_id"] = filters["graph_id"]
        if filters.get("metadata"):
            search_filters["metadata"] = filters["metadata"]

        query, params = DatabaseQuery.build_search_query(
            "assistants",
            search_filters,
            limit=filters.get("limit", 10),
            offset=filters.get("offset", 0),
            order_by="created_at",
            order_direction="DESC",
        )

        async with db.get_connection() as conn:
            results = await conn.fetch(query, *params)

            total = len(results)
            if results:
                total = results[0]["total_count"]

            for row in results:
                assistant = Assistant.from_db_row(row)
                # Convert to dictionary format matching JavaScript implementation
                # JavaScript: assistant.name ?? assistant.graph_id
                assistant_dict = {
                    "assistant_id": assistant.assistant_id,
                    "graph_id": assistant.graph_id,
                    "config": assistant.config,
                    "created_at": (
                        assistant.created_at.isoformat() + "Z"
                        if hasattr(assistant.created_at, "isoformat")
                        else str(assistant.created_at)
                    ),
                    "updated_at": (
                        assistant.updated_at.isoformat() + "Z"
                        if hasattr(assistant.updated_at, "isoformat")
                        else str(assistant.updated_at)
                    ),
                    "metadata": assistant.metadata,
                    "name": assistant.name or assistant.graph_id,
                    "version": assistant.version,
                }
                yield {"assistant": assistant_dict, "total": total}

    async def delete(self, assistant_id: str) -> List[str]:
        """Delete an assistant."""
        db = await get_database()

        async with db.transaction() as conn:
            result = await conn.fetchrow(
                "DELETE FROM assistants WHERE assistant_id = $1 RETURNING assistant_id",
                assistant_id,
            )

            if not result:
                raise HTTPException(status_code=404, detail="Assistant not found")

            return [str(result["assistant_id"])]

    async def patch(self, assistant_id: str, updates: Dict[str, Any]) -> "Assistant":
        """Update an assistant."""
        db = await get_database()

        async with db.transaction() as conn:
            # Get current assistant
            current = await conn.fetchrow(
                "SELECT * FROM assistants WHERE assistant_id = $1", assistant_id
            )

            if not current:
                raise HTTPException(status_code=404, detail="Assistant not found")

            # Build update query
            set_clauses = []
            params = []
            param_counter = 1

            if "name" in updates:
                set_clauses.append(f"name = ${param_counter}")
                params.append(updates["name"])
                param_counter += 1

            if "graph_id" in updates:
                set_clauses.append(f"graph_id = ${param_counter}")
                params.append(updates["graph_id"])
                param_counter += 1

            if "config" in updates:
                set_clauses.append(f"config = ${param_counter}")
                params.append(json.dumps(updates["config"]))
                param_counter += 1

            if "metadata" in updates:
                # Merge metadata
                current_metadata = current["metadata"] if current["metadata"] else {}
                new_metadata = {**current_metadata, **updates["metadata"]}
                set_clauses.append(f"metadata = ${param_counter}")
                params.append(json.dumps(new_metadata))
                param_counter += 1

            if not set_clauses:
                return Assistant.from_db_row(current)

            # Add updated_at
            set_clauses.append("updated_at = NOW()")

            # Add assistant_id parameter
            params.append(assistant_id)

            result = await conn.fetchrow(
                f"""
                UPDATE assistants
                SET {", ".join(set_clauses)}
                WHERE assistant_id = ${param_counter}
                RETURNING *
                """,
                *params,
            )

            return Assistant.from_db_row(result)

    async def get_versions(
        self, assistant_id: str, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get assistant versions (simplified - just return current version)."""
        assistant = await self.get(assistant_id)
        if not assistant:
            return []

        return [
            {
                "assistant_id": assistant.assistant_id,
                "graph_id": assistant.graph_id,
                "name": assistant.name,
                "version": assistant.version,
                "config": assistant.config,
                "metadata": assistant.metadata,
                "created_at": assistant.created_at.isoformat() + "Z",
                "updated_at": assistant.updated_at.isoformat() + "Z",
            }
        ]

    async def set_latest(self, assistant_id: str, version: int) -> "Assistant":
        """Set latest version (simplified - just update version)."""
        return await self.patch(assistant_id, {"version": version})


# Data classes matching the interface expected by ops.py
class ThreadData:
    """Thread data class that matches the API models."""

    def __init__(
        self,
        thread_id: str,
        created_at: str,
        updated_at: str,
        metadata: Dict[str, Any],
        status: ThreadStatus,
        values: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        interrupts: Optional[Dict[str, Any]] = None,
    ):
        self.thread_id = thread_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.metadata = metadata
        self.status = status
        self.values = values
        self.config = config or {}
        self.interrupts = interrupts or {}

    @classmethod
    def from_db_row(cls, row) -> Optional["ThreadData"]:
        """Create ThreadData from database row."""
        if row is None:
            return None
            
        # Handle datetime properly
        created_at = row["created_at"]
        if hasattr(created_at, "isoformat"):
            created_at_str = created_at.isoformat()
            if not created_at_str.endswith("Z") and "+" not in created_at_str:
                created_at_str += "Z"
        else:
            created_at_str = str(created_at)

        updated_at = row["updated_at"]
        if hasattr(updated_at, "isoformat"):
            updated_at_str = updated_at.isoformat()
            if not updated_at_str.endswith("Z") and "+" not in updated_at_str:
                updated_at_str += "Z"
        else:
            updated_at_str = str(updated_at)

        # Handle metadata - it might be a JSON string or dict
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}
        elif metadata is None:
            metadata = {}

        # Handle values similarly
        values = row["values"]
        if isinstance(values, str):
            values = json.loads(values) if values else None

        # Handle config similarly
        config = row.get("config", {})
        if isinstance(config, str):
            config = json.loads(config) if config else {}
        elif config is None:
            config = {}

        # Handle interrupts similarly
        interrupts = row.get("interrupts", {})
        if isinstance(interrupts, str):
            interrupts = json.loads(interrupts) if interrupts else {}
        elif interrupts is None:
            interrupts = {}

        return cls(
            thread_id=str(row["thread_id"]),
            created_at=created_at_str,
            updated_at=updated_at_str,
            metadata=metadata,
            status=ThreadStatus(row["status"]),
            values=values,
            config=config,
            interrupts=interrupts,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "thread_id": self.thread_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "status": (
                self.status.value
                if isinstance(self.status, ThreadStatus)
                else self.status
            ),
            "values": self.values,
            "config": self.config,
            "interrupts": self.interrupts,
        }


class ThreadStateData:
    """Thread state data class."""

    def __init__(
        self,
        values: Dict[str, Any],
        next: List[str],
        checkpoint: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        parent_checkpoint: Optional[Dict[str, Any]] = None,
        tasks: Optional[List[Dict[str, Any]]] = None,
    ):
        self.values = values
        self.next = next
        self.checkpoint = checkpoint
        self.metadata = metadata
        self.created_at = created_at
        self.parent_checkpoint = parent_checkpoint
        self.tasks = tasks or []

    @classmethod
    def from_db_row(cls, row) -> Optional["ThreadStateData"]:
        """Create ThreadStateData from database row."""
        if row is None:
            return None
        return cls(
            values=row["values"] if row["values"] else {},
            next=row["next_steps"] if row["next_steps"] else [],
            checkpoint=(
                row["checkpoint"]
                if row["checkpoint"]
                else {"checkpoint_id": str(row["checkpoint_id"])}
            ),
            metadata=row["metadata"] if row["metadata"] else {},
            created_at=row["created_at"].isoformat() + "Z",
            parent_checkpoint=None,
            tasks=row["tasks"] if row["tasks"] else [],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "values": self.values,
            "next": self.next,
            "checkpoint": self.checkpoint,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "parent_checkpoint": self.parent_checkpoint,
            "tasks": self.tasks,
        }


class RunData:
    """Run data class."""

    def __init__(
        self,
        run_id: str,
        thread_id: str,
        assistant_id: str,
        created_at: datetime,
        updated_at: datetime,
        status: RunStatus,
        metadata: Dict[str, Any],
        kwargs: "RunKwargs",
        multitask_strategy: str,
    ):
        self.run_id = run_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.status = status
        self.metadata = metadata
        self.kwargs = kwargs
        self.multitask_strategy = multitask_strategy

    @classmethod
    def from_db_row(cls, row) -> Optional["RunData"]:
        """Create RunData from database row."""
        if row is None:
            return None
        from .schemas import RunKwargs  # Import here to avoid circular imports

        # Handle kwargs - it might be a JSON string or dict
        kwargs_data = row["kwargs"]
        logger.info(
            f"[CONFIG DEBUG] Raw kwargs from database: {kwargs_data} (type: {type(kwargs_data)})"
        )

        if isinstance(kwargs_data, str):
            try:
                kwargs_data = json.loads(kwargs_data) if kwargs_data else {}
                logger.info(
                    f"[CONFIG DEBUG] Parsed kwargs from JSON string: {kwargs_data}"
                )
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"[CONFIG DEBUG] Failed to parse kwargs JSON: {e}")
                kwargs_data = {}
        elif kwargs_data is None:
            logger.info("[CONFIG DEBUG] kwargs_data was None, using empty dict")
            kwargs_data = {}

        # Ensure kwargs_data is a dictionary and has proper structure
        if not isinstance(kwargs_data, dict):
            logger.warning(
                f"[CONFIG DEBUG] kwargs_data is not a dict, resetting. Was: {type(kwargs_data)}"
            )
            kwargs_data = {}

        logger.info(
            f"[CONFIG DEBUG] kwargs_data before config normalization: {kwargs_data}"
        )

        # Ensure config field exists and is properly structured
        if "config" not in kwargs_data or kwargs_data["config"] is None:
            logger.info("[CONFIG DEBUG] Config missing or None, setting default")
            kwargs_data["config"] = {"configurable": {}}
        elif isinstance(kwargs_data["config"], dict):
            logger.info(f"[CONFIG DEBUG] Config is dict: {kwargs_data['config']}")
            # Ensure configurable key exists
            if (
                "configurable" not in kwargs_data["config"]
                or kwargs_data["config"]["configurable"] is None
            ):
                logger.info(
                    "[CONFIG DEBUG] Configurable missing or None, setting empty dict"
                )
                kwargs_data["config"]["configurable"] = {}
            elif isinstance(kwargs_data["config"]["configurable"], dict):
                # Filter out None values from configurable during deserialization
                original_configurable = kwargs_data["config"]["configurable"]
                filtered_configurable = {
                    k: v for k, v in original_configurable.items() if v is not None
                }
                kwargs_data["config"]["configurable"] = filtered_configurable
                logger.info(
                    f"[CONFIG DEBUG] Filtered None values from configurable. Original: {original_configurable}, Filtered: {filtered_configurable}"
                )
            else:
                logger.warning(
                    f"[CONFIG DEBUG] Configurable is not a dict: {type(kwargs_data['config']['configurable'])}"
                )
                kwargs_data["config"]["configurable"] = {}
        else:
            # Invalid config type, reset to default
            logger.warning(
                f"[CONFIG DEBUG] Config is invalid type {type(kwargs_data['config'])}, resetting"
            )
            kwargs_data["config"] = {"configurable": {}}

        logger.info(
            f"[CONFIG DEBUG] Final kwargs_data before RunKwargs creation: {kwargs_data}"
        )
        kwargs = RunKwargs(**kwargs_data)
        logger.info(f"[CONFIG DEBUG] RunKwargs.config after creation: {kwargs.config}")

        # Handle metadata - it might be a JSON string or dict
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}
        elif metadata is None:
            metadata = {}

        return cls(
            run_id=str(row["run_id"]),
            thread_id=str(row["thread_id"]) if row["thread_id"] else "",
            assistant_id=row["assistant_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=RunStatus(row["status"]),
            metadata=metadata,
            kwargs=kwargs,
            multitask_strategy=row["multitask_strategy"],
        )


class Assistant:
    """Assistant data class."""

    def __init__(
        self,
        assistant_id: str,
        name: Optional[str],
        graph_id: str,
        created_at: datetime,
        updated_at: datetime,
        version: int,
        config: Dict[str, Any],
        metadata: Dict[str, Any],
    ):
        self.assistant_id = assistant_id
        self.name = name
        self.graph_id = graph_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.version = version
        self.config = config
        self.metadata = metadata

    @classmethod
    def from_db_row(cls, row) -> Optional["Assistant"]:
        """Create Assistant from database row."""
        try:
            logger.debug(f"from_db_row called with row type: {type(row)}, value: {row}")
            if row is None:
                logger.debug("Row is None, returning None")
                return None
                
            # Handle config - it might be a JSON string or dict
            logger.debug(f"Accessing row['config']")
            config = row["config"]
            if isinstance(config, str):
                config = json.loads(config) if config else {}
            elif config is None:
                config = {}

            # Handle metadata - it might be a JSON string or dict
            logger.debug(f"Accessing row['metadata']")
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata) if metadata else {}
            elif metadata is None:
                metadata = {}

            logger.debug(f"Creating Assistant instance")
            return cls(
                assistant_id=row["assistant_id"],
                name=row["name"],
                graph_id=row["graph_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                version=row["version"],
                config=config,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Error in from_db_row: {e}, row type: {type(row)}, row: {row}")
            return None


class PostgresStoreStorage:
    """PostgreSQL-based store implementation for key-value storage with namespacing."""

    def _namespace_to_string(self, namespace: Optional[List[str]]) -> str:
        """Convert namespace array to string representation, filtering out None values."""
        if namespace:
            # Filter out None values and convert to strings
            clean_namespace = [str(n) for n in namespace if n is not None]
            return ".".join(clean_namespace)
        else:
            return ""

    async def put(self, namespace: List[str], key: str, value: Dict[str, Any]) -> None:
        """Store an item in the specified namespace."""
        db = await get_database()

        # Convert namespace array to string representation
        namespace_str = self._namespace_to_string(namespace)

        async with db.transaction() as conn:
            # Use UPSERT to insert or update
            await conn.execute(
                """
                INSERT INTO store_items (namespace, key, value)
                VALUES ($1, $2, $3)
                ON CONFLICT (namespace, key)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = NOW()
                """,
                namespace_str,
                key,
                json.dumps(value),
            )

    async def get(self, namespace: List[str], key: str) -> Optional[Dict[str, Any]]:
        """Get an item by namespace and key."""
        db = await get_database()

        # Convert namespace array to string representation
        namespace_str = self._namespace_to_string(namespace)

        async with db.get_connection() as conn:
            result = await conn.fetchrow(
                "SELECT * FROM store_items WHERE namespace = $1 AND key = $2",
                namespace_str,
                key,
            )

            if not result:
                return None

            return StoreItem.from_db_row(result)

    async def delete(self, namespace: List[str], key: str) -> bool:
        """Delete an item from the specified namespace."""
        db = await get_database()

        # Convert namespace array to string representation
        namespace_str = self._namespace_to_string(namespace)

        async with db.transaction() as conn:
            result = await conn.fetchrow(
                "DELETE FROM store_items WHERE namespace = $1 AND key = $2 RETURNING item_id",
                namespace_str,
                key,
            )

            return result is not None

    async def search(
        self, namespace_prefix: Optional[List[str]], options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search for items within a namespace with optional filtering."""
        db = await get_database()

        # Build the base query
        query_parts = ["SELECT * FROM store_items"]
        params = []
        param_counter = 1
        where_conditions = []

        # Handle namespace prefix filtering
        if namespace_prefix:
            namespace_prefix_str = self._namespace_to_string(namespace_prefix)
            where_conditions.append(f"namespace LIKE ${param_counter}")
            params.append(f"{namespace_prefix_str}.%")
            param_counter += 1

        # Handle filter (JSONB filtering)
        if options.get("filter"):
            filter_value = options["filter"]
            where_conditions.append(f"value @> ${param_counter}")
            params.append(json.dumps(filter_value))
            param_counter += 1

        # Handle query (full-text search on value)
        if options.get("query"):
            query_text = options["query"]
            where_conditions.append(f"value::text ILIKE ${param_counter}")
            params.append(f"%{query_text}%")
            param_counter += 1

        # Build WHERE clause
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))

        # Add ordering
        query_parts.append("ORDER BY created_at DESC")

        # Handle pagination
        limit = options.get("limit", 10)
        offset = options.get("offset", 0)

        query_parts.append(f"LIMIT ${param_counter}")
        params.append(limit)
        param_counter += 1

        query_parts.append(f"OFFSET ${param_counter}")
        params.append(offset)

        query = " ".join(query_parts)

        async with db.get_connection() as conn:
            results = await conn.fetch(query, *params)

            return [StoreItem.from_db_row(row) for row in results]

    async def list_namespaces(self, options: Dict[str, Any]) -> List[List[str]]:
        """List available namespaces with optional filtering."""
        db = await get_database()

        # Build the base query to get distinct namespaces
        query_parts = ["SELECT DISTINCT namespace FROM store_items"]
        params = []
        param_counter = 1
        where_conditions = []

        # Handle prefix filtering
        if options.get("prefix"):
            prefix = self._namespace_to_string(options["prefix"])
            where_conditions.append(f"namespace LIKE ${param_counter}")
            params.append(f"{prefix}.%")
            param_counter += 1

        # Handle suffix filtering
        if options.get("suffix"):
            suffix = self._namespace_to_string(options["suffix"])
            where_conditions.append(f"namespace LIKE ${param_counter}")
            params.append(f"%.{suffix}")
            param_counter += 1

        # Build WHERE clause
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))

        # Add ordering
        query_parts.append("ORDER BY namespace")

        # Handle pagination
        limit = options.get("limit", 100)
        offset = options.get("offset", 0)

        query_parts.append(f"LIMIT ${param_counter}")
        params.append(limit)
        param_counter += 1

        query_parts.append(f"OFFSET ${param_counter}")
        params.append(offset)

        query = " ".join(query_parts)

        async with db.get_connection() as conn:
            results = await conn.fetch(query, *params)

            # Convert namespace strings back to arrays and filter by max_depth
            namespaces = []
            max_depth = options.get("max_depth")

            for row in results:
                namespace_str = row["namespace"]
                if namespace_str:
                    namespace_array = namespace_str.split(".")
                    # Apply max_depth filtering if specified
                    if max_depth is None or len(namespace_array) <= max_depth:
                        namespaces.append(namespace_array)
                else:
                    # Empty namespace
                    if max_depth is None or max_depth >= 0:
                        namespaces.append([])

            return namespaces


class PostgresStore(BaseStore):
    """
    LangGraph BaseStore implementation backed by PostgreSQL.

    This provides the LangGraph store interface that graphs expect,
    while using our PostgresStoreStorage for the actual database operations.
    """

    def __init__(self, storage: PostgresStoreStorage):
        self.storage = storage

    async def aget(
        self,
        namespace: Tuple[str, ...],
        key: str,
        *,
        refresh_ttl: Optional[bool] = None,
    ) -> Optional[Item]:
        """Get an item by namespace and key."""
        # Convert tuple to list for our storage
        namespace_list = list(namespace)

        result = await self.storage.get(namespace_list, key)

        if result is None:
            return None

        # Convert to LangGraph Item format
        return Item(
            namespace=namespace,
            key=key,
            value=result["value"],
            created_at=datetime.fromisoformat(
                result["created_at"].replace("Z", "+00:00")
            ),
            updated_at=datetime.fromisoformat(
                result["updated_at"].replace("Z", "+00:00")
            ),
        )

    async def aput(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Dict[str, Any],
        index: Optional[List[str]] = None,
        *,
        ttl: Optional[float] = None,
    ) -> None:
        """Store an item in the specified namespace."""
        # Convert tuple to list for our storage
        namespace_list = list(namespace)

        await self.storage.put(namespace_list, key, value)

    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Delete an item from the specified namespace."""
        # Convert tuple to list for our storage
        namespace_list = list(namespace)

        success = await self.storage.delete(namespace_list, key)

        # LangGraph delete doesn't return anything, but our storage returns bool
        # We could raise an exception if not found, but LangGraph spec doesn't require it

    async def asearch(
        self,
        namespace_prefix: Tuple[str, ...],
        /,
        *,
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
    ) -> List[SearchItem]:
        """Search for items within a namespace prefix."""
        # Convert tuple to list for our storage
        namespace_prefix_list = list(namespace_prefix) if namespace_prefix else None

        search_options = {
            "query": query,
            "filter": filter,
            "limit": limit,
            "offset": offset,
        }

        results = await self.storage.search(namespace_prefix_list, search_options)

        # Convert to LangGraph SearchItem format
        search_items = []
        for result in results:
            search_item = SearchItem(
                namespace=tuple(result["namespace"]),
                key=result["key"],
                value=result["value"],
                created_at=datetime.fromisoformat(
                    result["created_at"].replace("Z", "+00:00")
                ),
                updated_at=datetime.fromisoformat(
                    result["updated_at"].replace("Z", "+00:00")
                ),
                score=None,  # We don't have scoring in our implementation yet
            )
            search_items.append(search_item)

        return search_items

    async def alist_namespaces(
        self,
        *,
        prefix: Optional[Tuple[str, ...]] = None,
        suffix: Optional[Tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Tuple[str, ...]]:
        """List available namespaces with optional filtering."""
        list_options = {
            "prefix": list(prefix) if prefix else None,
            "suffix": list(suffix) if suffix else None,
            "max_depth": max_depth,
            "limit": limit,
            "offset": offset,
        }

        namespaces_list = await self.storage.list_namespaces(list_options)

        # Convert list of lists to list of tuples
        return [tuple(ns) for ns in namespaces_list]

    async def abatch(self, ops) -> List[Any]:
        """Batch operations - not implemented yet."""
        # For now, execute operations sequentially
        # This could be optimized later with actual batch operations
        results = []
        for op in ops:
            if op.type == "get":
                result = await self.aget(op.namespace, op.key)
                results.append(result)
            elif op.type == "put":
                await self.aput(op.namespace, op.key, op.value)
                results.append(None)
            elif op.type == "delete":
                await self.adelete(op.namespace, op.key)
                results.append(None)
            elif op.type == "search":
                result = await self.asearch(
                    op.namespace_prefix,
                    query=getattr(op, "query", None),
                    filter=getattr(op, "filter", None),
                    limit=getattr(op, "limit", 10),
                    offset=getattr(op, "offset", 0),
                )
                results.append(result)
            else:
                raise ValueError(f"Unknown operation type: {op.type}")

        return results

    def batch(self, ops) -> List[Any]:
        """Synchronous batch operations - not implemented yet."""
        # For BaseStore interface compatibility, we need this method
        # But we'll delegate to async version
        import asyncio

        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, can't use asyncio.run()
            # Create a new task
            return asyncio.create_task(self.abatch(ops))
        except RuntimeError:
            # No running loop, we can use asyncio.run()
            return asyncio.run(self.abatch(ops))

    # Synchronous versions for BaseStore interface compatibility
    def get(self, namespace: Tuple[str, ...], key: str, **kwargs):
        """Synchronous get method."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            return asyncio.create_task(self.aget(namespace, key, **kwargs))
        except RuntimeError:
            return asyncio.run(self.aget(namespace, key, **kwargs))

    def put(
        self, namespace: Tuple[str, ...], key: str, value: Dict[str, Any], **kwargs
    ):
        """Synchronous put method."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            return asyncio.create_task(self.aput(namespace, key, value, **kwargs))
        except RuntimeError:
            return asyncio.run(self.aput(namespace, key, value, **kwargs))

    def delete(self, namespace: Tuple[str, ...], key: str):
        """Synchronous delete method."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            return asyncio.create_task(self.adelete(namespace, key))
        except RuntimeError:
            return asyncio.run(self.adelete(namespace, key))

    def search(self, namespace_prefix: Tuple[str, ...], /, **kwargs):
        """Synchronous search method."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            return asyncio.create_task(self.asearch(namespace_prefix, **kwargs))
        except RuntimeError:
            return asyncio.run(self.asearch(namespace_prefix, **kwargs))

    def list_namespaces(self, **kwargs):
        """Synchronous list_namespaces method."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            return asyncio.create_task(self.alist_namespaces(**kwargs))
        except RuntimeError:
            return asyncio.run(self.alist_namespaces(**kwargs))


class StoreItem:
    """Store item data class."""

    def __init__(
        self,
        namespace: List[str],
        key: str,
        value: Dict[str, Any],
        created_at: str,
        updated_at: str,
    ):
        self.namespace = namespace
        self.key = key
        self.value = value
        self.created_at = created_at
        self.updated_at = updated_at

    @classmethod
    def from_db_row(cls, row) -> Dict[str, Any]:
        """Create StoreItem from database row and return as API format dict."""
        # Handle datetime properly
        created_at = row["created_at"]
        if hasattr(created_at, "isoformat"):
            created_at_str = created_at.isoformat()
            if not created_at_str.endswith("Z") and "+" not in created_at_str:
                created_at_str += "Z"
        else:
            created_at_str = str(created_at)

        updated_at = row["updated_at"]
        if hasattr(updated_at, "isoformat"):
            updated_at_str = updated_at.isoformat()
            if not updated_at_str.endswith("Z") and "+" not in updated_at_str:
                updated_at_str += "Z"
        else:
            updated_at_str = str(updated_at)

        # Convert namespace string back to array
        namespace_str = row["namespace"]
        namespace = namespace_str.split(".") if namespace_str else []

        # Handle value - it might be a JSON string or dict
        value = row["value"]
        if isinstance(value, str):
            value = json.loads(value) if value else {}
        elif value is None:
            value = {}

        # Return in API format (matching JavaScript mapItemsToApi)
        return {
            "namespace": namespace,
            "key": row["key"],
            "value": value,
            "created_at": created_at_str,
            "updated_at": updated_at_str,
        }


# Global instances
postgres_thread_storage = PostgresThreadStorage()
postgres_run_storage = PostgresRunStorage()
postgres_assistant_storage = PostgresAssistantStorage()
postgres_store_storage = PostgresStoreStorage()

# LangGraph BaseStore implementation
postgres_store = PostgresStore(postgres_store_storage)
