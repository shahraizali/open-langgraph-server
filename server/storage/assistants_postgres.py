"""PostgreSQL-based assistant storage implementation."""

from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import BaseStorage, ItemNotFoundError, ItemExistsError, get_timestamp
from .database import get_database, DatabaseQuery

logger = logging.getLogger(__name__)


class Assistant:
    """Assistant data model for PostgreSQL storage."""

    def __init__(
        self,
        assistant_id: str,
        graph_id: str,
        name: str = "Untitled",
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        version: int = 1,
    ):
        self.assistant_id = assistant_id
        self.graph_id = graph_id
        self.name = name
        self.config = config or {}
        self.metadata = metadata or {}
        self.created_at = created_at or get_timestamp()
        self.updated_at = updated_at or get_timestamp()
        self.version = version

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "assistant_id": self.assistant_id,
            "graph_id": self.graph_id,
            "name": self.name,
            "config": self.config,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    @classmethod
    def from_row(cls, row: Any) -> Assistant:
        """Create from database row."""
        return cls(
            assistant_id=str(row["assistant_id"]),
            graph_id=row["graph_id"],
            name=row["name"],
            config=row["config"] or {},
            metadata=row["metadata"] or {},
            created_at=(
                row["created_at"].isoformat() + "Z" if row["created_at"] else None
            ),
            updated_at=(
                row["updated_at"].isoformat() + "Z" if row["updated_at"] else None
            ),
            version=row["version"],
        )


class PostgreSQLAssistantStorage(BaseStorage[Assistant]):
    """PostgreSQL implementation of assistant storage."""

    async def get(
        self, assistant_id: str, auth_context: Optional[Any] = None
    ) -> Assistant:
        """Get an assistant by ID."""
        db = await get_database()

        async with db.get_connection() as conn:
            query = "SELECT * FROM assistants WHERE assistant_id = $1"
            row = await conn.fetchrow(query, uuid.UUID(assistant_id))

            if not row:
                raise ItemNotFoundError("Assistant", assistant_id)

            return Assistant.from_row(row)

    async def put(
        self,
        assistant_id: str,
        data: Dict[str, Any],
        auth_context: Optional[Any] = None,
    ) -> Assistant:
        """Create or update an assistant."""
        db = await get_database()

        if_exists = data.get("if_exists", "raise")

        async with db.transaction() as conn:
            # Check if assistant exists
            existing_query = "SELECT * FROM assistants WHERE assistant_id = $1"
            existing_row = await conn.fetchrow(existing_query, uuid.UUID(assistant_id))

            if existing_row:
                if if_exists == "raise":
                    raise ItemExistsError("Assistant", assistant_id)
                elif if_exists == "do_nothing":
                    return Assistant.from_row(existing_row)

                # Update existing assistant
                update_data = {
                    "graph_id": data.get("graph_id", existing_row["graph_id"]),
                    "name": data.get("name", existing_row["name"]),
                    "config": data.get("config", existing_row["config"]),
                    "metadata": data.get("metadata", existing_row["metadata"]),
                    "version": existing_row["version"] + 1,
                }

                query, params = DatabaseQuery.build_update_query(
                    "assistants", update_data, {"assistant_id": uuid.UUID(assistant_id)}
                )

                row = await conn.fetchrow(query, *params)

                # Store version history
                await self._store_version(conn, Assistant.from_row(row))

            else:
                # Create new assistant
                insert_data = {
                    "assistant_id": uuid.UUID(assistant_id),
                    "graph_id": data["graph_id"],
                    "name": data.get("name", "Untitled"),
                    "config": data.get("config", {}),
                    "metadata": data.get("metadata", {}),
                    "version": 1,
                }

                query, params = DatabaseQuery.build_insert_query(
                    "assistants", insert_data
                )
                row = await conn.fetchrow(query, *params)

                # Store initial version
                await self._store_version(conn, Assistant.from_row(row))

            logger.info(
                f"{'Updated' if existing_row else 'Created'} assistant {assistant_id}"
            )
            return Assistant.from_row(row)

    async def delete(
        self, assistant_id: str, auth_context: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Delete an assistant."""
        db = await get_database()

        async with db.transaction() as conn:
            # Check if assistant exists
            check_query = "SELECT assistant_id FROM assistants WHERE assistant_id = $1"
            existing = await conn.fetchval(check_query, uuid.UUID(assistant_id))

            if not existing:
                raise ItemNotFoundError("Assistant", assistant_id)

            # Delete assistant (CASCADE will handle versions)
            delete_query = "DELETE FROM assistants WHERE assistant_id = $1"
            await conn.execute(delete_query, uuid.UUID(assistant_id))

            logger.info(f"Deleted assistant {assistant_id}")
            return {"deleted": True, "assistant_id": assistant_id}

    async def search(
        self, filters: Dict[str, Any], auth_context: Optional[Any] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Search for assistants matching filters."""
        db = await get_database()

        # Build search filters
        search_filters = {}
        if filters.get("graph_id"):
            search_filters["graph_id"] = filters["graph_id"]
        if filters.get("metadata"):
            search_filters["metadata"] = filters["metadata"]

        limit = filters.get("limit", 10)
        offset = filters.get("offset", 0)

        async with db.get_connection() as conn:
            query, params = DatabaseQuery.build_search_query(
                "assistants",
                search_filters,
                limit=limit,
                offset=offset,
                order_by="created_at",
                order_direction="DESC",
            )

            rows = await conn.fetch(query, *params)

            total = rows[0]["total_count"] if rows else 0

            for row in rows:
                assistant = Assistant.from_row(row)
                yield {"assistant": assistant.to_dict(), "total": total}

    async def patch(
        self,
        assistant_id: str,
        updates: Dict[str, Any],
        auth_context: Optional[Any] = None,
    ) -> Assistant:
        """Update specific fields of an assistant."""
        db = await get_database()

        async with db.transaction() as conn:
            # Get current assistant
            current_query = "SELECT * FROM assistants WHERE assistant_id = $1"
            current_row = await conn.fetchrow(current_query, uuid.UUID(assistant_id))

            if not current_row:
                raise ItemNotFoundError("Assistant", assistant_id)

            # Build update data
            update_data = {}

            if "graph_id" in updates:
                update_data["graph_id"] = updates["graph_id"]
            if "name" in updates:
                update_data["name"] = updates["name"]
            if "config" in updates:
                update_data["config"] = updates["config"]
            if "metadata" in updates:
                # Merge metadata
                current_metadata = current_row["metadata"] or {}
                current_metadata.update(updates["metadata"])
                update_data["metadata"] = current_metadata

            update_data["version"] = current_row["version"] + 1

            query, params = DatabaseQuery.build_update_query(
                "assistants", update_data, {"assistant_id": uuid.UUID(assistant_id)}
            )

            row = await conn.fetchrow(query, *params)

            # Store version history
            assistant = Assistant.from_row(row)
            await self._store_version(conn, assistant)

            logger.info(f"Patched assistant {assistant_id}")
            return assistant

    async def get_versions(
        self,
        assistant_id: str,
        filters: Optional[Dict[str, Any]] = None,
        auth_context: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Get versions of an assistant."""
        db = await get_database()

        # Check if assistant exists
        await self.get(assistant_id, auth_context)

        async with db.get_connection() as conn:
            query = """
            SELECT * FROM assistant_versions 
            WHERE assistant_id = $1 
            ORDER BY version DESC
            """

            params = [uuid.UUID(assistant_id)]

            if filters:
                limit = filters.get("limit", 10)
                offset = filters.get("offset", 0)
                query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
                params.extend([limit, offset])

            rows = await conn.fetch(query, *params)

            versions = []
            for row in rows:
                version_data = {
                    "assistant_id": str(row["assistant_id"]),
                    "version": row["version"],
                    "graph_id": row["graph_id"],
                    "name": row["name"],
                    "config": row["config"] or {},
                    "metadata": row["metadata"] or {},
                    "created_at": (
                        row["created_at"].isoformat() + "Z"
                        if row["created_at"]
                        else None
                    ),
                }
                versions.append(version_data)

            return versions

    async def set_latest(
        self, assistant_id: str, version: int, auth_context: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Set the latest version of an assistant."""
        db = await get_database()

        async with db.transaction() as conn:
            # Get the specific version
            version_query = """
            SELECT * FROM assistant_versions 
            WHERE assistant_id = $1 AND version = $2
            """
            version_row = await conn.fetchrow(
                version_query, uuid.UUID(assistant_id), version
            )

            if not version_row:
                raise ValueError(
                    f"Version {version} not found for assistant {assistant_id}"
                )

            # Update the main assistant record
            update_data = {
                "graph_id": version_row["graph_id"],
                "name": version_row["name"],
                "config": version_row["config"],
                "metadata": version_row["metadata"],
                "version": version,
            }

            query, params = DatabaseQuery.build_update_query(
                "assistants", update_data, {"assistant_id": uuid.UUID(assistant_id)}
            )

            await conn.execute(query, *params)

            logger.info(f"Set assistant {assistant_id} to version {version}")
            return {"success": True, "version": version}

    async def _store_version(self, conn: Any, assistant: Assistant) -> None:
        """Store a version of an assistant."""
        version_data = {
            "assistant_id": uuid.UUID(assistant.assistant_id),
            "version": assistant.version,
            "graph_id": assistant.graph_id,
            "name": assistant.name,
            "config": assistant.config,
            "metadata": assistant.metadata,
        }

        # Insert or update version
        insert_query = """
        INSERT INTO assistant_versions (assistant_id, version, graph_id, name, config, metadata)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (assistant_id, version) 
        DO UPDATE SET 
            graph_id = EXCLUDED.graph_id,
            name = EXCLUDED.name,
            config = EXCLUDED.config,
            metadata = EXCLUDED.metadata
        """

        await conn.execute(
            insert_query,
            version_data["assistant_id"],
            version_data["version"],
            version_data["graph_id"],
            version_data["name"],
            version_data["config"],
            version_data["metadata"],
        )


# Global instance
AssistantStorage = PostgreSQLAssistantStorage()
