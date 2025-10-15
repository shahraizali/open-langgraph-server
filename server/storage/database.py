"""PostgreSQL database connection management and utilities."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

import asyncpg
from asyncpg import Pool, Connection

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration from environment variables."""

    def __init__(self):
        self.database_url = os.getenv(
            "DATABASE_URL", "postgresql://postgres:password@localhost:5432/langgraph"
        )
        self.pool_size = int(os.getenv("DATABASE_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
        self.command_timeout = int(os.getenv("DATABASE_COMMAND_TIMEOUT", "60"))
        self.server_timeout = int(os.getenv("DATABASE_SERVER_TIMEOUT", "10"))


class DatabaseManager:
    """Manages PostgreSQL connections and operations."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.pool: Optional[Pool] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database connection pool."""
        if self._initialized:
            return

        try:
            logger.info("Initializing database connection pool...")

            self.pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=1,
                max_size=self.config.pool_size,
                command_timeout=self.config.command_timeout,
                server_settings={"application_name": "langgraph-server", "jit": "off"},
            )

            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")

            logger.info(
                f"Database pool initialized with {self.config.pool_size} connections"
            )
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self) -> None:
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Connection, None]:
        """Get a database connection from the pool."""
        if not self._initialized or not self.pool:
            raise RuntimeError("Database not initialized")

        async with self.pool.acquire() as connection:
            yield connection

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Connection, None]:
        """Get a database connection with transaction management."""
        async with self.get_connection() as conn:
            async with conn.transaction():
                yield conn

    async def execute_schema(self, schema_path: Optional[Path] = None) -> None:
        """Execute the database schema from SQL file."""
        if schema_path is None:
            schema_path = Path(__file__).parent / "schema.sql"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        schema_sql = schema_path.read_text()

        async with self.get_connection() as conn:
            try:
                await conn.execute(schema_sql)
                logger.info("Database schema executed successfully")
            except Exception as e:
                logger.error(f"Failed to execute schema: {e}")
                raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the database."""
        try:
            async with self.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")

                # Get some basic stats
                pool_stats = {
                    "size": self.pool.get_size() if self.pool else 0,
                    "min_size": getattr(self.pool, "_min_size", 0) if self.pool else 0,
                    "max_size": getattr(self.pool, "_max_size", 0) if self.pool else 0,
                }

                return {
                    "status": "healthy",
                    "connection_test": result == 1,
                    "pool": pool_stats,
                }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "pool": {"size": 0, "min_size": 0, "max_size": 0},
            }

    async def migrate(self) -> None:
        """Run database migrations."""
        try:
            await self.execute_schema()
            logger.info("Database migration completed")
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            raise


class DatabaseQuery:
    """Utility class for building database queries."""

    @staticmethod
    def build_search_query(
        table: str,
        filters: Dict[str, Any],
        limit: int = 10,
        offset: int = 0,
        order_by: str = "created_at",
        order_direction: str = "DESC",
    ) -> tuple[str, list]:
        """Build a search query with filters."""
        where_conditions = []
        params = []
        param_counter = 1

        for key, value in filters.items():
            if value is None:
                continue

            if key == "metadata" and isinstance(value, dict):
                # JSONB containment search
                where_conditions.append(f"metadata @> ${param_counter}")
                params.append(json.dumps(value))
                param_counter += 1
            elif key == "values" and isinstance(value, dict):
                # JSONB containment search for values
                where_conditions.append(f"values @> ${param_counter}")
                params.append(json.dumps(value))
                param_counter += 1
            else:
                # Exact match
                where_conditions.append(f"{key} = ${param_counter}")
                params.append(value)
                param_counter += 1

        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)

        query = f"""
        SELECT *, COUNT(*) OVER() as total_count 
        FROM {table} 
        {where_clause}
        ORDER BY {order_by} {order_direction}
        LIMIT ${param_counter} OFFSET ${param_counter + 1}
        """

        params.extend([limit, offset])

        return query, params

    @staticmethod
    def build_insert_query(table: str, data: Dict[str, Any]) -> tuple[str, list]:
        """Build an INSERT query."""
        columns = list(data.keys())
        placeholders = [f"${i + 1}" for i in range(len(columns))]
        values = list(data.values())

        query = f"""
        INSERT INTO {table} ({", ".join(columns)})
        VALUES ({", ".join(placeholders)})
        RETURNING *
        """

        return query, values

    @staticmethod
    def build_update_query(
        table: str, data: Dict[str, Any], where_conditions: Dict[str, Any]
    ) -> tuple[str, list]:
        """Build an UPDATE query."""
        set_clauses = []
        params = []
        param_counter = 1

        # Build SET clause
        for key, value in data.items():
            set_clauses.append(f"{key} = ${param_counter}")
            params.append(value)
            param_counter += 1

        # Build WHERE clause
        where_clauses = []
        for key, value in where_conditions.items():
            where_clauses.append(f"{key} = ${param_counter}")
            params.append(value)
            param_counter += 1

        query = f"""
        UPDATE {table}
        SET {", ".join(set_clauses)}
        WHERE {" AND ".join(where_clauses)}
        RETURNING *
        """

        return query, params


# Global database manager instance
db_manager = DatabaseManager()


async def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    if not db_manager._initialized:
        await db_manager.initialize()
    return db_manager


async def close_database() -> None:
    """Close the global database manager."""
    await db_manager.close()
