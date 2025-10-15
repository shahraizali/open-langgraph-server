"""Database startup and initialization functions."""

import logging
import os
from alembic import command
from alembic.config import Config

from .base import engine
from .migrations import run_migrations

logger = logging.getLogger(__name__)


async def initialize_database():
    """Initialize database with migrations."""
    try:
        logger.info("Initializing database with Alembic migrations...")
        
        # Run migrations
        await run_migrations()
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_database():
    """Close database connections."""
    try:
        await engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


def check_database_url():
    """Check if database URL is configured."""
    database_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres")
    
    if "postgresql+asyncpg://" not in database_url:
        raise ValueError(
            f"Invalid DATABASE_URL format. Expected postgresql+asyncpg://, got: {database_url[:20]}..."
        )
    
    logger.info(f"Using database URL: {database_url.split('@')[0]}@[HIDDEN]")
    return database_url