"""Database migration utilities."""

import asyncio
import logging
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from .base import engine

logger = logging.getLogger(__name__)


def get_alembic_config() -> Config:
    """Get Alembic configuration."""
    config = Config("alembic.ini")
    return config


async def run_migrations():
    """Run all pending migrations."""
    config = get_alembic_config()
    
    def run_upgrade(connection):
        """Run migrations in sync mode."""
        config.attributes['connection'] = connection
        command.upgrade(config, "head")
    
    # Run migrations in async context
    async with engine.begin() as conn:
        await conn.run_sync(run_upgrade)


async def check_migration_status():
    """Check current migration status."""
    config = get_alembic_config()
    script_dir = ScriptDirectory.from_config(config)
    
    def get_current_revision():
        """Get current database revision."""
        return script_dir.get_current_head()
    
    async with engine.begin() as conn:
        current_rev = await conn.run_sync(get_current_revision)
        logger.info(f"Current migration revision: {current_rev}")
        return current_rev


async def create_all_tables():
    """Create all tables (for development/testing)."""
    from .base import Base
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("All tables created successfully")


if __name__ == "__main__":
    # Run migrations if called directly
    asyncio.run(run_migrations())