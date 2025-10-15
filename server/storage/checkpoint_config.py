"""Checkpoint configuration and factory."""

from .checkpoint import checkpointer as db_checkpointer


def get_checkpointer():
    """
    Get the PostgreSQL checkpoint implementation.

    Environment variables:
    - DATABASE_URL: Required for database storage
    """
    return db_checkpointer


# Export the configured checkpointer
checkpointer = get_checkpointer()
