"""SQLAlchemy models for LangGraph Agent Protocol Server."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, DateTime, Integer, JSON, Text, ForeignKey, UniqueConstraint, Index,
    PrimaryKeyConstraint, LargeBinary
)
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base


class Assistant(Base):
    """Assistant model."""
    __tablename__ = "assistants"

    assistant_id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=True)
    graph_id = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    version = Column(Integer, nullable=False, default=1)
    config = Column(JSONB, nullable=False, default={})
    context = Column(JSONB, nullable=False, default={})
    metadata_ = Column("metadata", JSONB, nullable=False, default={})

    # Indexes
    __table_args__ = (
        Index('idx_assistants_graph_id', 'graph_id'),
        Index('idx_assistants_created_at', 'created_at'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "assistant_id": self.assistant_id,
            "name": self.name,
            "graph_id": self.graph_id,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
            "updated_at": self.updated_at.isoformat() + "Z" if self.updated_at else None,
            "version": self.version,
            "config": self.config or {},
            "context": self.context or {},
            "metadata": self.metadata_ or {},
        }


class AssistantVersion(Base):
    """Assistant version history model."""
    __tablename__ = "assistant_versions"

    version_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    assistant_id = Column(String(255), ForeignKey('assistants.assistant_id', ondelete='CASCADE'), nullable=False)
    version = Column(Integer, nullable=False)
    graph_id = Column(String(255), nullable=False)
    config = Column(JSONB, nullable=False, default={})
    context = Column(JSONB, nullable=False, default={})
    metadata_ = Column("metadata", JSONB, nullable=False, default={})
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    name = Column(String(255), nullable=True)

    # Relationships
    assistant = relationship("Assistant", backref="versions")

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('assistant_id', 'version'),
        Index('idx_assistant_versions_assistant_id', 'assistant_id'),
        Index('idx_assistant_versions_version', 'version'),
        Index('idx_assistant_versions_created_at', 'created_at'),
    )


class Thread(Base):
    """Thread model."""
    __tablename__ = "threads"

    thread_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    metadata_ = Column("metadata", JSONB, nullable=False, default={})
    status = Column(String(50), nullable=False, default='idle')
    values = Column(JSONB, nullable=True)
    config = Column(JSONB, default={})
    interrupts = Column(JSONB, default={})

    # Indexes (matching schema.sql)
    __table_args__ = (
        Index('idx_threads_created_at', 'created_at'),
        Index('idx_threads_updated_at', 'updated_at'),
        Index('idx_threads_status', 'status'),
        Index('idx_threads_metadata_gin', metadata_, postgresql_using='gin'),  # Use column object
    )


class Run(Base):
    """Run model."""
    __tablename__ = "runs"

    run_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    thread_id = Column(PG_UUID(as_uuid=True), ForeignKey('threads.thread_id', ondelete='CASCADE'), nullable=True)
    assistant_id = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    status = Column(String(50), nullable=False, default='pending')
    metadata_ = Column("metadata", JSONB, nullable=False, default={})  # Missing from original model
    kwargs = Column(JSONB, nullable=False, default={})
    multitask_strategy = Column(String(50), nullable=False, default='reject')

    # Relationships
    thread = relationship("Thread", backref="runs")

    # Indexes (matching schema.sql)
    __table_args__ = (
        Index('idx_runs_thread_id', 'thread_id'),
        Index('idx_runs_assistant_id', 'assistant_id'),
        Index('idx_runs_status', 'status'),
        Index('idx_runs_created_at', 'created_at'),
        Index('idx_runs_updated_at', 'updated_at'),
        Index('idx_runs_metadata_gin', metadata_, postgresql_using='gin'),  # Use column object
    )


class BackgroundRun(Base):
    """Background run model."""
    __tablename__ = "background_runs"

    background_run_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)  # Different primary key name
    run_id = Column(PG_UUID(as_uuid=True), ForeignKey('runs.run_id', ondelete='CASCADE'), nullable=False)  # References runs table
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    status = Column(String(50), nullable=False, default='pending')
    result = Column(JSONB, nullable=True)  # Different field structure
    error_message = Column(Text, nullable=True)

    # Relationships - references runs instead of threads
    run = relationship("Run", backref="background_runs")

    # Indexes (matching schema.sql)
    __table_args__ = (
        Index('idx_background_runs_run_id', 'run_id'),
        Index('idx_background_runs_status', 'status'),
    )


class StoreItem(Base):
    """Store item model for key-value storage with namespacing."""
    __tablename__ = "store_items"

    item_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace = Column(String(255), nullable=False, default='')
    key = Column(String(255), nullable=False)
    value = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('namespace', 'key'),
        Index('idx_store_namespace', 'namespace'),
        Index('idx_store_key', 'key'),
        Index('idx_store_namespace_key', 'namespace', 'key'),
        Index('idx_store_value_gin', 'value', postgresql_using='gin'),
    )


class ThreadState(Base):
    """Thread state model."""
    __tablename__ = "thread_states"

    state_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    thread_id = Column(PG_UUID(as_uuid=True), ForeignKey('threads.thread_id', ondelete='CASCADE'), nullable=False)
    checkpoint_id = Column(PG_UUID(as_uuid=True), nullable=False, default=uuid4)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    values = Column(JSONB, nullable=False, default={})
    next_steps = Column(JSONB, nullable=False, default=[])
    checkpoint = Column(JSONB, nullable=True)  # Additional field from schema.sql
    metadata_ = Column("metadata", JSONB, nullable=True, default={})  # Additional field
    parent_checkpoint_id = Column(PG_UUID(as_uuid=True), nullable=True)  # Additional field
    tasks = Column(JSONB, nullable=True, default=[])  # Additional field

    # Relationships
    thread = relationship("Thread", backref="states")

    # Indexes
    __table_args__ = (
        Index('idx_thread_states_thread_id', 'thread_id'),
        Index('idx_thread_states_checkpoint_id', 'checkpoint_id'),
        Index('idx_thread_states_created_at', 'created_at'),
    )


# Note: Checkpoint tables (checkpoints, checkpoint_blobs, checkpoint_writes) 
# are created and managed automatically by LangGraph's PostgresSaver.
# No SQLAlchemy models are needed for these tables.