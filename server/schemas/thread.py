"""Thread-related schemas for the LangGraph Agent Protocol."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, AwareDatetime, ConfigDict

from .common import Message, IfExists


class ThreadStatus(Enum):
    idle = "idle"
    busy = "busy"
    interrupted = "interrupted"
    error = "error"


class ThreadCheckpoint(BaseModel):
    checkpoint_id: UUID = Field(
        ..., description="The ID of the checkpoint.", title="Checkpoint Id"
    )

    model_config = ConfigDict(
        extra="allow",
    )


class ThreadCreate(BaseModel):
    thread_id: Optional[UUID] = Field(
        None,
        description="The ID of the thread. If not provided, a random UUID will be generated.",
        title="Thread Id",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadata to add to thread.", title="Metadata"
    )
    if_exists: Optional[IfExists] = Field(
        "raise",
        description="How to handle duplicate creation. Must be either 'raise' (raise error if duplicate), or 'do_nothing' (return existing thread).",
        title="If Exists",
    )


class Thread(BaseModel):
    thread_id: UUID = Field(..., description="The ID of the thread.", title="Thread Id")
    created_at: AwareDatetime = Field(
        ..., description="The time the thread was created.", title="Created At"
    )
    updated_at: AwareDatetime = Field(
        ..., description="The last time the thread was updated.", title="Updated At"
    )
    metadata: Dict[str, Any] = Field(
        ..., description="The thread metadata.", title="Metadata"
    )
    status: ThreadStatus = Field(
        ..., description="The status of the thread.", title="Thread Status"
    )
    values: Optional[Dict[str, Any]] = Field(
        None, description="The current state of the thread.", title="Values"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="The thread configuration.", title="Config"
    )
    interrupts: Optional[Dict[str, Any]] = Field(
        None, description="The thread interrupts.", title="Interrupts"
    )
    messages: Optional[List[Message]] = Field(
        None,
        description="The current Messages of the thread. If messages are contained in Thread.values, implementations should remove them from values when returning messages. When this key isn't present it means the thread/agent doesn't support messages.",
        title="Messages",
    )


class ThreadState(BaseModel):
    checkpoint: ThreadCheckpoint = Field(
        ..., description="The identifier for this checkpoint.", title="Checkpoint"
    )
    values: Dict[str, Any] = Field(
        ..., description="The current state of the thread.", title="Values"
    )
    messages: Optional[List[Message]] = Field(
        None,
        description="The current messages of the thread. This key isn't present for agents that don't support messages.",
        title="Messages",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="The checkpoint metadata.", title="Metadata"
    )


class ThreadPatch(BaseModel):
    checkpoint: Optional[ThreadCheckpoint] = Field(
        None,
        description="The identifier of the checkpoint to branch from. Ignored for metadata-only patches. If not provided, defaults to the latest checkpoint.",
        title="Checkpoint",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata to merge with existing thread metadata.",
        title="Metadata",
    )
    values: Optional[Dict[str, Any]] = Field(
        None, description="Values to merge with existing thread values.", title="Values"
    )
    messages: Optional[List[Message]] = Field(
        None,
        description="Messages to combine with current thread messages.",
        title="Messages",
    )


class ThreadSearchRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Thread metadata to filter on.", title="Metadata"
    )
    values: Optional[Dict[str, Any]] = Field(
        None, description="State values to filter on.", title="Values"
    )
    status: Optional[ThreadStatus] = Field(
        None, description="Thread status to filter on.", title="Thread Status"
    )
    limit: Optional[int] = Field(
        10, description="Maximum number to return.", title="Limit"
    )
    offset: Optional[int] = Field(
        0, description="Offset to start from.", title="Offset"
    )