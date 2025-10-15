"""Common schemas used across the LangGraph Agent Protocol."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, AnyUrl, AwareDatetime, conint


class StreamMode(str, Enum):
    values = "values"
    messages = "messages"
    messages_tuple = "messages-tuple"
    updates = "updates"
    events = "events"
    debug = "debug"
    custom = "custom"


class Config(BaseModel):
    tags: Optional[List[str]] = Field(None, title="Tags")
    recursion_limit: Optional[int] = Field(None, title="Recursion Limit")
    configurable: Optional[Dict[str, Any]] = Field(None, title="Configurable")


class RunnableConfig(BaseModel):
    """Runnable configuration for graph execution."""

    tags: Optional[List[str]] = Field(None, description="Tags for execution")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Execution metadata")
    run_name: Optional[str] = Field(None, description="Run name")
    max_concurrency: Optional[int] = Field(None, description="Maximum concurrency")
    recursion_limit: Optional[int] = Field(None, description="Recursion limit")
    configurable: Optional[Dict[str, Any]] = Field(
        None, description="Configurable parameters"
    )
    run_id: Optional[UUID] = Field(None, description="Run ID")


class OnCompletion(Enum):
    delete = "delete"
    keep = "keep"


class OnDisconnect(Enum):
    cancel = "cancel"
    continue_ = "continue"


class IfNotExists(Enum):
    create = "create"
    reject = "reject"


class IfExists(Enum):
    raise_ = "raise"
    do_nothing = "do_nothing"


class CommandSchema(BaseModel):
    goto: Optional[Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]] = None
    update: Optional[Union[Dict[str, Any], List[tuple[str, Any]]]] = None
    resume: Optional[Any] = None


class CheckpointSchema(BaseModel):
    checkpoint_id: Optional[str] = None
    checkpoint_ns: Optional[str] = None
    checkpoint_map: Optional[Dict[str, Any]] = None


class LangsmithTracer(BaseModel):
    project_name: Optional[str] = None
    example_id: Optional[str] = None


class Content(BaseModel):
    text: str
    type: Literal["text"]
    metadata: Optional[Dict[str, Any]] = None


class Content1(BaseModel):
    type: str
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class Message(BaseModel):
    role: str = Field(..., description="The role of the message.", title="Role")
    content: Union[str, List[Union[Content, Content1]]] = Field(
        ..., description="The content of the message.", title="Content"
    )
    id: Optional[str] = Field(None, description="The ID of the message.", title="Id")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="The metadata of the message.", title="Metadata"
    )

    class Config:
        extra = "allow"


class Action(Enum):
    interrupt = "interrupt"
    rollback = "rollback"


class ErrorResponse(BaseModel):
    code: Optional[str] = Field(
        None,
        description="For some errors that could be handled programmatically, a short string indicating the error code reported.",
    )
    message: Optional[str] = Field(
        None, description="A human-readable short description of the error."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="A dictionary of additional information about the error."
    )