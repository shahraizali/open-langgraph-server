"""Run-related schemas for the LangGraph Agent Protocol."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, AnyUrl, AwareDatetime, conint, field_validator, RootModel

from .common import Config, Message, OnCompletion, OnDisconnect, IfNotExists, StreamMode, CommandSchema, CheckpointSchema, LangsmithTracer


class RunStatus(Enum):
    pending = "pending"
    error = "error"
    success = "success"
    timeout = "timeout"
    interrupted = "interrupted"


class RunCreate(BaseModel):
    assistant_id: Union[str, UUID] = Field(
        ..., description="The agent ID to run.", title="Agent Id"
    )
    thread_id: Optional[UUID] = Field(
        None,
        description="The ID of the thread to run. If not provided, creates a stateless run. 'thread_id' is ignored unless Threads stage is implemented.",
        title="Thread Id",
    )
    input: Optional[Union[Dict[str, Any], List, str, float, bool]] = Field(
        None, description="The input to the agent.", title="Input"
    )
    messages: Optional[List[Message]] = Field(
        None,
        description="The messages to pass an input to the agent.",
        title="Messages",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadata to assign to the run.", title="Metadata"
    )
    config: Optional[Config] = Field(
        None, description="The configuration for the agent.", title="Config"
    )
    webhook: Optional[AnyUrl] = Field(
        None, description="Webhook to call after run finishes.", title="Webhook"
    )
    interrupt_before: Optional[Union[Literal["*"], List[str]]] = Field(
        None, description="Nodes to interrupt before."
    )
    interrupt_after: Optional[Union[Literal["*"], List[str]]] = Field(
        None, description="Nodes to interrupt after."
    )
    on_completion: Optional[OnCompletion] = Field(
        None,
        description="Whether to delete or keep the thread when run completes. Must be one of 'delete' or 'keep'. Defaults to 'delete' when thread_id not provided, otherwise 'keep'.",
        title="On Completion",
    )
    on_disconnect: Optional[OnDisconnect] = Field(
        "cancel",
        description="The disconnect mode to use. Must be one of 'cancel' or 'continue'.",
        title="On Disconnect",
    )
    if_not_exists: Optional[IfNotExists] = Field(
        "reject",
        description="How to handle missing thread. Must be either 'reject' (raise error if missing), or 'create' (create new thread).",
        title="If Not Exists",
    )
    stream_mode: Optional[Union[StreamMode, List[StreamMode]]] = Field(
        "values", description="The stream mode(s) to use.", title="Stream Mode"
    )
    checkpoint_id: Optional[str] = None
    checkpoint: Optional[CheckpointSchema] = None
    command: Optional[CommandSchema] = None
    stream_subgraphs: Optional[bool] = None
    stream_resumable: Optional[bool] = None
    after_seconds: Optional[int] = None
    feedback_keys: Optional[List[str]] = None
    langsmith_tracer: Optional[LangsmithTracer] = None
    multitask_strategy: Optional[str] = None

    @field_validator("assistant_id", mode="before")
    @classmethod
    def validate_assistant_id(cls, v):
        """Ensure assistant_id is always converted to string for consistent processing."""
        return str(v)


class Run(RunCreate):
    run_id: UUID = Field(..., description="The ID of the run.", title="Run Id")
    created_at: AwareDatetime = Field(
        ..., description="The time the run was created.", title="Created At"
    )
    updated_at: AwareDatetime = Field(
        ..., description="The last time the run was updated.", title="Updated At"
    )
    status: RunStatus


class RunSearchRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Run metadata to filter on.", title="Metadata"
    )
    status: Optional[RunStatus] = Field(
        None, description="Run status to filter on.", title="Run Status"
    )
    thread_id: Optional[UUID] = Field(
        None, description="The ID of the thread to filter on.", title="Thread Id"
    )
    agent_id: Optional[str] = Field(
        None, description="The ID of the agent to filter on.", title="Agent Id"
    )
    limit: Optional[conint(ge=1, le=1000)] = Field(
        10, description="Maximum number to return.", title="Limit"
    )
    offset: Optional[conint(ge=0)] = Field(
        0, description="Offset to start from.", title="Offset"
    )


class RunWaitResponse(BaseModel):
    run: Optional[Run] = Field(None, description="The run information.", title="Run")
    values: Optional[Dict[str, Any]] = Field(
        None, description="The values returned by the run.", title="Values"
    )
    messages: Optional[List[Message]] = Field(
        None, description="The messages returned by the run.", title="Messages"
    )


class RunBatchCreate(RootModel[List[RunCreate]]):
    root: List[RunCreate]