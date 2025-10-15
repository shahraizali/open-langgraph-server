"""Assistant-related schemas for the LangGraph Agent Protocol."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, conint

from .common import Config


class AssistantConfigurable(BaseModel):
    """Configuration parameters for assistants."""

    thread_id: Optional[str] = Field(
        None, description="Thread ID for stateful execution"
    )
    thread_ts: Optional[str] = Field(None, description="Thread timestamp")

    class Config:
        extra = "allow"


class AssistantConfig(BaseModel):
    """Assistant configuration."""

    tags: Optional[List[str]] = Field(None, description="Tags for the assistant")
    recursion_limit: Optional[int] = Field(None, description="Maximum recursion depth")
    configurable: Optional[AssistantConfigurable] = Field(
        None, description="Configurable parameters"
    )

    class Config:
        extra = "allow"


class Assistant(BaseModel):
    """LangGraph Assistant model."""

    assistant_id: str = Field(..., description="The ID of the assistant")
    graph_id: str = Field(..., description="The graph ID this assistant uses")
    config: AssistantConfig = Field(
        default_factory=AssistantConfig, description="Assistant configuration"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Assistant context"
    )
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Assistant metadata"
    )
    name: Optional[str] = Field(None, description="Assistant name")
    version: int = Field(1, description="Assistant version")


class AssistantCreate(BaseModel):
    """Request model for creating an assistant."""

    assistant_id: Optional[str] = Field(
        None, description="Assistant ID (generated if not provided)"
    )
    graph_id: str = Field(..., description="The graph to use")
    config: Optional[AssistantConfig] = Field(
        None, description="Assistant configuration"
    )
    context: Optional[Dict[str, Any]] = Field(None, description="Assistant context")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Assistant metadata")
    if_exists: Optional[Literal["raise", "do_nothing"]] = Field(
        "raise", description="Behavior if assistant exists"
    )
    name: Optional[str] = Field(None, description="Assistant name")


class AssistantPatch(BaseModel):
    """Request model for updating an assistant."""

    graph_id: Optional[str] = Field(None, description="The graph to use")
    config: Optional[AssistantConfig] = Field(
        None, description="Assistant configuration"
    )
    context: Optional[Dict[str, Any]] = Field(None, description="Assistant context")
    name: Optional[str] = Field(None, description="Assistant name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to merge")


class AssistantSearchRequest(BaseModel):
    """Request model for searching assistants."""

    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadata to search for"
    )
    graph_id: Optional[str] = Field(None, description="Filter by graph ID")
    limit: Optional[conint(ge=1, le=1000)] = Field(
        10, description="Maximum number to return"
    )
    offset: Optional[conint(ge=0)] = Field(0, description="Offset to start from")


class AssistantLatestVersion(BaseModel):
    """Request model for setting latest assistant version."""

    version: int = Field(..., description="Version number to set as latest")


class AssistantSchema(BaseModel):
    """Assistant schema response model."""

    graph_id: str = Field(..., description="The graph ID")
    input_schema: Optional[Dict[str, Any]] = Field(
        None, description="Input JSON schema"
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        None, description="Output JSON schema"
    )
    state_schema: Optional[Dict[str, Any]] = Field(
        None, description="State JSON schema"
    )
    config_schema: Optional[Dict[str, Any]] = Field(
        None, description="Config JSON schema"
    )