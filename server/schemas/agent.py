"""Agent-related schemas for compatibility with the LangGraph Agent Protocol."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, RootModel, conint


class Capabilities(BaseModel):
    ap_io_messages: Optional[bool] = Field(
        None,
        alias="ap.io.messages",
        description="Whether the agent supports Messages as input/output/state. If true, the agent uses the `messages` key in threads/runs endpoints.",
        title="Messages",
    )
    ap_io_streaming: Optional[bool] = Field(
        None,
        alias="ap.io.streaming",
        description="Whether the agent supports streaming output.",
        title="Streaming",
    )

    model_config = ConfigDict(
        extra="allow",
    )


class Agent(BaseModel):
    agent_id: str = Field(..., description="The ID of the agent.", title="Agent Id")
    name: str = Field(..., description="The name of the agent", title="Agent Name")
    description: Optional[str] = Field(
        None, description="The description of the agent.", title="Description"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="The agent metadata.", title="Metadata"
    )
    capabilities: Capabilities = Field(
        ...,
        description="Describes which protocol features the agent supports. In addition to the standard capabilities (prefixed with ap.), implementations can declare custom capabilities, named in reverse domain notation (eg. com.example.some.capability).",
        title="Agent Capabilities",
    )


class AgentSchema(BaseModel):
    agent_id: str = Field(..., description="The ID of the agent.", title="Agent Id")
    input_schema: Dict[str, Any] = Field(
        ...,
        description="The schema for the agent input. In JSON Schema format.",
        title="Input Schema",
    )
    output_schema: Dict[str, Any] = Field(
        ...,
        description="The schema for the agent output. In JSON Schema format.",
        title="Output Schema",
    )
    state_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="The schema for the agent's internal state. In JSON Schema format.",
        title="State Schema",
    )
    config_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="The schema for the agent config. In JSON Schema format.",
        title="Config Schema",
    )


class AgentsSearchPostRequest(BaseModel):
    name: Optional[str] = Field(None, description="Name of the agent to search.")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadata of the agent to search."
    )
    limit: Optional[conint(ge=1, le=1000)] = Field(
        10, description="Maximum number to return.", title="Limit"
    )
    offset: Optional[conint(ge=0)] = Field(
        0, description="Offset to start from.", title="Offset"
    )


class AgentsSearchPostResponse(RootModel[List[Agent]]):
    root: List[Agent] = Field(..., title="Response Search Agents")