"""Cron-related schemas for the LangGraph Agent Protocol."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel

from .assistant import AssistantConfig


class CronCreate(BaseModel):
    thread_id: UUID
    assistant_id: UUID
    checkpoint_id: Optional[str] = None
    input: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    config: Optional[AssistantConfig] = None
    webhook: Optional[str] = None
    interrupt_before: Optional[Literal["*"] | List[str]] = None
    interrupt_after: Optional[Literal["*"] | List[str]] = None
    multitask_strategy: Optional[str] = None


class CronSearch(BaseModel):
    assistant_id: Optional[UUID] = None
    thread_id: Optional[UUID] = None
    limit: Optional[int] = 10
    offset: Optional[int] = 0