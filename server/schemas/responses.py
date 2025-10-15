"""Response schemas for the LangGraph Agent Protocol."""

from __future__ import annotations

from typing import List
from pydantic import Field, RootModel

from .thread import Thread, ThreadState
from .run import Run


class ThreadsSearchPostResponse(RootModel[List[Thread]]):
    root: List[Thread] = Field(..., title="Response Search Threads Threads Search Post")


class ThreadsThreadIdHistoryGetResponse(RootModel[List[ThreadState]]):
    root: List[ThreadState]


class RunsSearchPostResponse(RootModel[List[Run]]):
    root: List[Run]