from __future__ import annotations

from fastapi import APIRouter

from typing import Any, Optional, Union
from uuid import UUID

from ..schemas import (
    Action,
    ErrorResponse,
    Run,
    RunSearchRequest,
    RunWaitResponse,
    RunsSearchPostResponse,
)

# Import RunStream from storage
from ..storage.streaming import RunStream

router = APIRouter(tags=["Background Runs"])


@router.post(
    "/runs",
    response_model=Run,
    responses={
        "404": {"model": ErrorResponse},
        "409": {"model": ErrorResponse},
        "422": {"model": ErrorResponse},
    },
    tags=["Background Runs"],
)
def create_run(body: RunStream) -> Union[Run, ErrorResponse]:
    """
    Create Background Run
    """
    pass


@router.post(
    "/runs/search",
    response_model=RunsSearchPostResponse,
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
    tags=["Background Runs"],
)
def search_runs(body: RunSearchRequest) -> Union[RunsSearchPostResponse, ErrorResponse]:
    """
    Search Runs
    """
    pass


@router.get(
    "/runs/{run_id}",
    response_model=Run,
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
    tags=["Background Runs"],
)
def get_run(run_id: UUID) -> Union[Run, ErrorResponse]:
    """
    Get Run
    """
    pass


@router.delete(
    "/runs/{run_id}",
    response_model=None,
    status_code=204,
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
    tags=["Background Runs"],
)
def delete_run(run_id: UUID) -> Optional[ErrorResponse]:
    """
    Delete Run
    """
    pass


@router.post(
    "/runs/{run_id}/cancel",
    response_model=None,
    status_code=204,
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
    tags=["Background Runs"],
)
def cancel_run(
    run_id: UUID, wait: Optional[bool] = False, action: Optional[Action] = "interrupt"
) -> Optional[ErrorResponse]:
    """
    Cancel Run
    """
    pass


@router.get(
    "/runs/{run_id}/stream",
    response_model=Any,
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
    tags=["Background Runs"],
)
def stream_run(run_id: UUID) -> Union[Any, ErrorResponse]:
    """
    Stream output from Run
    """
    pass


@router.get(
    "/runs/{run_id}/wait",
    response_model=RunWaitResponse,
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
    tags=["Background Runs"],
)
def wait_run(run_id: UUID) -> Union[RunWaitResponse, ErrorResponse]:
    """
    Wait for Run output
    """
    pass
