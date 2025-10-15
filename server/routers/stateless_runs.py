from __future__ import annotations

from fastapi import APIRouter

from typing import Any, Union
from ..schemas import ErrorResponse, RunCreate, RunWaitResponse

router = APIRouter(tags=["Stateless Runs"])


@router.post(
    "/runs",
    response_model=Any,
    responses={
        "404": {"model": ErrorResponse},
        "409": {"model": ErrorResponse},
        "422": {"model": ErrorResponse},
    },
    tags=["Stateless Runs"],
)
def run_stateless_runs_post(body: RunCreate) -> Union[Any, ErrorResponse]:
    """
    Create Background Run
    """
    pass


@router.post(
    "/runs/stream",
    response_model=str,
    responses={
        "404": {"model": ErrorResponse},
        "409": {"model": ErrorResponse},
        "422": {"model": ErrorResponse},
    },
    tags=["Stateless Runs"],
)
def stream_run_stateless_runs_stream_post(body: RunCreate) -> Union[str, ErrorResponse]:
    """
    Create Run, Stream Output
    """
    pass


@router.post(
    "/runs/wait",
    response_model=RunWaitResponse,
    responses={
        "404": {"model": ErrorResponse},
        "409": {"model": ErrorResponse},
        "422": {"model": ErrorResponse},
    },
    tags=["Stateless Runs"],
)
def wait_run_stateless_runs_wait_post(
    body: RunCreate,
) -> Union[RunWaitResponse, ErrorResponse]:
    """
    Create Run, Wait for Output
    """
    pass
