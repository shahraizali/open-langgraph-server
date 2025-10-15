from __future__ import annotations

import logging
import asyncio
from typing import Any, Dict, List, Optional, AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Query
from sse_starlette.sse import EventSourceResponse

from ..schemas import (
    Run,
    RunCreate,
    RunBatchCreate,
    CronCreate,
    CronSearch,
    ErrorResponse,
)
from ..storage.ops import Runs, RunKwargs, RunData
from ..storage.streaming import RunStream, stream_manager
from ..graph.loader import validate_assistant_id
from ..utils.auth import AuthContext, extract_auth_headers
from ..utils.serde import serialize_as_dict

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Runs"])


def get_auth_context(request: Request) -> Optional[AuthContext]:
    """Extract auth context from request."""
    return getattr(request.state, "auth", None)


async def create_valid_run(
    thread_id: Optional[str],
    payload: RunCreate,
    auth_context: Optional[AuthContext],
    headers: Optional[Dict[str, str]],
) -> RunData:
    """
    Create a valid run, exactly
    This is the core function that handles all the complex run creation logic,
    including config merging, multitask strategies, and auth integration.
    """
    assistant_id = payload.assistant_id
    run_id = str(uuid4())

    # Process and validate assistant ID
    logger.info(
        f"[ASSISTANT_ID DEBUG] Raw assistant_id from request: {assistant_id} (type: {type(assistant_id)})"
    )
    assistant_id_str = str(assistant_id)
    logger.info(f"[ASSISTANT_ID DEBUG] Converted to string: {assistant_id_str}")

    try:
        processed_assistant_id = await validate_assistant_id(assistant_id_str)
        logger.info(
            f"[ASSISTANT_ID DEBUG] Validated assistant_id: {processed_assistant_id}"
        )
    except ValueError as e:
        logger.error(f"[ASSISTANT_ID DEBUG] Validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    # Handle stream mode
    stream_mode = payload.stream_mode or ["values"]
    if not isinstance(stream_mode, list):
        stream_mode = [stream_mode]
    if not stream_mode:
        stream_mode = ["values"]

    # Handle multitask strategy - default to "interrupt" for thread runs to match LangGraph behavior
    if thread_id is not None:
        # For thread runs, use "interrupt" by default to match LangGraph
        multitask_strategy = payload.multitask_strategy or "interrupt"
    else:
        # For stateless runs, keep "reject" as default
        multitask_strategy = payload.multitask_strategy or "reject"

    prevent_insert_in_inflight = multitask_strategy == "reject"

    # Build base config from payload
    config = payload.config.model_dump() if payload.config else {}
    logger.info(f"[CONFIG DEBUG] Initial config from payload: {config}")

    # Handle checkpoint configuration
    if payload.checkpoint_id:
        if "configurable" not in config:
            config["configurable"] = {}
        config["configurable"]["checkpoint_id"] = payload.checkpoint_id
        logger.info(f"[CONFIG DEBUG] After adding checkpoint_id: {config}")

    if payload.checkpoint:
        if "configurable" not in config:
            config["configurable"] = {}

        # Filter out None/null values from checkpoint to prevent LangGraph errors
        checkpoint_config = {}
        for key, value in payload.checkpoint.model_dump().items():
            if value is not None:
                checkpoint_config[key] = value

        config["configurable"].update(checkpoint_config)
        logger.info(
            f"[CONFIG DEBUG] After adding checkpoint: {config}, payload.checkpoint: {payload.checkpoint}"
        )
        logger.info(
            f"[CONFIG DEBUG] Filtered checkpoint config (None values removed): {checkpoint_config}"
        )

    # Handle LangSmith tracer configuration
    if payload.langsmith_tracer:
        if "configurable" not in config:
            config["configurable"] = {}
        config["configurable"].update(
            {
                "langsmith_project": payload.langsmith_tracer.project_name,
                "langsmith_example_id": payload.langsmith_tracer.example_id,
            }
        )

    # Process headers
    if headers:
        auth_headers = extract_auth_headers(dict(headers))
        if auth_headers:
            if "configurable" not in config:
                config["configurable"] = {}
            config["configurable"].update(auth_headers)

    # Handle auth context injection
    user_id = None
    if auth_context:
        user_id = auth_context.user_id
        if "configurable" not in config:
            config["configurable"] = {}
        config["configurable"].update(
            {
                "langgraph_auth_user": auth_context.user,
                "langgraph_auth_user_id": user_id,
                "langgraph_auth_permissions": auth_context.scopes,
            }
        )

    # Handle feedback keys
    feedback_keys = None
    if payload.feedback_keys:
        if isinstance(payload.feedback_keys, list):
            feedback_keys = payload.feedback_keys
        else:
            feedback_keys = [payload.feedback_keys]
        if not feedback_keys:
            feedback_keys = None

    # Final validation: ensure no None values in configurable that could break LangGraph
    if "configurable" in config and isinstance(config["configurable"], dict):
        config["configurable"] = {
            k: v for k, v in config["configurable"].items() if v is not None
        }
        logger.info(f"[CONFIG DEBUG] Config after final None value cleanup: {config}")

    logger.info(f"[CONFIG DEBUG] Final config before RunKwargs creation: {config}")

    # Create run kwargs
    run_kwargs = RunKwargs(
        input=payload.input,
        command=payload.command,
        config=config,
        stream_mode=stream_mode,
        interrupt_before=payload.interrupt_before,
        interrupt_after=payload.interrupt_after,
        webhook=payload.webhook,
        feedback_keys=feedback_keys,
        temporary=thread_id is None and (payload.on_completion or "delete") == "delete",
        subgraphs=payload.stream_subgraphs or False,
        resumable=payload.stream_resumable or False,
    )

    logger.info(f"[CONFIG DEBUG] RunKwargs.config after creation: {run_kwargs.config}")

    # Create run options
    run_options = {
        "thread_id": thread_id,
        "user_id": user_id,
        "metadata": payload.metadata or {},
        "status": "pending",
        "multitask_strategy": multitask_strategy,
        "prevent_insert_if_inflight": prevent_insert_in_inflight,
        "after_seconds": payload.after_seconds or 0,
        "if_not_exists": (
            "create"
            if thread_id and payload.if_not_exists == "reject"
            else payload.if_not_exists
        ),
    }

    # Create run in storage
    first, inflight = await Runs.put(
        run_id, processed_assistant_id, run_kwargs, run_options, auth_context
    )

    if first and first.run_id == run_id:
        logger.info(f"Created run {run_id} for thread {thread_id}")

        # Update thread metadata with assistant information
        if thread_id:
            try:
                # Get assistant information to extract graph_id (with graph name fallback)
                from ..storage.postgres_storage import postgres_assistant_storage

                logger.info(
                    f"[ASSISTANT_ID DEBUG] Looking up assistant: {processed_assistant_id}"
                )
                assistant = await postgres_assistant_storage.get(processed_assistant_id)

                # If not found in database, check if it's a registered graph name
                if not assistant:
                    from ..graph.loader import GRAPHS

                    if processed_assistant_id in GRAPHS:
                        # Create virtual assistant object for graph
                        from ..storage.postgres_storage import Assistant
                        from datetime import datetime

                        assistant = Assistant(
                            assistant_id=processed_assistant_id,
                            name=processed_assistant_id,
                            graph_id=processed_assistant_id,
                            created_at=datetime.now(),
                            updated_at=datetime.now(),
                            version=1,
                            config={},
                            metadata={"created_by": "system", "type": "graph"},
                        )

                if assistant:
                    logger.info(
                        f"[ASSISTANT_ID DEBUG] Found assistant: {assistant.assistant_id}, graph_id: {assistant.graph_id}"
                    )
                    # Update thread metadata with assistant_id and graph_id
                    from ..storage.postgres_storage import postgres_thread_storage

                    await postgres_thread_storage.update_metadata_with_assistant(
                        thread_id, processed_assistant_id, assistant.graph_id
                    )
                    logger.info(
                        f"Updated thread {thread_id} metadata with assistant {processed_assistant_id}"
                    )
                else:
                    logger.warning(
                        f"[ASSISTANT_ID DEBUG] Assistant not found: {processed_assistant_id}"
                    )
            except Exception as e:
                logger.warning(f"Failed to update thread metadata: {e}")
                logger.warning(
                    f"[ASSISTANT_ID DEBUG] Exception details: {type(e).__name__}: {e}"
                )

        # Handle multitask strategies for inflight runs
        if (
            multitask_strategy == "interrupt" or multitask_strategy == "rollback"
        ) and inflight:
            try:
                await Runs.cancel(
                    thread_id,
                    [r.run_id for r in inflight],
                    {"action": multitask_strategy},
                    auth_context,
                )
            except Exception as error:
                logger.warning(
                    f"Failed to cancel inflight runs, might be already cancelled: {error}",
                    extra={
                        "run_ids": [r.run_id for r in inflight],
                        "thread_id": thread_id,
                    },
                )

        return first

    elif multitask_strategy == "reject":
        raise HTTPException(
            status_code=422,
            detail="Thread is already running a task. Wait for it to finish or choose a different multitask strategy.",
        )

    raise HTTPException(status_code=500, detail="Unreachable state when creating run")


def convert_run_data_to_model(run_data: RunData) -> Run:
    """Convert internal RunData to API Run model."""
    # Handle datetime formatting properly - don't add Z if already has timezone info
    created_at = run_data.created_at.isoformat()
    if not created_at.endswith("Z") and "+" not in created_at:
        created_at += "Z"

    updated_at = run_data.updated_at.isoformat()
    if not updated_at.endswith("Z") and "+" not in updated_at:
        updated_at += "Z"

    return Run(
        run_id=run_data.run_id,
        thread_id=run_data.thread_id,
        assistant_id=run_data.assistant_id,
        created_at=created_at,
        updated_at=updated_at,
        status=run_data.status,
        metadata=run_data.metadata,
        input=run_data.kwargs.input,
        stream_mode=run_data.kwargs.stream_mode,
    )


async def sse_generator(
    run_id: str, request: Request
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    SSE event generator for streaming runs.
    Handles disconnection, resumable streaming, and proper event formatting.
    """
    cancel_on_disconnect = (
        request.query_params.get("cancel_on_disconnect", "false").lower() == "true"
    )
    last_event_id = request.headers.get("Last-Event-ID")

    # Set up stream options
    stream_options = {
        "ignore404": True,
        "lastEventId": last_event_id,
    }

    if cancel_on_disconnect:
        # Create a cancellation signal
        signal = stream_manager.get_control(run_id)
        if signal:
            stream_options["cancelOnDisconnect"] = signal

    try:
        async for event in RunStream.join(
            run_id,
            None,  # Thread ID will be determined from run
            stream_options,
            get_auth_context(request),
        ):
            # Check if client disconnected
            if await request.is_disconnected():
                break

            # Format event for SSE
            yield {
                "id": event.get("id"),
                "event": event.get("event", "message"),
                "data": serialize_as_dict(event.get("data", {})),
            }

    except asyncio.CancelledError:
        logger.info(f"Client disconnected from run {run_id}")
    except Exception as e:
        logger.error(f"Error in SSE stream for run {run_id}: {e}")
        yield {
            "event": "error",
            "data": serialize_as_dict({"error": str(e)}),
        }


# Cron Endpoints (Not Implemented)
@router.post("/runs/crons", response_model=ErrorResponse, status_code=501)
async def create_cron(body: CronCreate):
    raise HTTPException(status_code=501, detail="Not Implemented")


@router.post("/runs/crons/search", response_model=ErrorResponse, status_code=501)
async def search_crons(body: CronSearch):
    raise HTTPException(status_code=501, detail="Not Implemented")


@router.delete("/runs/crons/{cron_id}", response_model=ErrorResponse, status_code=501)
async def delete_cron(cron_id: str):
    raise HTTPException(status_code=501, detail="Not Implemented")


@router.post(
    "/threads/{thread_id}/runs/crons", response_model=ErrorResponse, status_code=501
)
async def create_thread_cron(thread_id: str, body: CronCreate):
    raise HTTPException(status_code=501, detail="Not Implemented")


# Stateless Run Endpoints
@router.post("/runs/stream")
async def stream_stateless_run(request: Request, body: RunCreate):
    """Stream a stateless run"""
    run_data = await create_valid_run(
        None, body, get_auth_context(request), dict(request.headers)
    )

    return EventSourceResponse(
        sse_generator(run_data.run_id, request), media_type="text/event-stream"
    )


@router.get("/runs/{run_id}/stream")
async def stream_run_by_id(
    request: Request,
    run_id: str,
    cancel_on_disconnect: Optional[bool] = Query(False),
):
    """Stream an existing run by ID"""
    return EventSourceResponse(
        sse_generator(run_id, request), media_type="text/event-stream"
    )


@router.post("/runs/wait", response_model=Any)
async def wait_stateless_run(request: Request, body: RunCreate):
    """Wait for a stateless run to complete"""
    run_data = await create_valid_run(
        None, body, get_auth_context(request), dict(request.headers)
    )

    result = await Runs.wait(run_data.run_id, None, get_auth_context(request))
    return result


@router.post("/runs", response_model=Run)
async def create_stateless_run(request: Request, body: RunCreate):
    """Create a stateless run"""
    run_data = await create_valid_run(
        None, body, get_auth_context(request), dict(request.headers)
    )
    return convert_run_data_to_model(run_data)


@router.post("/runs/batch", response_model=List[Run])
async def batch_runs(request: Request, body: RunBatchCreate):
    """Create a batch of stateless runs."""
    auth_context = get_auth_context(request)
    headers = dict(request.headers)

    # Create all runs concurrently
    run_tasks = [
        create_valid_run(None, run_payload, auth_context, headers)
        for run_payload in body.root
    ]

    run_data_list = await asyncio.gather(*run_tasks)
    return [convert_run_data_to_model(run_data) for run_data in run_data_list]


# Stateful Run Endpoints (within a thread)
