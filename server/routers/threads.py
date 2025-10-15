"""Thread API implementation matching JavaScript server specification."""

from __future__ import annotations

import logging
import uuid
import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Response, Query, Request, Depends
from uuid import UUID

from sse_starlette.sse import EventSourceResponse

from ..schemas import (
    Thread,
    ThreadCreate,
    ThreadPatch,
    ThreadSearchRequest,
    ThreadStatus,
    Run,
    RunCreate,
    ErrorResponse,
)
from ..storage.postgres_storage import postgres_thread_storage as thread_storage
from ..helpers.thread_state import state_snapshot_to_thread_state

from ..storage.ops import Runs
from ..storage.streaming import RunStream, stream_manager
from ..utils.serde import serialize_as_dict

# Import functions from runs.py that are needed for thread runs
from .runs import create_valid_run, convert_run_data_to_model, get_auth_context


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Threads"])


async def parse_thread_search_request(request: Request) -> ThreadSearchRequest:
    """Custom dependency to handle both JSON and form data for thread search requests."""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        # Handle JSON request
        body = await request.json()
        return ThreadSearchRequest(**body)
    else:
        # Handle form data or other content types
        form_data = await request.form()
        body = {}

        # Parse form data
        for key, value in form_data.items():
            if key == "metadata" and value:
                try:
                    body[key] = json.loads(value)
                except json.JSONDecodeError:
                    body[key] = {}
            elif key in ["limit", "offset"]:
                body[key] = int(value) if value else None
            elif key == "status" and value:
                body[key] = ThreadStatus(value)
            elif key == "values" and value:
                try:
                    body[key] = json.loads(value)
                except json.JSONDecodeError:
                    body[key] = {}
            else:
                body[key] = value

        return ThreadSearchRequest(**body)


@router.post(
    "/threads",
    response_model=Thread,
    responses={"409": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def create_thread(body: ThreadCreate) -> Thread:
    """Create Thread"""
    try:
        # Generate ID if not provided
        thread_id = str(body.thread_id or uuid.uuid4())

        # Create thread
        thread_data = await thread_storage.put(
            thread_id,
            {"metadata": body.metadata or {}, "if_exists": body.if_exists or "raise"},
        )

        # Handle supersteps if provided (TODO: implement when needed)
        # if body.supersteps:
        #     await thread_storage.State.bulk(
        #         {"configurable": {"thread_id": thread_id}},
        #         body.supersteps
        #     )

        # Convert to API model
        thread_dict = thread_data.to_dict()
        return Thread(
            thread_id=UUID(thread_dict["thread_id"]),
            created_at=thread_dict["created_at"],
            updated_at=thread_dict["updated_at"],
            metadata=thread_dict["metadata"],
            status=ThreadStatus(thread_dict["status"]),
            values=thread_dict.get("values"),
            config=thread_dict.get("config"),
            interrupts=thread_dict.get("interrupts", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating thread: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/threads/search",
    response_model=List[Thread],
    responses={"422": {"model": ErrorResponse}},
)
async def search_threads(
    body: ThreadSearchRequest = Depends(parse_thread_search_request),
    response: Response = None,
) -> List[Thread]:
    """Search Threads"""
    try:
        # Convert request to storage filters
        filters = {
            "limit": body.limit,
            "offset": body.offset,
            "sort_by": getattr(body, "sort_by", "created_at"),
            "sort_order": getattr(body, "sort_order", "desc"),
        }

        if body.metadata:
            filters["metadata"] = body.metadata

        if body.status:
            filters["status"] = body.status.value

        if body.values:
            filters["values"] = body.values

        # Search threads
        threads = []
        total = 0

        async for result in thread_storage.search(filters):
            thread_data = result["thread"]

            # Convert to API model
            thread_dict = thread_data.to_dict()
            thread = Thread(
                thread_id=UUID(thread_dict["thread_id"]),
                created_at=thread_dict["created_at"],
                updated_at=thread_dict["updated_at"],
                metadata=thread_dict["metadata"],
                status=ThreadStatus(thread_dict["status"]),
                values=thread_dict.get("values"),
                config=thread_dict.get("config"),
                interrupts=thread_dict.get("interrupts", {}),
            )

            threads.append(thread)
            if total == 0:
                total = result["total"]

        # Set pagination header
        response.headers["X-Pagination-Total"] = str(total)

        return threads

    except Exception as e:
        logger.error(f"Error searching threads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/threads/{thread_id}",
    response_model=Thread,
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def get_thread(thread_id: UUID) -> Thread:
    """Get Thread"""
    try:
        # Get thread from storage
        thread_data = await thread_storage.get(str(thread_id))

        # Convert to API model
        thread_dict = thread_data.to_dict()
        return Thread(
            thread_id=UUID(thread_dict["thread_id"]),
            created_at=thread_dict["created_at"],
            updated_at=thread_dict["updated_at"],
            metadata=thread_dict["metadata"],
            status=ThreadStatus(thread_dict["status"]),
            values=thread_dict.get("values"),
            config=thread_dict.get("config"),
            interrupts=thread_dict.get("interrupts", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/threads/{thread_id}",
    status_code=204,
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def delete_thread(thread_id: UUID):
    """Delete Thread"""
    try:
        # Delete thread
        await thread_storage.delete(str(thread_id))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/threads/{thread_id}",
    response_model=Thread,
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def patch_thread(thread_id: UUID, body: ThreadPatch) -> Thread:
    """Patch Thread"""
    try:
        # Update thread
        updates = {}
        if body.metadata is not None:
            updates["metadata"] = body.metadata

        thread_data = await thread_storage.patch(str(thread_id), updates)

        # Convert to API model
        thread_dict = thread_data.to_dict()
        return Thread(
            thread_id=UUID(thread_dict["thread_id"]),
            created_at=thread_dict["created_at"],
            updated_at=thread_dict["updated_at"],
            metadata=thread_dict["metadata"],
            status=ThreadStatus(thread_dict["status"]),
            values=thread_dict.get("values"),
            config=thread_dict.get("config"),
            interrupts=thread_dict.get("interrupts", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error patching thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/threads/{thread_id}/copy",
    response_model=Thread,
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def copy_thread(thread_id: UUID) -> Thread:
    """Copy Thread"""
    try:
        # Copy thread
        thread_data = await thread_storage.copy(str(thread_id))

        # Convert to API model
        thread_dict = thread_data.to_dict()
        return Thread(
            thread_id=UUID(thread_dict["thread_id"]),
            created_at=thread_dict["created_at"],
            updated_at=thread_dict["updated_at"],
            metadata=thread_dict["metadata"],
            status=ThreadStatus(thread_dict["status"]),
            values=thread_dict.get("values"),
            config=thread_dict.get("config"),
            interrupts=thread_dict.get("interrupts", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error copying thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Thread State Management Endpoints


@router.get(
    "/threads/{thread_id}/state",
    response_model=Dict[str, Any],
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def get_thread_state(
    thread_id: UUID, subgraphs: Optional[bool] = False
) -> Dict[str, Any]:
    """Get Latest Thread State"""
    try:
        # Get state from storage
        state_data = await thread_storage.State.get(
            {"configurable": {"thread_id": str(thread_id)}}, {"subgraphs": subgraphs}
        )

        # Convert to API format
        return state_snapshot_to_thread_state(state_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread state {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/threads/{thread_id}/state",
    response_model=Dict[str, Any],
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def update_thread_state(thread_id: UUID, body: Dict[str, Any]) -> Dict[str, Any]:
    """Update Thread State"""
    try:
        # Build config
        config = {"configurable": {"thread_id": str(thread_id)}}

        if body.get("checkpoint_id"):
            config["configurable"]["checkpoint_id"] = body["checkpoint_id"]

        if body.get("checkpoint"):
            config["configurable"].update(body["checkpoint"])

        # Update state
        result = await thread_storage.State.post(
            config, body.get("values"), body.get("as_node")
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating thread state {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/threads/{thread_id}/state/{checkpoint_id}",
    response_model=Dict[str, Any],
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def get_thread_state_at_checkpoint(
    thread_id: UUID, checkpoint_id: UUID, subgraphs: Optional[bool] = False
) -> Dict[str, Any]:
    """Get Thread State At Checkpoint"""
    try:
        # Get state at checkpoint
        state_data = await thread_storage.State.get(
            {
                "configurable": {
                    "thread_id": str(thread_id),
                    "checkpoint_id": str(checkpoint_id),
                }
            },
            {"subgraphs": subgraphs},
        )

        # Convert to API format
        return state_snapshot_to_thread_state(state_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting thread state at checkpoint {thread_id}/{checkpoint_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/threads/{thread_id}/state/checkpoint",
    response_model=Dict[str, Any],
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def get_thread_state_at_checkpoint_post(
    thread_id: UUID, body: Dict[str, Any]
) -> Dict[str, Any]:
    """Get Thread State At Checkpoint Post"""
    try:
        # Build config with checkpoint
        config = {"configurable": {"thread_id": str(thread_id)}}
        if body.get("checkpoint"):
            config["configurable"].update(body["checkpoint"])

        # Get state
        state_data = await thread_storage.State.get(
            config, {"subgraphs": body.get("subgraphs", False)}
        )

        # Convert to API format
        return state_snapshot_to_thread_state(state_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread state at checkpoint {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Thread History Endpoints


@router.get(
    "/threads/{thread_id}/history",
    response_model=List[Dict[str, Any]],
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def get_thread_history(
    thread_id: UUID, limit: Optional[int] = 10, before: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get Thread History"""
    try:
        # Get states from LangGraph checkpoints table
        states = await _get_langgraph_thread_history(
            str(thread_id), limit=limit, before=before
        )

        return states

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread history {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _get_langgraph_thread_history(
    thread_id: str,
    limit: Optional[int] = 10,
    before: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Get thread history using LangGraph's state history."""
    try:
        # Get thread to access graph configuration
        thread_data = await thread_storage.get(thread_id)
        if not thread_data:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Get the graph_id from thread metadata
        graph_id = thread_data.metadata.get("graph_id")
        if not graph_id:
            # Fallback: try to infer from stored run data or use a default
            logger.warning(
                f"No graph_id found in thread {thread_id} metadata, attempting fallback"
            )
            # Check if we have any runs for this thread to infer graph_id
            from ..storage.ops import store

            thread_runs = [
                run for run in store.runs.values() if run.thread_id == thread_id
            ]
            if thread_runs:
                # Try to get graph_id from assistant_id of most recent run
                recent_run = max(thread_runs, key=lambda r: r.created_at)
                try:
                    assistant = await store.assistants.get(recent_run.assistant_id)
                    if assistant:
                        graph_id = assistant.graph_id
                    else:
                        # Might be a direct graph name
                        from ..graph.loader import GRAPHS

                        if recent_run.assistant_id in GRAPHS:
                            graph_id = recent_run.assistant_id
                except Exception:
                    pass

            if not graph_id:
                logger.error(f"Cannot determine graph_id for thread {thread_id}")
                return []

        # Load the graph
        from ..graph.loader import get_graph
        from ..storage.checkpoint import checkpointer
        from ..storage.ops import store

        # Initialize checkpointer
        await checkpointer.initialize()

        # Get the graph with proper configuration
        graph = get_graph(
            graph_id,
            thread_data.config or {},
            {"checkpointer": checkpointer, "store": store.store},
        )

        # Build config for state history retrieval
        config = {"configurable": {"thread_id": thread_id}}

        # Handle before parameter - build before config if provided
        before_config = None
        if before:
            before_config = {"configurable": {"checkpoint_id": before}}

        # Use LangGraph's native getStateHistory method
        states = []

        history_options = {"limit": limit or 10}

        # Handle before parameter
        if before_config:
            history_options["before"] = before_config

        # Handle metadata filter
        if metadata_filter:
            history_options["filter"] = metadata_filter

        try:
            # Use the native getStateHistory
            logger.info(
                f"[THREAD_HISTORY] Calling graph.getStateHistory with config: {config}, options: {history_options}"
            )

            if hasattr(graph, "aget_state_history"):
                # Try async version first
                async for state in graph.aget_state_history(config, **history_options):
                    thread_state = await _transform_state_snapshot_to_thread_state(
                        state
                    )
                    states.append(thread_state)
            else:
                # Fall back to sync version
                for state in graph.get_state_history(config, **history_options):
                    thread_state = await _transform_state_snapshot_to_thread_state(
                        state
                    )
                    states.append(thread_state)

            logger.info(
                f"[THREAD_HISTORY] Successfully retrieved {len(states)} states from graph.getStateHistory"
            )

        except Exception as state_history_error:
            logger.warning(
                f"[THREAD_HISTORY] graph.getStateHistory failed: {state_history_error}"
            )

            # Fallback: try without options if parameter issues persist
            try:
                logger.info(f"[THREAD_HISTORY] Trying fallback without options")

                if hasattr(graph, "aget_state_history"):
                    count = 0
                    async for state in graph.aget_state_history(config):
                        thread_state = await _transform_state_snapshot_to_thread_state(
                            state
                        )
                        states.append(thread_state)
                        count += 1
                        if count >= (limit or 10):
                            break
                else:
                    count = 0
                    for state in graph.get_state_history(config):
                        thread_state = await _transform_state_snapshot_to_thread_state(
                            state
                        )
                        states.append(thread_state)
                        count += 1
                        if count >= (limit or 10):
                            break

                logger.info(f"[THREAD_HISTORY] Fallback retrieved {len(states)} states")

            except Exception as fallback_error:
                logger.error(f"[THREAD_HISTORY] All methods failed: {fallback_error}")
                # Final fallback: try to get current state
                try:
                    current_state = graph.get_state(config)
                    if current_state:
                        thread_state = await _transform_state_snapshot_to_thread_state(
                            current_state
                        )
                        states.append(thread_state)
                        logger.info(
                            f"[THREAD_HISTORY] Final fallback: returning current state only"
                        )
                except Exception:
                    logger.error(
                        f"[THREAD_HISTORY] Even current state retrieval failed"
                    )
                    return []

        return states

    except Exception as e:
        logger.error(f"Error getting thread history for {thread_id}: {e}")
        # Fallback to empty list instead of error to avoid breaking API
        return []


async def _transform_state_snapshot_to_thread_state(state) -> Dict[str, Any]:
    """Transform LangGraph state snapshot to Agent Protocol ThreadState format."""
    try:
        # Extract values from state - handle both callable and property access
        values = getattr(state, "values", {})
        if callable(values):
            values = values()

        # Ensure values is a dict
        if not isinstance(values, dict):
            values = {}

        # Serialize messages if present
        if "messages" in values and values["messages"]:
            serialized_messages = []
            for msg in values["messages"]:
                if hasattr(msg, "to_dict"):
                    serialized_messages.append(msg.to_dict())
                elif hasattr(msg, "dict"):
                    serialized_messages.append(msg.dict())
                elif hasattr(msg, "__dict__"):
                    # Convert message object to dict
                    msg_dict = {
                        "content": getattr(msg, "content", ""),
                        "type": getattr(msg, "type", "unknown"),
                        "id": getattr(msg, "id", ""),
                        "additional_kwargs": getattr(msg, "additional_kwargs", {}),
                        "response_metadata": getattr(msg, "response_metadata", {}),
                        "name": getattr(msg, "name", None),
                        "example": getattr(msg, "example", False),
                    }
                    # Add tool-related fields if they exist
                    if hasattr(msg, "tool_calls"):
                        msg_dict["tool_calls"] = getattr(msg, "tool_calls", [])
                    if hasattr(msg, "invalid_tool_calls"):
                        msg_dict["invalid_tool_calls"] = getattr(
                            msg, "invalid_tool_calls", []
                        )
                    if hasattr(msg, "usage_metadata"):
                        msg_dict["usage_metadata"] = getattr(
                            msg, "usage_metadata", None
                        )
                    serialized_messages.append(msg_dict)
                else:
                    # Fallback for unknown message types
                    serialized_messages.append(
                        {
                            "content": str(msg),
                            "type": "unknown",
                            "id": "",
                            "additional_kwargs": {},
                            "response_metadata": {},
                        }
                    )
            values["messages"] = serialized_messages

        # Extract next steps - handle both callable and property access
        next_steps = getattr(state, "next", [])
        if callable(next_steps):
            next_steps = next_steps()

        # Ensure next_steps is a list
        if not isinstance(next_steps, list):
            next_steps = []

        # Extract and transform tasks with subgraph support
        tasks = []
        state_tasks = getattr(state, "tasks", [])
        if callable(state_tasks):
            state_tasks = state_tasks()

        # Ensure state_tasks is a list
        if not isinstance(state_tasks, list):
            state_tasks = []
        for task in state_tasks:
            task_dict = {
                "id": getattr(task, "id", ""),
                "name": getattr(task, "name", ""),
                "error": None,
                "interrupts": getattr(task, "interrupts", []),
                "path": getattr(task, "path", []),
                "checkpoint": None,
                "state": None,
                "result": getattr(task, "result", None),
            }

            # Handle task error
            if hasattr(task, "error") and task.error:
                task_dict["error"] = str(task.error)

            # Handle task checkpoint (for subgraphs)
            if (
                hasattr(task, "state")
                and task.state
                and hasattr(task.state, "configurable")
            ):
                task_dict["checkpoint"] = task.state.configurable

            # Handle nested task state (for subgraphs)
            if hasattr(task, "state") and task.state and hasattr(task.state, "values"):
                # Recursively transform nested state
                task_dict["state"] = await _transform_state_snapshot_to_thread_state(
                    task.state
                )

            tasks.append(task_dict)

        # Extract metadata - handle both callable and property access
        metadata = getattr(state, "metadata", {})
        if callable(metadata):
            metadata = metadata()

        # Ensure metadata is a dict
        if not isinstance(metadata, dict):
            metadata = {}

        # Extract checkpoint information
        checkpoint = None
        parent_checkpoint = None

        if hasattr(state, "config") and state.config:
            config = state.config
            if "configurable" in config:
                configurable = config["configurable"]
                checkpoint = {
                    "checkpoint_id": configurable.get("checkpoint_id"),
                    "thread_id": configurable.get("thread_id"),
                    "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                }

        if hasattr(state, "parent_config") and state.parent_config:
            parent_config = state.parent_config
            if "configurable" in parent_config:
                parent_configurable = parent_config["configurable"]
                parent_checkpoint = {
                    "checkpoint_id": parent_configurable.get("checkpoint_id"),
                    "thread_id": parent_configurable.get("thread_id"),
                    "checkpoint_ns": parent_configurable.get("checkpoint_ns", ""),
                }

        # Extract creation time - handle both callable and property access
        created_at = getattr(state, "created_at", None)
        if callable(created_at):
            created_at = created_at()

        # Convert to ISO format if it's a datetime
        if created_at and hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat()

        # Build the thread state
        thread_state = {
            "values": values,
            "next": next_steps,
            "tasks": tasks,
            "metadata": metadata,
            "created_at": created_at,
            "checkpoint": checkpoint,
            "parent_checkpoint": parent_checkpoint,
        }

        return thread_state

    except Exception as e:
        logger.error(f"Error transforming state snapshot: {e}")
        # Return minimal valid structure
        return {
            "values": {},
            "next": [],
            "tasks": [],
            "metadata": {},
            "created_at": None,
            "checkpoint": None,
            "parent_checkpoint": None,
        }


@router.post(
    "/threads/{thread_id}/history",
    response_model=List[Dict[str, Any]],
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def get_thread_history_post(
    thread_id: UUID, body: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Get Thread History Post"""
    try:
        # Get states from LangGraph checkpoints table
        states = await _get_langgraph_thread_history(
            str(thread_id), limit=body.get("limit", 10), before=body.get("before")
        )

        return states

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread history {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads/{thread_id}/runs", response_model=List[Run])
async def list_runs(
    request: Request,
    thread_id: str,
    limit: Optional[int] = Query(10),
    offset: Optional[int] = Query(0),
    status: Optional[str] = Query(None),
):
    """List runs for a thread."""
    try:
        thread = await thread_storage.get(thread_id)
    except HTTPException:
        raise HTTPException(status_code=404, detail="Thread not found")

    runs_data = await Runs.search(
        thread_id,
        {"limit": limit, "offset": offset, "status": status, "metadata": None},
        get_auth_context(request),
    )

    return [convert_run_data_to_model(run_data) for run_data in runs_data]


@router.post("/threads/{thread_id}/runs", response_model=Run)
async def create_run(request: Request, thread_id: str, body: RunCreate):
    """Create a run within a thread, matching JS /threads/:thread_id/runs endpoint."""
    run_data = await create_valid_run(
        thread_id, body, get_auth_context(request), dict(request.headers)
    )

    response = Response()
    response.headers["Content-Location"] = (
        f"/threads/{thread_id}/runs/{run_data.run_id}"
    )

    return convert_run_data_to_model(run_data)


@router.post("/threads/{thread_id}/runs/stream")
async def stream_run(request: Request, thread_id: str, body: RunCreate):
    """Stream a run within a thread"""
    run_data = await create_valid_run(
        thread_id, body, get_auth_context(request), dict(request.headers)
    )

    # Create thread-specific SSE generator
    async def thread_sse_generator():
        cancel_on_disconnect = (
            body.on_disconnect == "cancel" if hasattr(body, "on_disconnect") else False
        )
        last_event_id = "-1" if getattr(run_data.kwargs, "resumable", False) else None

        # Set up stream options
        stream_options = {
            "cancelOnDisconnect": None,  # Will be set below if needed
            "lastEventId": last_event_id,
        }

        if cancel_on_disconnect:
            # Create a cancellation signal
            signal = stream_manager.get_control(run_data.run_id)
            if signal:
                stream_options["cancelOnDisconnect"] = signal

        try:
            async for event in RunStream.join(
                run_data.run_id,
                thread_id,  # Pass the actual thread_id
                stream_options,
                get_auth_context(request),
            ):
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                # Format event for SSE
                # Preserve the original event type instead of defaulting to "message"
                event_type = event.get("event")
                if not event_type:
                    event_type = "message"  # Fallback only if no event type provided

                yield {
                    "id": event.get("id"),
                    "event": event_type,
                    "data": serialize_as_dict(event.get("data", {})),
                }

        except Exception as e:
            logger.error(f"Error in thread SSE stream: {e}")
            yield {
                "event": "error",
                "data": serialize_as_dict({"error": str(e)}),
            }

    # For SSE responses, we don't set Content-Location header to avoid Content-Length conflicts
    return EventSourceResponse(thread_sse_generator(), media_type="text/event-stream")


@router.post("/threads/{thread_id}/runs/wait", response_model=Any)
async def wait_run(request: Request, thread_id: str, body: RunCreate):
    """Wait for a run within a thread to complete"""
    run_data = await create_valid_run(
        thread_id, body, get_auth_context(request), dict(request.headers)
    )

    # Set Content-Location header like
    response = Response()
    response.headers["Content-Location"] = (
        f"/threads/{thread_id}/runs/{run_data.run_id}"
    )

    result = await Runs.join(run_data.run_id, thread_id, get_auth_context(request))
    return result


@router.get("/threads/{thread_id}/runs/{run_id}", response_model=Run)
async def get_run(request: Request, thread_id: str, run_id: str):
    """Get a specific run from a thread."""
    # Check both run and thread exist
    run_data = await Runs.get(run_id, thread_id, get_auth_context(request))
    try:
        thread = await thread_storage.get(thread_id)
    except HTTPException:
        thread = None

    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    return convert_run_data_to_model(run_data)


@router.delete("/threads/{thread_id}/runs/{run_id}", status_code=204)
async def delete_run(request: Request, thread_id: str, run_id: str):
    """Delete a specific run from a thread"""
    await Runs.delete(run_id, thread_id, get_auth_context(request))
    return Response(status_code=204)


@router.get("/threads/{thread_id}/runs/{run_id}/join", response_model=Any)
async def join_run(request: Request, thread_id: str, run_id: str):
    """Join a run and get its final output, matching JS /threads/:thread_id/runs/:run_id/join endpoint."""
    result = await Runs.join(run_id, thread_id, get_auth_context(request))
    return result


@router.get("/threads/{thread_id}/runs/{run_id}/stream")
async def stream_existing_run(
    request: Request,
    thread_id: str,
    run_id: str,
    cancel_on_disconnect: Optional[bool] = Query(False),
):
    """Stream an existing run within a thread, matching JS /threads/:thread_id/runs/:run_id/stream endpoint."""
    last_event_id = request.headers.get("Last-Event-ID")

    async def thread_sse_generator():
        stream_options = {
            "ignore404": False,
            "lastEventId": last_event_id,
        }

        if cancel_on_disconnect:
            signal = stream_manager.get_control(run_id)
            if signal:
                stream_options["cancelOnDisconnect"] = signal

        try:
            async for event in RunStream.join(
                run_id, thread_id, stream_options, get_auth_context(request)
            ):
                if await request.is_disconnected():
                    break

                # Format event for SSE (matching JS: stream.writeSSE({ id, data: serialiseAsDict(data), event }))
                # Preserve the original event type instead of defaulting to "message"
                event_type = event.get("event")
                if not event_type:
                    event_type = "message"  # Fallback only if no event type provided

                yield {
                    "id": event.get("id"),
                    "event": event_type,
                    "data": serialize_as_dict(event.get("data", {})),
                }

        except Exception as e:
            logger.error(f"Error in thread SSE stream: {e}")
            yield {
                "event": "error",
                "data": serialize_as_dict({"error": str(e)}),
            }

    return EventSourceResponse(thread_sse_generator(), media_type="text/event-stream")


@router.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(
    request: Request,
    thread_id: str,
    run_id: str,
    wait: Optional[bool] = Query(False),
    action: Optional[str] = Query("interrupt"),
):
    """Cancel a run, matching JS /threads/:thread_id/runs/:run_id/cancel endpoint."""
    await Runs.cancel(
        thread_id, [run_id], {"action": action}, get_auth_context(request)
    )

    if wait:
        await Runs.join(run_id, thread_id, get_auth_context(request))
        return Response(status_code=204)

    return Response(status_code=202)
