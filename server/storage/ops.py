from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator, Literal
from uuid import UUID
from datetime import datetime
import traceback

from fastapi import HTTPException

from ..schemas import RunStatus, ThreadStatus
from .streaming import stream_manager, RunStream, send_control_message
from .checkpoint import checkpointer
from .postgres_storage import (
    postgres_thread_storage,
    postgres_run_storage,
    postgres_assistant_storage,
    postgres_store_storage,
    PostgresStore,
    RunData as PgRunData,
    Assistant as PgAssistant,
)
from ..utils.serde import serialize_error
from .schemas import RunKwargs
from ..utils.auth import (
    AuthContext,
    handle_auth_event,
    is_auth_matching,
)

logger = logging.getLogger(__name__)


# Type definitions
Metadata = Dict[str, Any]
MultitaskStrategy = Literal["reject", "rollback", "interrupt", "enqueue"]
OnConflictBehavior = Literal["raise", "do_nothing"]
IfNotExists = Literal["create", "reject"]


class RunnableConfig(Dict[str, Any]):
    """Runnable configuration"""

    pass


class Assistant:
    """Assistant data structure"""

    def __init__(
        self,
        assistant_id: str,
        name: Optional[str],
        graph_id: str,
        created_at: datetime,
        updated_at: datetime,
        version: int,
        config: RunnableConfig,
        metadata: Metadata,
    ):
        self.assistant_id = assistant_id
        self.name = name
        self.graph_id = graph_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.version = version
        self.config = config
        self.metadata = metadata





class RunData:
    """Run data structure"""

    def __init__(
        self,
        run_id: str,
        thread_id: str,
        assistant_id: str,
        created_at: datetime,
        updated_at: datetime,
        status: RunStatus,
        metadata: Metadata,
        kwargs: RunKwargs,
        multitask_strategy: MultitaskStrategy,
    ):
        self.run_id = run_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.status = status
        self.metadata = metadata
        self.kwargs = kwargs
        self.multitask_strategy = multitask_strategy


class ThreadData:
    """Thread data structure"""

    def __init__(
        self,
        thread_id: str,
        created_at: datetime,
        updated_at: datetime,
        metadata: Metadata,
        config: Optional[RunnableConfig],
        status: ThreadStatus,
        values: Optional[Dict[str, Any]] = None,
        interrupts: Optional[Dict[str, Any]] = None,
    ):
        self.thread_id = thread_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.metadata = metadata
        self.config = config
        self.status = status
        self.values = values
        self.interrupts = interrupts


class Store:
    """PostgreSQL-backed store"""

    def __init__(self):
        self.retry_counter: Dict[str, int] = {}
        # Use PostgreSQL storage
        self.threads = postgres_thread_storage
        self.runs_storage = postgres_run_storage
        self.assistants_storage = postgres_assistant_storage
        self.store_storage = postgres_store_storage
        # Add BaseStore implementation for LangGraph compatibility
        self.store = PostgresStore(postgres_store_storage)
        self.runs: Dict[str, Any] = {}
        self.assistants = postgres_assistant_storage
        self.assistant_versions: List[Any] = []


# Global store instance
store = Store()


class RunsImpl:
    """
    Runs management class
    Handles run lifecycle, streaming, cancellation, and multitask strategies.
    """

    @staticmethod
    async def next() -> AsyncGenerator[Dict[str, Any], None]:
        """
        Get next pending runs for execution.
        """
        now = datetime.utcnow()
        pending_run_ids = [
            run.run_id
            for run in store.runs.values()
            if run.status == RunStatus.pending and run.created_at <= now
        ]

        # Sort by creation time
        pending_run_ids.sort(key=lambda rid: store.runs[rid].created_at)

        for run_id in pending_run_ids:
            if stream_manager.is_locked(run_id):
                continue

            try:
                signal = stream_manager.lock(run_id)
                run = store.runs.get(run_id)

                if not run or run.status != RunStatus.pending:
                    continue

                try:
                    thread = await store.threads.get(run.thread_id)
                except HTTPException:
                    logger.warning(f"Missing thread {run.thread_id} for run {run_id}")
                    continue

                # Check if there are other running runs on the same thread
                has_running = any(
                    r.thread_id == run.thread_id and str(r.status) == "running"
                    for r in store.runs.values()
                )
                if has_running:
                    continue

                # Update retry counter and status
                store.retry_counter[run_id] = store.retry_counter.get(run_id, 0) + 1
                run.status = "running"
                run.updated_at = datetime.utcnow()

                yield {
                    "run": run,
                    "attempt": store.retry_counter[run_id],
                    "signal": signal,
                }

            finally:
                stream_manager.unlock(run_id)

    @staticmethod
    async def put(
        run_id: str,
        assistant_id: str,
        kwargs: RunKwargs,
        options: Dict[str, Any],
        auth: Optional[AuthContext] = None,
    ) -> tuple[Optional[PgRunData], List[PgRunData]]:
        """
        Create a new run with complex multitask handling.
        """
        # Get assistant from PostgreSQL or graph registry
        assistant = await store.assistants_storage.get(assistant_id)

        # If not found in database, check if it's a registered graph name
        if not assistant:
            # Import GRAPHS locally to avoid circular import
            from ..graph.loader import GRAPHS

            if assistant_id in GRAPHS:
                # Create virtual assistant object for graph
                assistant = PgAssistant(
                    assistant_id=assistant_id,
                    name=assistant_id,
                    graph_id=assistant_id,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    version=1,
                    config={},
                    metadata={"created_by": "system", "type": "graph"},
                )

        if not assistant:
            # Import GRAPHS locally for error message
            from ..graph.loader import GRAPHS

            # Generate error message with available options
            available_graphs = list(GRAPHS.keys())
            error_msg = f'Invalid assistant: "{assistant_id}". Must be either:\n'
            error_msg += "- A valid assistant UUID, or\n"
            if available_graphs:
                graphs_list = ", ".join(available_graphs)
                error_msg += f"- One of the registered graphs: {graphs_list}"
            else:
                error_msg += (
                    "- One of the registered graphs (none currently registered)"
                )

            raise HTTPException(status_code=400, detail=error_msg)

        # Extract options
        if_not_exists = options.get("if_not_exists", "reject")
        multitask_strategy = options.get("multitask_strategy", "reject")
        after_seconds = options.get("after_seconds", 0)
        status = options.get("status", "pending")
        thread_id = options.get("thread_id")
        user_id = options.get("user_id")

        # Handle auth events
        filters, mutable = await handle_auth_event(
            auth,
            "threads:create_run",
            {
                "thread_id": thread_id,
                "assistant_id": assistant_id,
                "run_id": run_id,
                "status": status,
                "metadata": options.get("metadata", {}),
                "prevent_insert_if_inflight": options.get("prevent_insert_if_inflight"),
                "multitask_strategy": multitask_strategy,
                "if_not_exists": if_not_exists,
                "after_seconds": after_seconds,
                "kwargs": kwargs.__dict__ if hasattr(kwargs, "__dict__") else kwargs,
            },
        )

        metadata = mutable.get("metadata", {})
        config = kwargs.config or {}

        # Use PostgreSQL storage to create the run
        result = await store.runs_storage.put(
            run_id, assistant_id, kwargs, options, auth
        )

        if result and result[0]:  # result is a tuple (run_data, list_of_runs)
            run_data = result[0]
            # Convert PgRunData to RunData for in-memory storage
            run = RunData(
                run_id=run_data.run_id,
                thread_id=run_data.thread_id,
                assistant_id=run_data.assistant_id,
                created_at=run_data.created_at,
                updated_at=run_data.updated_at,
                status=run_data.status,
                metadata=run_data.metadata,
                kwargs=run_data.kwargs,
                multitask_strategy=run_data.multitask_strategy,
            )
            store.runs[run_id] = run

            # Start execution in background
            if status == "pending":
                logger.info(f"Starting execution for run {run_id}")
                asyncio.create_task(RunsImpl._execute_run(run_id))

        return result



    @staticmethod
    async def _execute_run(run_id: str):
        """Execute a run with clean, modular architecture."""
        from ..stream.processor import StreamProcessor
        from ..execution.config import ExecutionConfigBuilder
        from ..execution.core import GraphExecutor
        
        run = store.runs.get(run_id)
        if not run:
            logger.error(f"Run {run_id} not found")
            return

        try:
            # Get assistant (with graph name fallback)
            assistant = await RunsImpl._get_or_create_assistant(run.assistant_id)
            if not assistant:
                logger.error(f"Assistant {run.assistant_id} not found for run {run_id}")
                return

            # Get graph
            graph_id = assistant.graph_id
            from ..graph.loader import get_graph
            graph = get_graph(
                graph_id,
                run.kwargs.config,
                {"checkpointer": checkpointer, "store": store.store},
            )

            # Prepare execution components
            input_data = GraphExecutor.prepare_input_data(run.kwargs.input)
            config = ExecutionConfigBuilder.build_execution_config(
                run_id, run.thread_id, run.kwargs.config
            )
            stream_mode = ExecutionConfigBuilder.prepare_stream_mode(run.kwargs.stream_mode)

            logger.info(f"Executing graph {graph_id} for run {run_id}")

            # Execute with streaming
            stream_processor = StreamProcessor(run_id, run.kwargs.resumable)
            await stream_processor.process_stream_events(graph, input_data, config, stream_mode)

            # Get final state and update database
            final_values = await GraphExecutor.execute_graph(
                graph, input_data, config, run_id, run.thread_id
            )

            # Publish final events
            await stream_processor.publish_final_values(final_values or {})
            await stream_processor.publish_completion("success")

            # Update run status
            run.status = RunStatus.success
            run.updated_at = datetime.utcnow()
            await GraphExecutor.update_run_status(run_id, "success")

        except Exception as e:
            logger.error(f"Error executing run {run_id}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            # Handle error
            run.status = RunStatus.error
            run.updated_at = datetime.utcnow()
            await GraphExecutor.update_run_status(run_id, "error")

            # Publish error
            stream_processor = StreamProcessor(run_id, run.kwargs.resumable)
            await stream_processor.publish_error(serialize_error(e))

        finally:
            # Update thread status and cleanup
            await RunsImpl._update_thread_status(run.thread_id)
            send_control_message(run_id, "done")

    @staticmethod
    async def _get_or_create_assistant(assistant_id: str):
        """Get assistant from database or create virtual assistant for graph."""
        # Try to get from database first
        assistant = await store.assistants_storage.get(assistant_id)
        
        if not assistant:
            # Check if it's a registered graph name
            from ..graph.loader import GRAPHS
            if assistant_id in GRAPHS:
                assistant = PgAssistant(
                    assistant_id=assistant_id,
                    name=assistant_id,
                    graph_id=assistant_id,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    version=1,
                    config={},
                    metadata={"created_by": "system", "type": "graph"},
                )
        
        return assistant


    @staticmethod
    async def _update_thread_status(thread_id: str):
        """Update thread status based on remaining runs."""
        try:
            thread = await store.threads.get(thread_id)
        except HTTPException:
            return

        # Check for pending/running runs
        has_active_runs = any(
            run.thread_id == thread_id and str(run.status) in ["pending", "running"]
            for run in store.runs.values()
        )

        # Check for errors
        has_error_runs = any(
            run.thread_id == thread_id and str(run.status) == "error"
            for run in store.runs.values()
        )

        if has_error_runs:
            thread.status = ThreadStatus.error
        elif has_active_runs:
            thread.status = ThreadStatus.busy
        else:
            thread.status = ThreadStatus.idle

        thread.updated_at = datetime.utcnow().isoformat() + "Z"

    @staticmethod
    async def get(
        run_id: str, thread_id: Optional[str], auth: Optional[AuthContext] = None
    ) -> Optional[PgRunData]:
        """Get a run by ID with auth checking."""
        return await store.runs_storage.get(run_id, thread_id, auth)

    @staticmethod
    async def delete(
        run_id: str, thread_id: Optional[str], auth: Optional[AuthContext] = None
    ) -> Optional[str]:
        """Delete a run."""
        result = await store.runs_storage.delete(run_id, thread_id, auth)
        if result:
            stream_manager.cleanup_run(run_id)
        return result

    @staticmethod
    async def cancel(
        thread_id: Optional[str],
        run_ids: List[str],
        options: Dict[str, Any],
        auth: Optional[AuthContext] = None,
    ):
        """Cancel runs with rollback/interrupt support."""
        action = options.get("action", "interrupt")

        filters, _ = await handle_auth_event(
            auth,
            "threads:update",
            {
                "thread_id": thread_id,
                "action": action,
                "metadata": {"run_ids": run_ids, "status": "pending"},
            },
        )

        found_runs = 0
        delete_tasks = []

        for run_id in run_ids:
            run = store.runs.get(run_id)
            if not run or (thread_id is not None and run.thread_id != thread_id):
                continue

            if filters is not None:
                thread = store.threads.get(run.thread_id)
                if thread and not is_auth_matching(thread.metadata, filters):
                    continue

            found_runs += 1

            # Send cancellation signal
            control = stream_manager.get_control(run_id)
            if control:
                control.abort(action)

            if str(run.status) == "pending":
                if control or action != "rollback":
                    run.status = RunStatus.interrupted
                    run.updated_at = datetime.utcnow()

                    # Update thread status
                    try:
                        thread = await store.threads.get(run.thread_id)
                        thread.status = ThreadStatus.idle
                        thread.updated_at = datetime.utcnow().isoformat() + "Z"
                    except HTTPException:
                        pass
                else:
                    # Eagerly delete unscheduled run with rollback
                    logger.info(
                        f"Eagerly deleting unscheduled run {run_id} with rollback"
                    )
                    delete_tasks.append(RunsImpl.delete(run_id, thread_id, auth))
            else:
                logger.warning(
                    f"Attempted to cancel non-pending run {run_id} with status {run.status}"
                )

        # Execute deletes
        if delete_tasks:
            await asyncio.gather(*delete_tasks, return_exceptions=True)

        if found_runs != len(run_ids):
            raise HTTPException(status_code=404, detail="Run not found")

        logger.info(
            f"Cancelled runs {run_ids} on thread {thread_id} with action {action}"
        )

    @staticmethod
    async def search(
        thread_id: str, options: Dict[str, Any], auth: Optional[AuthContext] = None
    ) -> List[PgRunData]:
        """Search runs in a thread."""
        return await store.runs_storage.search(thread_id, options, auth)

    @staticmethod
    async def wait(
        run_id: str, thread_id: Optional[str], auth: Optional[AuthContext] = None
    ) -> Any:
        """Wait for a run to complete and return final result."""
        last_chunk = None

        async for event in RunStream.join(
            run_id,
            thread_id,
            {"ignore404": thread_id is None, "lastEventId": None},
            auth,
        ):
            if event.get("event") == "values":
                last_chunk = event.get("data")
            elif event.get("event") == "error":
                last_chunk = {"__error__": event.get("data")}

        return last_chunk

    @staticmethod
    async def join(
        run_id: str, thread_id: str, auth: Optional[AuthContext] = None
    ) -> Any:
        """Join a run and get its final output."""
        # Ensure thread exists
        try:
            thread = await store.threads.get(thread_id)
        except HTTPException:
            raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

        last_chunk = await RunsImpl.wait(run_id, thread_id, auth)
        if last_chunk is not None:
            return last_chunk

        # Return thread values if no result
        return thread.values or {}


# Global runs instance
Runs = RunsImpl()

# Export run_storage for compatibility with streaming.py
run_storage = Runs


# Create default assistant for testing
async def ensure_default_assistant():
    """Create a default assistant if none exists."""
    if not store.assistants:
        assistant_id = "agent"  # Changed from "default" to "agent"
        now = datetime.utcnow()

        assistant = Assistant(
            assistant_id=assistant_id,
            name="Default Assistant",
            graph_id="default_graph",
            created_at=now,
            updated_at=now,
            version=1,
            config={},
            metadata={"created_by": "system"},
        )

        store.assistants[assistant_id] = assistant
        store.assistant_versions.append(assistant)
        logger.info("Created default assistant")


# Note: ensure_default_assistant() should be called during app startup
