"""Core execution module for LangGraph graph execution."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..storage.database import get_database
from ..utils.serde import serialize_as_dict

logger = logging.getLogger(__name__)


class GraphExecutor:
    """Handles graph execution and state management."""

    @staticmethod
    async def execute_graph(
        graph,
        input_data: Dict[str, Any],
        config: Dict[str, Any],
        run_id: str,
        thread_id: str
    ) -> Optional[Dict[str, Any]]:
        """Execute graph and return final state.
        
        Args:
            graph: The compiled LangGraph
            input_data: Input data for execution
            config: Execution configuration
            run_id: Run ID
            thread_id: Thread ID
            
        Returns:
            Final state values or None if execution failed
        """
        logger.info(f"Executing graph for run {run_id}")

        try:
            # Get final state after execution
            latest_state = graph.get_state(config)
            if latest_state and hasattr(latest_state, "values"):
                final_values = latest_state.values
                # Update thread values in database
                await GraphExecutor.update_thread_values(thread_id, final_values)
                return final_values
            else:
                logger.warning(f"No final state available for run {run_id}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to get final state for run {run_id}: {e}")
            return None

    @staticmethod
    async def update_thread_values(thread_id: str, values: Dict[str, Any]) -> None:
        """Update thread values in the database after run completion.
        
        Args:
            thread_id: Thread ID
            values: Values to update
        """
        try:
            db = await get_database()
            async with db.get_connection() as conn:
                await conn.execute(
                    """
                    UPDATE threads
                    SET values = $2, updated_at = NOW()
                    WHERE thread_id = $1
                    """,
                    UUID(thread_id),
                    serialize_as_dict(values),
                )
                logger.info(f"Updated thread {thread_id} values in database")
        except Exception as e:
            logger.error(f"Failed to update thread {thread_id} values in database: {e}")

    @staticmethod
    async def save_thread_state(
        thread_id: str,
        checkpoint_id: str,
        values: Dict[str, Any],
        next_steps: List[str],
        metadata: Dict[str, Any],
        parent_checkpoint_id: Optional[str] = None,
        tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Save thread state data to the thread_states table.
        
        Args:
            thread_id: Thread ID
            checkpoint_id: Checkpoint ID
            values: State values
            next_steps: Next steps list
            metadata: State metadata
            parent_checkpoint_id: Optional parent checkpoint ID
            tasks: Optional tasks list
        """
        try:
            db = await get_database()
            async with db.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO thread_states
                    (thread_id, checkpoint_id, values, next_steps, metadata, parent_checkpoint_id, tasks)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    UUID(thread_id),
                    UUID(checkpoint_id),
                    json.dumps(values),
                    json.dumps(next_steps),
                    json.dumps(metadata),
                    UUID(parent_checkpoint_id) if parent_checkpoint_id else None,
                    json.dumps(tasks or []),
                )
                logger.info(f"Saved thread state for thread {thread_id}, checkpoint {checkpoint_id}")
        except Exception as e:
            logger.error(f"Failed to save thread state for thread {thread_id}: {e}")

    @staticmethod
    async def update_run_status(run_id: str, status: str) -> None:
        """Update run status in the database.
        
        Args:
            run_id: Run ID
            status: New status
        """
        try:
            db = await get_database()
            async with db.get_connection() as conn:
                await conn.execute(
                    """
                    UPDATE runs
                    SET status = $2, updated_at = NOW()
                    WHERE run_id = $1
                    """,
                    UUID(run_id),
                    status,
                )
                logger.info(f"Updated run {run_id} status to {status} in database")
        except Exception as e:
            logger.error(f"Failed to update run {run_id} status in database: {e}")

    @staticmethod
    def prepare_input_data(input_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare input data for graph execution.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Prepared input data with defaults
        """
        if not input_data:
            input_data = {}

        # Ensure messages field exists (generic requirement for LangGraph)
        if "messages" not in input_data:
            input_data["messages"] = []

        return input_data

    @staticmethod
    def get_graph_config(assistant, run_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get graph configuration from assistant and run config.
        
        Args:
            assistant: Assistant object
            run_config: Optional run configuration
            
        Returns:
            Graph configuration
        """
        # Start with assistant config
        graph_config = {}
        if hasattr(assistant, 'config') and assistant.config:
            graph_config.update(assistant.config)

        # Merge run-specific config
        if run_config:
            graph_config.update(run_config)

        return graph_config