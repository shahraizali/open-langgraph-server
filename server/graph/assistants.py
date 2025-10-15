"""Assistant utilities and integration with graph system."""

from __future__ import annotations

import uuid
import logging
from typing import Dict, Any, Optional

from ..storage.postgres_storage import postgres_assistant_storage as AssistantStorage
from .loader import GRAPHS, GRAPH_SPEC, get_assistant_id as get_graph_assistant_id

logger = logging.getLogger(__name__)


def get_assistant_id(graph_id_or_assistant_id: str) -> str:
    """Convert graph ID or assistant ID to proper assistant UUID.

    This function handles the mapping between graph IDs and assistant IDs,
    similar to the JavaScript implementation.

    Args:
        graph_id_or_assistant_id: Either a graph ID or assistant ID

    Returns:
        The assistant UUID string
    """
    logger.debug(f"Resolving assistant ID for input: '{graph_id_or_assistant_id}'")

    # If it's already a valid UUID, return as-is
    try:
        uuid.UUID(graph_id_or_assistant_id)
        logger.debug(f"Input '{graph_id_or_assistant_id}' is already a valid UUID")
        return graph_id_or_assistant_id
    except ValueError:
        logger.debug(f"Input '{graph_id_or_assistant_id}' is not a valid UUID")

    # If it's a graph ID that exists, generate assistant ID
    if graph_id_or_assistant_id in GRAPHS:
        assistant_id = get_graph_assistant_id(graph_id_or_assistant_id)
        logger.debug(
            f"Graph ID '{graph_id_or_assistant_id}' mapped to assistant ID: '{assistant_id}'"
        )
        return assistant_id

    # Otherwise, return as-is (let downstream handle validation)
    logger.debug(f"No graph found for '{graph_id_or_assistant_id}', returning as-is")
    return graph_id_or_assistant_id


async def create_default_assistants() -> Dict[str, str]:
    """Create default assistants for all loaded graphs.

    Returns:
        Dictionary mapping graph_id to assistant_id for created assistants
    """
    logger.info(f"Creating default assistants for {len(GRAPHS)} loaded graphs")
    created_assistants = {}

    for graph_id in GRAPHS.keys():
        try:
            assistant_id = get_graph_assistant_id(graph_id)
            logger.info(
                f"Processing graph '{graph_id}' with assistant ID: '{assistant_id}'"
            )

            # Check if assistant already exists
            try:
                existing_assistant = await AssistantStorage.get(assistant_id)
                logger.info(
                    f"Assistant {assistant_id} for graph {graph_id} already exists"
                )
                created_assistants[graph_id] = assistant_id
                continue
            except Exception as e:
                # Assistant doesn't exist, create it
                logger.debug(
                    f"Assistant {assistant_id} doesn't exist, will create: {e}"
                )

            # Create default assistant for this graph
            assistant_data = {
                "graph_id": graph_id,
                "name": graph_id,
                "config": {},
                "metadata": {"created_by": "system"},
                "if_exists": "do_nothing",
            }

            logger.debug(f"Creating assistant with data: {assistant_data}")
            assistant = await AssistantStorage.put(assistant_id, assistant_data)
            created_assistants[graph_id] = assistant_id

            logger.info(
                f"Successfully created default assistant {assistant_id} for graph {graph_id}"
            )

        except Exception as e:
            logger.error(
                f"Failed to create assistant for graph {graph_id}: {e}", exc_info=True
            )

    logger.info(f"Created default assistants: {created_assistants}")
    return created_assistants


async def get_assistant_for_graph(graph_id: str) -> Optional[str]:
    """Get the assistant ID for a given graph ID.

    Args:
        graph_id: The graph ID to find assistant for

    Returns:
        Assistant ID if found, None otherwise
    """
    if graph_id not in GRAPHS:
        return None

    assistant_id = get_graph_assistant_id(graph_id)

    try:
        await AssistantStorage.get(assistant_id)
        return assistant_id
    except Exception:
        return None


def get_runnable_config(assistant_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert assistant config to runnable config format.

    Args:
        assistant_config: Assistant configuration dictionary

    Returns:
        Runnable configuration for graph execution
    """
    if not assistant_config:
        return {}

    runnable_config = {}

    # Map assistant config fields to runnable config
    if "tags" in assistant_config:
        runnable_config["tags"] = assistant_config["tags"]

    if "recursion_limit" in assistant_config:
        runnable_config["recursionLimit"] = assistant_config["recursion_limit"]

    if "configurable" in assistant_config:
        runnable_config["configurable"] = assistant_config["configurable"]

    # Additional fields that might be in assistant config
    for field in ["metadata", "run_name", "max_concurrency", "run_id"]:
        if field in assistant_config:
            # Convert snake_case to camelCase for some fields
            if field == "run_name":
                runnable_config["runName"] = assistant_config[field]
            elif field == "max_concurrency":
                runnable_config["maxConcurrency"] = assistant_config[field]
            elif field == "run_id":
                runnable_config["runId"] = assistant_config[field]
            else:
                runnable_config[field] = assistant_config[field]

    return runnable_config


def validate_assistant_graph_exists(graph_id: str) -> bool:
    """Validate that a graph exists for the given graph ID.

    Args:
        graph_id: The graph ID to validate

    Returns:
        True if graph exists, False otherwise
    """
    return graph_id in GRAPHS


def get_available_graphs() -> Dict[str, Dict[str, Any]]:
    """Get information about all available graphs.

    Returns:
        Dictionary mapping graph_id to graph metadata
    """
    available_graphs = {}

    for graph_id in GRAPHS.keys():
        spec = GRAPH_SPEC.get(graph_id)
        available_graphs[graph_id] = {
            "graph_id": graph_id,
            "assistant_id": get_graph_assistant_id(graph_id),
            "source_file": spec.get("sourceFile") if spec else None,
            "export_symbol": spec.get("exportSymbol") if spec else None,
        }

    return available_graphs
