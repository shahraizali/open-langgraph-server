"""Graph loading implementation matching the JavaScript server."""

import json
import importlib.util
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableConfig

from ..storage.checkpoint import checkpointer
from ..storage.ops import store

# UUID namespace for assistant IDs
NAMESPACE_GRAPH = uuid.UUID("6ba7b821-9dad-11d1-80b4-00c04fd430c8")

# Global registries
GRAPHS: Dict[str, Runnable] = {}
GRAPH_SPEC: Dict[str, Dict[str, Any]] = {}


async def validate_assistant_id(assistant_id: str) -> str:
    """
    Validate assistant_id is either a registered graph name or valid assistant UUID.

    Args:
        assistant_id: The assistant ID to validate

    Returns:
        str: The validated assistant ID (kept as-is, no conversion)

    Raises:
        ValueError: If assistant_id is invalid with detailed error message
    """
    # First check if it's a registered graph name
    if assistant_id in GRAPHS:
        return assistant_id  # Return graph name as-is

    # Check if it's a valid UUID format and exists in database
    try:
        # Validate UUID format
        uuid.UUID(assistant_id)

        # Check if assistant exists in database
        from ..storage.postgres_storage import postgres_assistant_storage

        assistant = await postgres_assistant_storage.get(assistant_id)
        if assistant:
            return assistant_id

    except (ValueError, TypeError):
        # Invalid UUID format, continue to error handling
        pass

    # Generate error message with available options
    available_graphs = list(GRAPHS.keys())
    error_msg = f"Invalid assistant: '{assistant_id}'. Must be either:\n"
    error_msg += "- A valid assistant UUID, or\n"
    if available_graphs:
        graphs_list = ", ".join(available_graphs)
        error_msg += f"- One of the registered graphs: {graphs_list}"
    else:
        error_msg += "- One of the registered graphs (none currently registered)"

    raise ValueError(error_msg)


def get_assistant_id(graph_id: str) -> str:
    """Convert graph ID to assistant UUID"""
    if graph_id in GRAPHS:
        return str(uuid.uuid5(NAMESPACE_GRAPH, graph_id))
    return graph_id


def get_graph_id_from_assistant_id(assistant_id: str) -> str:
    """Convert assistant UUID back to graph ID."""
    # Try to find the graph ID that generates this assistant ID
    for graph_id in GRAPHS.keys():
        expected_assistant_id = get_assistant_id(graph_id)
        if expected_assistant_id == assistant_id:
            return graph_id

    # If not found, this might be an auto-created assistant with a different UUID
    # In this case, we need to look up the graph_id from the database
    # For now, return the assistant_id and let the caller handle the error
    return assistant_id  # Return as-is if not found


def load_module_from_path(file_path: str, export_symbol: str = None) -> Any:
    """Load a Python module from file path and get the specified export."""
    try:
        # Resolve the file path
        abs_path = Path(file_path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"Graph file not found: {abs_path}")

        # Determine if we need to add examples directory to sys.path
        examples_dir = None
        if "examples" in str(abs_path):
            # Find the examples directory in the path
            path_parts = abs_path.parts
            try:
                examples_index = path_parts.index("examples")
                examples_dir = Path(*path_parts[: examples_index + 1])
                print(f"Adding examples directory to sys.path: {examples_dir}")
            except ValueError:
                pass

        # Temporarily add examples directory to sys.path if needed
        added_to_path = False
        if examples_dir and examples_dir.exists() and str(examples_dir) not in sys.path:
            sys.path.insert(0, str(examples_dir))
            added_to_path = True
            print(f"Added to sys.path: {examples_dir}")

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("graph_module", abs_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {abs_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            # Clean up sys.path modification
            if added_to_path and str(examples_dir) in sys.path:
                sys.path.remove(str(examples_dir))
                print(f"Removed from sys.path: {examples_dir}")

        # Get the export (default to 'default' or first available)
        if export_symbol:
            if hasattr(module, export_symbol):
                return getattr(module, export_symbol)
            else:
                raise AttributeError(
                    f"Export '{export_symbol}' not found in {abs_path}"
                )
        else:
            # Try to get default export or first available graph-like object
            if hasattr(module, "default"):
                return module.default
            elif hasattr(module, "graph"):
                return module.graph
            elif hasattr(module, "agent"):
                return module.agent
            else:
                # Look for any compiled graph or graph factory
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if hasattr(attr, "invoke") or (
                        hasattr(attr, "compile") and callable(getattr(attr, "compile"))
                    ):
                        return attr

                raise AttributeError(f"No graph export found in {abs_path}")

    except Exception as e:
        raise ImportError(f"Failed to load graph from {file_path}: {e}")


def resolve_graph(spec: str, cwd: str) -> Dict[str, Any]:
    """Resolve a graph specification to a compiled graph."""
    # Parse the spec (format: "path/to/file.py:export_symbol")
    if ":" in spec:
        file_path, export_symbol = spec.split(":", 1)
    else:
        file_path = spec
        export_symbol = None

    # Resolve relative path from cwd
    if not os.path.isabs(file_path):
        file_path = os.path.join(cwd, file_path)

    # Load the graph
    graph_obj = load_module_from_path(file_path, export_symbol)

    # Compile if it's a StateGraph
    if hasattr(graph_obj, "compile") and callable(getattr(graph_obj, "compile")):
        compiled_graph = graph_obj.compile()
    else:
        compiled_graph = graph_obj

    return {
        "sourceFile": file_path,
        "exportSymbol": export_symbol,
        "resolved": compiled_graph,
    }


def load_langgraph_config(config_path: str = "langgraph.json") -> Dict[str, Any]:
    """Load the langgraph.json configuration file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"langgraph.json not found at {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}")


async def register_graphs_from_config(
    config_path: str = "langgraph.json", cwd: str = None
) -> Dict[str, str]:
    """Register graphs from langgraph.json configuration."""
    if cwd is None:
        cwd = os.getcwd()

    # Load configuration
    config = load_langgraph_config(config_path)
    graphs_config = config.get("graphs", {})

    registered_graphs = {}

    for graph_id, spec in graphs_config.items():
        try:
            print(f"Registering graph with id '{graph_id}'")

            # Resolve the graph
            resolved = resolve_graph(spec, cwd)

            # Register the graph
            GRAPHS[graph_id] = resolved["resolved"]
            GRAPH_SPEC[graph_id] = {
                "sourceFile": resolved["sourceFile"],
                "exportSymbol": resolved["exportSymbol"],
            }

            # Generate assistant ID
            assistant_id = get_assistant_id(graph_id)
            registered_graphs[graph_id] = assistant_id
            print(
                f"[ASSISTANT_ID DEBUG] Graph '{graph_id}' mapped to assistant_id: '{assistant_id}'"
            )

            # Create assistant record in database
            from ..storage.postgres_storage import postgres_assistant_storage

            try:
                await postgres_assistant_storage.put(
                    assistant_id,
                    {
                        "graph_id": graph_id,
                        "metadata": {"created_by": "system"},
                        "config": {},
                        "name": graph_id,
                    },
                )
                print(
                    f"Created assistant record for '{graph_id}' -> Assistant ID: '{assistant_id}'"
                )
            except Exception as e:
                print(
                    f"Warning: Failed to create assistant record for '{graph_id}': {e}"
                )
                # This is not critical for graph registration, so continue

            print(
                f"Successfully registered graph '{graph_id}' -> Assistant ID: '{assistant_id}'"
            )

        except Exception as e:
            print(f"Failed to register graph '{graph_id}': {e}")
            raise

    return registered_graphs


def get_graph(
    graph_id: str,
    config: Optional[RunnableConfig] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Runnable:
    """Get a registered graph by ID with checkpoint integration."""
    if graph_id not in GRAPHS:
        raise ValueError(f"Graph '{graph_id}' not found")

    graph = GRAPHS[graph_id]

    # Check if graph is already compiled with checkpointer
    if hasattr(graph, "checkpointer") and graph.checkpointer is not None:
        # Graph already has a checkpointer, use as-is
        return graph

    # Set up checkpoint integration for graphs without checkpointer
    if hasattr(graph, "checkpointer"):
        if options and "checkpointer" in options:
            graph.checkpointer = options["checkpointer"]
        else:
            graph.checkpointer = checkpointer

    # Set up store integration
    if hasattr(graph, "store"):
        if options and "store" in options:
            graph.store = options["store"]
        else:
            # Pass the BaseStore implementation that graphs expect
            graph.store = store.store

    return graph


def get_graph_spec(graph_id: str) -> Dict[str, Any]:
    """Get the specification for a registered graph."""
    if graph_id not in GRAPH_SPEC:
        raise ValueError(f"Graph spec for '{graph_id}' not found")

    return GRAPH_SPEC[graph_id]
