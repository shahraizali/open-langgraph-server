"""Graph resolution logic for dynamic module loading."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Union

from langgraph.pregel import Pregel
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig


class GraphSpec:
    """Specification for a graph location."""

    def __init__(self, source_file: str, export_symbol: str):
        self.source_file = source_file
        self.export_symbol = export_symbol

    def __repr__(self) -> str:
        return f"GraphSpec(source_file='{self.source_file}', export_symbol='{self.export_symbol}')"


class GraphResolver:
    """Handles dynamic loading and resolution of graphs from Python modules."""

    @staticmethod
    def parse_graph_spec(spec: str) -> tuple[str, str]:
        """Parse a graph specification string into file path and export symbol.

        Args:
            spec: Graph specification in format 'file_path:export_symbol'

        Returns:
            Tuple of (file_path, export_symbol)

        Raises:
            ValueError: If spec format is invalid
        """
        if ":" not in spec:
            raise ValueError(
                f"Graph spec '{spec}' must be in format 'file_path:export_symbol'"
            )

        parts = spec.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Graph spec '{spec}' must be in format 'file_path:export_symbol'"
            )

        file_path, export_symbol = parts
        if not file_path.strip():
            raise ValueError(f"File path cannot be empty in spec '{spec}'")
        if not export_symbol.strip():
            raise ValueError(f"Export symbol cannot be empty in spec '{spec}'")

        return file_path.strip(), export_symbol.strip()

    @staticmethod
    def resolve_graph(
        spec: str, base_path: Path, only_file_presence: bool = False
    ) -> Union[tuple[GraphSpec, None], tuple[GraphSpec, Any]]:
        """Resolve a graph from a specification string.

        Args:
            spec: Graph specification in format 'file_path:export_symbol'
            base_path: Base directory path for resolving relative paths
            only_file_presence: If True, only check if file exists, don't load

        Returns:
            Tuple of (GraphSpec, resolved_graph) or (GraphSpec, None) if only_file_presence=True

        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ImportError: If the module cannot be imported
            AttributeError: If the export symbol doesn't exist in the module
            ValueError: If the resolved object is not a valid graph
        """
        file_path, export_symbol = GraphResolver.parse_graph_spec(spec)

        # Resolve the file path relative to base_path
        source_file = base_path / file_path
        if not source_file.exists():
            raise FileNotFoundError(f"Graph file not found: {source_file}")

        graph_spec = GraphSpec(str(source_file), export_symbol)

        if only_file_presence:
            return graph_spec, None

        # Load the module dynamically
        try:
            module = GraphResolver._load_module(source_file)
        except Exception as e:
            raise ImportError(f"Failed to load module from {source_file}: {e}")

        # Get the exported symbol
        if not hasattr(module, export_symbol):
            available_attrs = [attr for attr in dir(module) if not attr.startswith("_")]
            raise AttributeError(
                f"Module {source_file} has no attribute '{export_symbol}'. "
                f"Available attributes: {available_attrs}"
            )

        graph_obj = getattr(module, export_symbol)

        # Validate and resolve the graph object
        try:
            resolved_graph = GraphResolver._resolve_graph_object(graph_obj)
        except Exception as e:
            raise ValueError(
                f"Failed to resolve graph object '{export_symbol}' from {source_file}: {e}"
            )

        return graph_spec, resolved_graph

    @staticmethod
    def _load_module(file_path: Path) -> Any:
        """Load a Python module from a file path."""
        spec = importlib.util.spec_from_file_location("graph_module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _resolve_graph_object(
        graph_obj: Any,
    ) -> Union[Pregel, Callable[[RunnableConfig], Pregel]]:
        """Resolve a graph object into a compiled graph or graph factory.

        Args:
            graph_obj: The object to resolve (could be StateGraph, Pregel, or callable)

        Returns:
            Either a Pregel or a callable that returns a Pregel

        Raises:
            ValueError: If the object is not a valid graph type
        """
        # If it's already a Pregel, return it
        if isinstance(graph_obj, Pregel):
            return graph_obj

        # If it's a StateGraph, compile it
        if isinstance(graph_obj, StateGraph):
            return graph_obj.compile()

        # If it's a callable, check if it's a graph factory
        if callable(graph_obj):
            # Check if it's a graph factory function
            sig = inspect.signature(graph_obj)

            # If it takes a config parameter, treat it as a graph factory
            if "config" in sig.parameters or len(sig.parameters) > 0:

                def graph_factory(config: RunnableConfig) -> Pregel:
                    result = graph_obj(config)
                    return GraphResolver._resolve_graph_object(result)

                return graph_factory

            # If it's a no-arg callable, call it and resolve the result
            else:
                try:
                    result = graph_obj()
                    return GraphResolver._resolve_graph_object(result)
                except Exception as e:
                    raise ValueError(f"Failed to call graph factory: {e}")

        raise ValueError(
            f"Invalid graph object type: {type(graph_obj)}. "
            f"Expected StateGraph, Pregel, or callable."
        )

    @staticmethod
    def validate_graph_files(
        graph_specs: Dict[str, str], base_path: Path
    ) -> Dict[str, GraphSpec]:
        """Validate that all graph files exist and return GraphSpec objects.

        Args:
            graph_specs: Dictionary mapping graph_id to graph specification
            base_path: Base directory path for resolving relative paths

        Returns:
            Dictionary mapping graph_id to GraphSpec objects

        Raises:
            FileNotFoundError: If any graph file doesn't exist
            ValueError: If any graph specification is invalid
        """
        validated_specs = {}

        for graph_id, spec in graph_specs.items():
            try:
                graph_spec, _ = GraphResolver.resolve_graph(
                    spec, base_path, only_file_presence=True
                )
                validated_specs[graph_id] = graph_spec
            except Exception as e:
                raise ValueError(f"Invalid graph specification for '{graph_id}': {e}")

        return validated_specs
