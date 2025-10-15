"""Schema extraction and management for LangGraph instances."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union, get_type_hints, get_origin, get_args
import json

from langgraph.pregel import Pregel
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class GraphSchema:
    """Represents the schema of a LangGraph."""

    def __init__(
        self,
        state: Optional[Dict[str, Any]] = None,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.state = state
        self.input = input
        self.output = output
        self.config = config

    def to_dict(self) -> Dict[str, Any]:
        """Convert the schema to a dictionary."""
        return {
            "state": self.state,
            "input": self.input,
            "output": self.output,
            "config": self.config,
        }

    def is_empty(self) -> bool:
        """Check if the schema is empty."""
        return all(
            schema is None
            for schema in [self.state, self.input, self.output, self.config]
        )


class SchemaExtractor:
    """Extracts JSON schemas from LangGraph instances."""

    @staticmethod
    def extract_runtime_schema(graph: Pregel) -> Optional[GraphSchema]:
        """Extract schema from a compiled graph at runtime.

        Args:
            graph: The compiled graph to extract schema from

        Returns:
            GraphSchema object or None if extraction fails
        """
        try:
            # Extract schemas using Python LangGraph's built-in methods
            input_schema = None
            output_schema = None
            state_schema = None
            config_schema = None

            # Try to get input schema - this should match the state schema
            try:
                logger.debug(f"Graph has input_schema attribute: {hasattr(graph, 'input_schema')}")
                if hasattr(graph, "input_schema"):
                    logger.debug(f"Graph input_schema value: {graph.input_schema}")
                    
                if hasattr(graph, "input_schema") and graph.input_schema:
                    input_schema = SchemaExtractor._convert_to_json_schema(graph.input_schema)
                    if input_schema:
                        input_schema["title"] = f"{graph.name}_input" if hasattr(graph, 'name') and graph.name else "input"
                        logger.debug(f"Extracted input schema: {input_schema is not None}")
            except Exception as e:
                logger.error(f"Failed to extract input schema: {e}")
                import traceback
                logger.error(traceback.format_exc())

            # Try to get output schema - this should also match the state schema  
            try:
                if hasattr(graph, "output_schema") and graph.output_schema:
                    output_schema = SchemaExtractor._convert_to_json_schema(graph.output_schema)
                elif input_schema:
                    # Output schema is typically the same as input schema for stateful graphs
                    import copy
                    output_schema = copy.deepcopy(input_schema)
                    if output_schema:
                        output_schema["title"] = f"{graph.name}_output" if hasattr(graph, 'name') and graph.name else "output"
            except Exception as e:
                logger.debug(f"Failed to extract output schema: {e}")

            # Extract state schema from graph's input schema
            try:
                if hasattr(graph, "input_schema") and graph.input_schema:
                    state_schema = SchemaExtractor._convert_to_json_schema(graph.input_schema)
                    if state_schema:
                        # For state schema, make messages optional (nullable)
                        if "properties" in state_schema and "messages" in state_schema["properties"]:
                            # Make messages nullable for state schema
                            state_schema["properties"]["messages"] = {
                                **state_schema["properties"]["messages"],
                                "default": None
                            }
                            # Remove from required if present
                            if "required" in state_schema and "messages" in state_schema["required"]:
                                state_schema["required"] = [r for r in state_schema["required"] if r != "messages"]
                        
                        state_schema["title"] = f"{graph.name}_state" if hasattr(graph, 'name') and graph.name else "state"
            except Exception as e:
                logger.debug(f"Failed to extract state schema: {e}")

            # Extract config schema
            try:
                if hasattr(graph, "config_schema") and graph.config_schema:
                    config_schema = SchemaExtractor._convert_to_json_schema(graph.config_schema)
                elif hasattr(graph, "get_config_schema"):
                    config_schema_type = graph.get_config_schema()
                    if config_schema_type:
                        config_schema = SchemaExtractor._convert_to_json_schema(config_schema_type)
            except Exception as e:
                logger.debug(f"Failed to extract config schema: {e}")

            schema = GraphSchema(
                state=state_schema,
                input=input_schema,
                output=output_schema,
                config=config_schema,
            )

            return schema if not schema.is_empty() else None

        except Exception as e:
            logger.error(f"Failed to extract runtime schema: {e}")
            return None

    @staticmethod
    def _extract_state_schema_from_builder(builder) -> Optional[Dict[str, Any]]:
        """Extract state schema from graph builder using Pydantic model.

        Args:
            builder: The graph builder object

        Returns:
            State schema as JSON schema dict or None
        """
        try:
            # Get the main state schema from the builder
            if hasattr(builder, "schema") and builder.schema:
                # Use the Pydantic model's JSON schema
                if hasattr(builder.schema, "model_json_schema"):
                    return builder.schema.model_json_schema()

            # Fallback: try to get from schemas dict
            if hasattr(builder, "schemas") and builder.schemas:
                # Get the main state schema (usually the first one)
                state_schemas = list(builder.schemas.values())
                if not state_schemas:
                    return None

                # Use the first schema definition
                schema_def = state_schemas[0]

                # Build schema from channel definitions
                properties = {}
                required = []

                for channel_name, channel in schema_def.items():
                    if hasattr(channel, "UpdateType") and channel.UpdateType:
                        try:
                            # Get schema from channel type
                            if hasattr(channel.UpdateType, "model_json_schema"):
                                properties[channel_name] = (
                                    channel.UpdateType.model_json_schema()
                                )
                            elif hasattr(channel.UpdateType, "__annotations__"):
                                # Handle basic type annotations
                                properties[channel_name] = (
                                    SchemaExtractor._type_to_schema(
                                        channel.UpdateType.__annotations__.get(
                                            "__root__", str
                                        )
                                    )
                                )
                            else:
                                properties[channel_name] = {"type": "string"}
                        except Exception:
                            properties[channel_name] = {"type": "string"}
                    else:
                        properties[channel_name] = {"type": "string"}

                if properties:
                    return {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }

            return None

        except Exception as e:
            logger.debug(f"Failed to extract state schema from builder: {e}")
            return None

    @staticmethod
    def _extract_state_schema(graph: Pregel) -> Optional[Dict[str, Any]]:
        """Extract state schema from graph channels."""
        try:
            if not hasattr(graph, "channels") or not graph.channels:
                return None

            # Build schema from channel definitions
            properties = {}
            required = []

            for channel_name, channel in graph.channels.items():
                if hasattr(channel, "UpdateType") and channel.UpdateType:
                    try:
                        # Get schema from channel type
                        if hasattr(channel.UpdateType, "model_json_schema"):
                            properties[channel_name] = (
                                channel.UpdateType.model_json_schema()
                            )
                        elif hasattr(channel.UpdateType, "__annotations__"):
                            # Handle basic type annotations
                            properties[channel_name] = SchemaExtractor._type_to_schema(
                                channel.UpdateType.__annotations__.get("__root__", str)
                            )
                        else:
                            properties[channel_name] = {"type": "string"}
                    except Exception:
                        properties[channel_name] = {"type": "string"}
                else:
                    properties[channel_name] = {"type": "string"}

            if properties:
                return {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }

            return None

        except Exception as e:
            logger.debug(f"Failed to extract state schema: {e}")
            return None

    @staticmethod
    def _extract_config_schema(graph: Pregel) -> Optional[Dict[str, Any]]:
        """Extract config schema from graph configuration."""
        try:
            # Basic config schema - can be extended based on graph requirements
            return {
                "type": "object",
                "properties": {
                    "configurable": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "recursion_limit": {"type": "integer", "minimum": 1, "default": 25},
                    "max_concurrency": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            }
        except Exception as e:
            logger.debug(f"Failed to extract config schema: {e}")
            return None

    @staticmethod
    def _convert_to_json_schema(type_class) -> Optional[Dict[str, Any]]:
        """Convert a type (Pydantic model, TypedDict, etc.) to JSON schema with proper $defs.
        
        Args:
            type_class: Type to convert (Pydantic model, TypedDict, etc.)
            
        Returns:
            JSON schema dictionary with $defs section
        """
        try:
            # Handle Pydantic models
            if hasattr(type_class, "model_json_schema"):
                schema = type_class.model_json_schema()
                
                # Ensure we have proper JSON Schema structure
                if "$defs" not in schema and "definitions" in schema:
                    schema["$defs"] = schema.pop("definitions")
                    
                # Add proper schema version if not present
                if "$schema" not in schema:
                    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
                    
                return schema
                
            # Handle TypedDict classes  
            elif hasattr(type_class, '__annotations__'):
                return SchemaExtractor._typeddict_to_json_schema(type_class)
                
            return None
        except Exception as e:
            logger.debug(f"Failed to convert type to JSON schema: {e}")
            return None
            
    @staticmethod
    def _typeddict_to_json_schema(typed_dict_class) -> Dict[str, Any]:
        """Convert a TypedDict to JSON schema with proper $defs for BaseMessage types.
        
        Args:
            typed_dict_class: TypedDict class
            
        Returns:
            JSON schema with $defs section
        """
        try:
            annotations = getattr(typed_dict_class, '__annotations__', {})
            properties = {}
            required = []
            defs = {}
            
            for field_name, field_type in annotations.items():
                field_schema, field_defs = SchemaExtractor._type_to_json_schema_with_defs(field_type)
                properties[field_name] = field_schema
                defs.update(field_defs)
                
                # For now, consider all fields required
                # In the future, we could check for Optional types
                required.append(field_name)
            
            schema = {
                "type": "object", 
                "properties": properties,
                "required": required,
                "title": getattr(typed_dict_class, '__name__', 'Schema')
            }
            
            if defs:
                schema["$defs"] = defs
                
            return schema
        except Exception as e:
            logger.debug(f"Failed to convert TypedDict to JSON schema: {e}")
            return None
            
    @staticmethod 
    def _type_to_json_schema_with_defs(type_hint) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Convert a type hint to JSON schema, returning schema and any $defs.
        
        Returns:
            Tuple of (schema_dict, defs_dict)
        """
        defs = {}
        
        # Handle Annotated types (like Annotated[Sequence[BaseMessage], add_messages])
        if hasattr(type_hint, '__origin__') and getattr(type_hint, '__origin__', None) is not None:
            origin = get_origin(type_hint)
            args = get_args(type_hint)
            
            # Handle Annotated[...] 
            if hasattr(type_hint, '__metadata__'):
                # This is an Annotated type, get the first argument as the actual type
                if args:
                    actual_type = args[0]
                    return SchemaExtractor._type_to_json_schema_with_defs(actual_type)
                    
        # Handle generic types
        origin = get_origin(type_hint)
        args = get_args(type_hint) if hasattr(type_hint, '__args__') else None
        
        # Handle Sequence[BaseMessage], List[BaseMessage], etc.
        if origin in (list, tuple) or (hasattr(type_hint, '__name__') and type_hint.__name__ == 'Sequence'):
            if args:
                item_type = args[0]
                item_schema, item_defs = SchemaExtractor._type_to_json_schema_with_defs(item_type)
                defs.update(item_defs)
                return {
                    "type": "array",
                    "items": item_schema
                }, defs
            else:
                return {"type": "array"}, defs
                
        # Handle Literal types
        if hasattr(type_hint, '__origin__') and str(type_hint).startswith('typing.Literal'):
            if args:
                return {
                    "type": "string",
                    "enum": list(args)
                }, defs
                
        # Handle BaseMessage and its subclasses
        if (hasattr(type_hint, '__name__') and 
            (type_hint.__name__ == 'BaseMessage' or 
             (hasattr(type_hint, '__mro__') and any(cls.__name__ == 'BaseMessage' for cls in type_hint.__mro__)))):
            
            # Get the BaseMessage schema and add it to defs
            if hasattr(BaseMessage, 'model_json_schema'):
                base_message_schema = BaseMessage.model_json_schema()
                
                # Extract any nested definitions
                if 'definitions' in base_message_schema:
                    defs.update(base_message_schema['definitions'])
                elif '$defs' in base_message_schema:
                    defs.update(base_message_schema['$defs'])
                    
                # Clean the schema for inclusion in $defs
                clean_schema = {k: v for k, v in base_message_schema.items() 
                              if k not in ('definitions', '$defs', '$schema')}
                defs['BaseMessage'] = clean_schema
                
                return {"$ref": "#/$defs/BaseMessage"}, defs
                
        # Handle basic types
        if type_hint is str:
            return {"type": "string"}, defs
        elif type_hint is int:
            return {"type": "integer"}, defs
        elif type_hint is float:
            return {"type": "number"}, defs
        elif type_hint is bool:
            return {"type": "boolean"}, defs
        elif type_hint is list:
            return {"type": "array"}, defs
        elif type_hint is dict:
            return {"type": "object"}, defs
        else:
            return {"type": "string"}, defs  # Default fallback

    @staticmethod
    def _type_to_schema(type_hint: Any) -> Dict[str, Any]:
        """Convert a Python type hint to a JSON schema (legacy method)."""
        schema, _ = SchemaExtractor._type_to_json_schema_with_defs(type_hint)
        return schema

    @staticmethod
    def extract_static_schema(
        graph_factory: callable, config: Optional[RunnableConfig] = None
    ) -> Optional[GraphSchema]:
        """Extract schema from a graph factory without full compilation.

        Args:
            graph_factory: Factory function that creates a graph
            config: Optional configuration for the graph

        Returns:
            GraphSchema object or None if extraction fails
        """
        try:
            # Create a temporary graph instance
            if config is None:
                config = RunnableConfig()

            temp_graph = graph_factory(config)
            if not isinstance(temp_graph, Pregel):
                return None

            return SchemaExtractor.extract_runtime_schema(temp_graph)

        except Exception as e:
            logger.error(f"Failed to extract static schema: {e}")
            return None


def get_graph_schema(
    graph: Union[Pregel, callable], config: Optional[RunnableConfig] = None
) -> Optional[Dict[str, Any]]:
    """Get the schema for a graph.

    Args:
        graph: The compiled graph or graph factory
        config: Optional configuration for graph factories

    Returns:
        Dictionary containing the graph schema or None if extraction fails
    """
    try:
        if isinstance(graph, Pregel):
            schema = SchemaExtractor.extract_runtime_schema(graph)
        elif callable(graph):
            schema = SchemaExtractor.extract_static_schema(graph, config)
        else:
            logger.warning(f"Unknown graph type: {type(graph)}")
            return None

        return schema.to_dict() if schema else None

    except Exception as e:
        logger.error(f"Failed to get graph schema: {e}")
        return None
