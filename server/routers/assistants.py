from __future__ import annotations

import logging
import uuid
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Response, Query, Request, Depends

from ..schemas import (
    Assistant,
    AssistantCreate,
    AssistantPatch,
    AssistantSearchRequest,
    AssistantSchema,
    AssistantLatestVersion,
    ErrorResponse,
)
from ..storage.postgres_storage import postgres_assistant_storage as assistant_storage
from ..graph.assistants import get_assistant_id, get_runnable_config, GRAPHS
from ..graph.loader import get_graph
from ..graph.schema import get_graph_schema

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Assistants"])




async def _expand_subgraphs_for_xray(
    graph, base_result: Dict[str, Any], config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Expand subgraphs to show internal nodes when xray=true"""
    try:
        # Check if we have subgraphs to expand
        if not hasattr(graph, "get_subgraphs"):
            return None

        expanded_nodes = []
        expanded_edges = []

        # Get all subgraphs
        subgraphs_gen = graph.get_subgraphs()
        all_subgraphs = list(subgraphs_gen)

        # Track which high-level nodes are subgraphs
        subgraph_node_ids = set()

        for ns, subgraph in all_subgraphs:
            subgraph_node_ids.add(ns)

            try:
                # Get the internal structure of this subgraph
                if hasattr(subgraph, "get_graph"):
                    internal_graph = subgraph.get_graph(config)
                elif hasattr(subgraph, "get_graph_async"):
                    internal_graph = await subgraph.get_graph_async(config)
                else:
                    continue

                # Convert to dict if needed
                if hasattr(internal_graph, "to_json"):
                    internal_data = internal_graph.to_json()
                elif hasattr(internal_graph, "model_dump"):
                    internal_data = internal_graph.model_dump()
                elif isinstance(internal_graph, dict):
                    internal_data = internal_graph
                else:
                    continue

                # Add internal nodes with namespace prefix
                for node in internal_data.get("nodes", []):
                    expanded_node = {
                        "id": f"{ns}:{node['id']}",
                        "type": node.get("type", "runnable"),
                        "data": node.get(
                            "data",
                            {
                                "id": [
                                    "langgraph",
                                    "utils",
                                    "runnable",
                                    "RunnableCallable",
                                ],
                                "name": f"{ns}:{node['id']}",
                            },
                        ),
                    }
                    expanded_nodes.append(expanded_node)

                # Add internal edges with namespace prefix
                for edge in internal_data.get("edges", []):
                    expanded_edge = {
                        "source": f"{ns}:{edge['source']}",
                        "target": f"{ns}:{edge['target']}",
                    }
                    if edge.get("conditional"):
                        expanded_edge["conditional"] = True
                    expanded_edges.append(expanded_edge)

            except Exception as e:
                logger.warning(f"Error expanding subgraph {ns}: {e}")
                continue

        # Add non-subgraph nodes from base result
        for node in base_result.get("nodes", []):
            if node["id"] not in subgraph_node_ids:
                expanded_nodes.append(node)

        # Add edges between subgraphs and main nodes, with namespace prefixes
        for edge in base_result.get("edges", []):
            source = edge["source"]
            target = edge["target"]

            # If source is a subgraph, connect from its __end__ node
            if source in subgraph_node_ids:
                source = f"{source}:__end__"

            # If target is a subgraph, connect to its __start__ node
            if target in subgraph_node_ids:
                target = f"{target}:__start__"

            expanded_edge = {"source": source, "target": target}
            if edge.get("conditional"):
                expanded_edge["conditional"] = True
            expanded_edges.append(expanded_edge)

        return {"nodes": expanded_nodes, "edges": expanded_edges}

    except Exception as e:
        logger.error(f"Error in subgraph expansion: {e}")
        return None


def _generate_basic_graph_structure(
    assistant_id: str, graph_id: str, xray: bool = False
) -> Dict[str, Any]:
    """Generate a basic graph structure dynamically"""
    return {
        "nodes": [
            {"id": "__start__", "type": "start", "metadata": {"graph_id": graph_id}},
            {
                "id": f"{graph_id}_node",
                "type": "node",
                "metadata": {
                    "graph_id": graph_id,
                    "xray": xray,
                    "description": f"Main node for {graph_id}",
                },
            },
            {"id": "__end__", "type": "end", "metadata": {"graph_id": graph_id}},
        ],
        "edges": [
            {"source": "__start__", "target": f"{graph_id}_node"},
            {"source": f"{graph_id}_node", "target": "__end__"},
        ],
        "metadata": {
            "assistant_id": assistant_id,
            "graph_id": graph_id,
            "xray_mode": xray,
            "generated": True,
        },
    }


async def parse_search_request(request: Request) -> AssistantSearchRequest:
    """Custom dependency to handle both JSON and form data for search requests."""
    content_type = request.headers.get("content-type", "")

    try:
        if "application/json" in content_type:
            # Handle JSON request normally
            body = await request.json()
            return AssistantSearchRequest(**body)
        elif "application/x-www-form-urlencoded" in content_type:
            # Handle form data
            form_data = await request.form()

            # Convert form data to dict
            data = {}
            for key, value in form_data.items():
                if key == "metadata":
                    # Try to parse metadata as JSON if it's a string
                    try:
                        data[key] = (
                            json.loads(value) if isinstance(value, str) else value
                        )
                    except (json.JSONDecodeError, TypeError):
                        # If it fails, treat as simple key-value
                        data[key] = {value: True} if isinstance(value, str) else value
                elif key in ["limit", "offset"]:
                    # Convert numeric fields
                    try:
                        data[key] = int(value)
                    except (ValueError, TypeError):
                        data[key] = value
                else:
                    data[key] = value

            return AssistantSearchRequest(**data)
        else:
            # Try to parse as JSON anyway
            try:
                body = await request.json()
                return AssistantSearchRequest(**body)
            except Exception:
                # Last resort - treat as empty search
                return AssistantSearchRequest()

    except Exception as e:
        logger.error(f"Error parsing search request: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Invalid request format. Expected JSON with fields: limit, offset, metadata, graph_id. Error: {str(e)}",
        )


@router.post(
    "/assistants",
    response_model=Assistant,
    responses={"409": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def create_assistant(body: AssistantCreate) -> Assistant:
    """Create Assistant"""
    try:
        # Generate ID if not provided
        assistant_id = str(body.assistant_id or uuid.uuid4())

        # Create assistant
        assistant_data = {
            "config": body.config.model_dump() if body.config else {},
            "graph_id": body.graph_id,
            "metadata": body.metadata or {},
            "if_exists": body.if_exists or "raise",
            "name": body.name or f"Assistant for {body.graph_id}",
        }

        assistant_data = await assistant_storage.put(assistant_id, assistant_data)

        # Convert to API model with proper datetime formatting
        assistant = Assistant(
            assistant_id=assistant_data.assistant_id,
            graph_id=assistant_data.graph_id,
            config=assistant_data.config,
            created_at=(
                assistant_data.created_at.isoformat() + "Z"
                if hasattr(assistant_data.created_at, "isoformat")
                else str(assistant_data.created_at)
            ),
            updated_at=(
                assistant_data.updated_at.isoformat() + "Z"
                if hasattr(assistant_data.updated_at, "isoformat")
                else str(assistant_data.updated_at)
            ),
            metadata=assistant_data.metadata,
            name=assistant_data.name or f"Assistant for {assistant_data.graph_id}",
            version=assistant_data.version,
        )

        return assistant

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating assistant: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/assistants/search",
    response_model=List[Assistant],
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def search_assistants(
    body: AssistantSearchRequest = Depends(parse_search_request),
    response: Response = None,
) -> List[Assistant]:
    """Search Assistants"""
    try:
        # Convert request to storage filters
        filters = {
            "graph_id": body.graph_id,
            "metadata": body.metadata,
            "limit": body.limit or 10,
            "offset": body.offset or 0,
        }

        # Search assistants
        result = []
        total = 0

        async for item in assistant_storage.search(filters):
            # Add assistant object to result list
            result.append(item["assistant"])
            if total == 0:
                total = item["total"]

        # Set pagination header for total count
        response.headers["X-Pagination-Total"] = str(total)

        return result

    except Exception as e:
        logger.error(f"Error searching assistants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/assistants/{assistant_id}",
    response_model=Assistant,
    responses={"404": {"model": ErrorResponse}},
)
async def get_assistant(assistant_id: str) -> Assistant:
    """Get Assistant"""
    try:
        # Convert to proper assistant ID
        resolved_assistant_id = get_assistant_id(assistant_id)

        # Get assistant from storage
        assistant_data = await assistant_storage.get(resolved_assistant_id)

        if not assistant_data:
            raise HTTPException(status_code=404, detail="Assistant not found")

        # Convert to API model with proper datetime formatting
        assistant = Assistant(
            assistant_id=assistant_data.assistant_id,
            graph_id=assistant_data.graph_id,
            config=assistant_data.config,
            created_at=(
                assistant_data.created_at.isoformat() + "Z"
                if hasattr(assistant_data.created_at, "isoformat")
                else str(assistant_data.created_at)
            ),
            updated_at=(
                assistant_data.updated_at.isoformat() + "Z"
                if hasattr(assistant_data.updated_at, "isoformat")
                else str(assistant_data.updated_at)
            ),
            metadata=assistant_data.metadata,
            name=assistant_data.name or f"Assistant for {assistant_data.graph_id}",
            version=assistant_data.version,
        )

        return assistant

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assistant {assistant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/assistants/{assistant_id}",
    response_model=List[str],
    responses={"404": {"model": ErrorResponse}},
)
async def delete_assistant(assistant_id: str) -> List[str]:
    """Delete Assistant"""
    try:
        # Convert to proper assistant ID
        resolved_assistant_id = get_assistant_id(assistant_id)

        # Delete assistant
        deleted = await assistant_storage.delete(resolved_assistant_id)

        return deleted

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting assistant {assistant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/assistants/{assistant_id}",
    response_model=Assistant,
    responses={"404": {"model": ErrorResponse}},
)
async def patch_assistant(assistant_id: str, body: AssistantPatch) -> Assistant:
    """Patch Assistant"""
    try:
        # Convert to proper assistant ID
        resolved_assistant_id = get_assistant_id(assistant_id)

        # Prepare updates
        updates = {}
        if body.graph_id is not None:
            updates["graph_id"] = body.graph_id
        if body.config is not None:
            updates["config"] = body.config.model_dump()
        if body.name is not None:
            updates["name"] = body.name
        if body.metadata is not None:
            updates["metadata"] = body.metadata

        # Update assistant
        assistant_data = await assistant_storage.patch(resolved_assistant_id, updates)

        # Convert to API model with proper datetime formatting
        assistant = Assistant(
            assistant_id=assistant_data.assistant_id,
            graph_id=assistant_data.graph_id,
            config=assistant_data.config,
            created_at=(
                assistant_data.created_at.isoformat() + "Z"
                if hasattr(assistant_data.created_at, "isoformat")
                else str(assistant_data.created_at)
            ),
            updated_at=(
                assistant_data.updated_at.isoformat() + "Z"
                if hasattr(assistant_data.updated_at, "isoformat")
                else str(assistant_data.updated_at)
            ),
            metadata=assistant_data.metadata,
            name=assistant_data.name or f"Assistant for {assistant_data.graph_id}",
            version=assistant_data.version,
        )

        return assistant

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error patching assistant {assistant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/assistants/{assistant_id}/schemas",
    response_model=AssistantSchema,
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
)
async def get_assistant_schemas(assistant_id: str) -> AssistantSchema:
    """Get Assistant Schemas"""
    try:
        # Convert to proper assistant ID
        resolved_assistant_id = get_assistant_id(assistant_id)

        # Get assistant from storage
        assistant = await assistant_storage.get(resolved_assistant_id)
        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")

        # Get the graph for this assistant
        config = get_runnable_config(assistant.config)
        graph = get_graph(assistant.graph_id, config)

        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph {assistant.graph_id} not found")

        # Extract schema from the graph using runtime extraction
        runtime_schema = get_graph_schema(graph, config)
        
        if not runtime_schema:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to extract schema for graph {assistant.graph_id}"
            )

        # Create response model - the get_graph_schema returns the schema dict directly
        assistant_schema = AssistantSchema(
            graph_id=assistant.graph_id,
            input_schema=runtime_schema.get("input"),
            output_schema=runtime_schema.get("output"),
            state_schema=runtime_schema.get("state"),
            config_schema=runtime_schema.get("config"),
        )

        return assistant_schema

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assistant schemas for {assistant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/assistants/{assistant_id}/versions",
    response_model=List[Dict[str, Any]],
    responses={"404": {"model": ErrorResponse}},
)
async def get_assistant_versions(
    assistant_id: str, body: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Get Assistant Versions"""
    try:
        # Convert to proper assistant ID
        resolved_assistant_id = get_assistant_id(assistant_id)

        # Get filters from body
        filters = {
            "limit": body.get("limit", 10),
            "offset": body.get("offset", 0),
            "metadata": body.get("metadata"),
        }

        # Get versions
        versions = await assistant_storage.get_versions(resolved_assistant_id, filters)

        if not versions:
            raise HTTPException(
                status_code=404, detail=f'Assistant "{assistant_id}" not found.'
            )

        return versions

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assistant versions for {assistant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/assistants/{assistant_id}/latest",
    response_model=Assistant,
    responses={"404": {"model": ErrorResponse}},
)
async def set_latest_assistant_version(
    assistant_id: str, body: AssistantLatestVersion
) -> Assistant:
    """Set Latest Assistant Version"""
    try:
        # Convert to proper assistant ID
        resolved_assistant_id = get_assistant_id(assistant_id)

        # Set latest version
        assistant_data = await assistant_storage.set_latest(
            resolved_assistant_id, body.version
        )

        # Convert to API model with proper datetime formatting
        assistant = Assistant(
            assistant_id=assistant_data.assistant_id,
            graph_id=assistant_data.graph_id,
            config=assistant_data.config,
            created_at=(
                assistant_data.created_at.isoformat() + "Z"
                if hasattr(assistant_data.created_at, "isoformat")
                else str(assistant_data.created_at)
            ),
            updated_at=(
                assistant_data.updated_at.isoformat() + "Z"
                if hasattr(assistant_data.updated_at, "isoformat")
                else str(assistant_data.updated_at)
            ),
            metadata=assistant_data.metadata,
            name=assistant_data.name or f"Assistant for {assistant_data.graph_id}",
            version=assistant_data.version,
        )

        return assistant

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting latest version for assistant {assistant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/assistants/{assistant_id}/graph",
    response_model=Dict[str, Any],
    responses={"404": {"model": ErrorResponse}},
)
async def get_assistant_graph(
    assistant_id: str, xray: Optional[bool] = Query(False)
) -> Dict[str, Any]:
    """Get Assistant Graph"""
    try:
        # Convert to proper assistant ID
        resolved_assistant_id = get_assistant_id(assistant_id)

        # Get assistant from storage
        assistant = await assistant_storage.get(resolved_assistant_id)

        # Get the graph for this assistant
        config = get_runnable_config(assistant.config)
        graph = get_graph(assistant.graph_id, config)

        # Get graph visualization
        try:
            # Prepare config with xray parameter
            graph_config = {**config}
            if xray is not None:
                graph_config["xray"] = xray

            logger.debug(f"Getting graph with config: {graph_config}, xray: {xray}")

            # Try to get the graph visualization
            if hasattr(graph, "get_graph_async"):
                # Async method (newer LangGraph versions)
                drawable = await graph.get_graph_async(graph_config)
            elif hasattr(graph, "get_graph"):
                # Sync method (older LangGraph versions)
                drawable = graph.get_graph(graph_config)
            else:
                # Fallback to basic graph structure
                logger.info("Using fallback graph structure")
                drawable = _generate_basic_graph_structure(
                    assistant_id, assistant.graph_id, xray or False
                )

            # Convert to JSON format
            if hasattr(drawable, "to_json"):
                result = drawable.to_json()
            elif hasattr(drawable, "model_dump"):
                result = drawable.model_dump()
            elif isinstance(drawable, dict):
                result = drawable
            else:
                # Fallback: try to convert to dict
                result = (
                    dict(drawable)
                    if hasattr(drawable, "__dict__")
                    else {"error": str(drawable)}
                )

            # If xray is enabled and we don't have detailed subgraph nodes, expand them manually
            if xray and result.get("nodes"):
                expanded_result = await _expand_subgraphs_for_xray(
                    graph, result, config
                )
                if expanded_result:
                    logger.info(
                        f"Expanded xray graph from {len(result.get('nodes', []))} to {len(expanded_result.get('nodes', []))} nodes"
                    )
                    return expanded_result

            return result

        except Exception as graph_error:
            logger.warning(
                f"Error getting graph visualization for {assistant_id}: {graph_error}"
            )
            # Fallback to basic graph structure
            return _generate_basic_graph_structure(
                assistant_id, assistant.graph_id, xray or False
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assistant graph for {assistant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/assistants/{assistant_id}/subgraphs/{namespace}",
    response_model=Dict[str, Any],
    responses={"404": {"model": ErrorResponse}},
)
async def get_assistant_subgraph_by_namespace(
    assistant_id: str, namespace: str, recurse: Optional[bool] = Query(False)
) -> Dict[str, Any]:
    """Get Assistant Subgraph by Namespace"""
    # Call the main function with the namespace parameter
    return await get_assistant_subgraphs(assistant_id, namespace, recurse)


@router.get(
    "/assistants/{assistant_id}/subgraphs",
    response_model=Dict[str, Any],
    responses={"404": {"model": ErrorResponse}},
)
async def get_assistant_subgraphs(
    assistant_id: str,
    namespace: Optional[str] = None,
    recurse: Optional[bool] = Query(False),
) -> Dict[str, Any]:
    """Get Assistant Subgraphs"""
    try:
        # Convert to proper assistant ID
        resolved_assistant_id = get_assistant_id(assistant_id)

        # Get assistant from storage
        assistant = await assistant_storage.get(resolved_assistant_id)

        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")

        # Get the graph for this assistant
        config = get_runnable_config(assistant.config)
        graph = get_graph(assistant.graph_id, config)

        # Initialize result array for subgraph pairs: [name, schema]
        result_pairs = []

        # Determine subgraphs generator (Python LangGraph API)
        if hasattr(graph, "get_subgraphs"):
            # Get all subgraphs - this returns a generator of (namespace, subgraph) tuples
            subgraphs_gen = graph.get_subgraphs()

            # Convert generator to list to enable filtering
            all_subgraphs_list = list(subgraphs_gen)

            # Filter by namespace if provided
            if namespace:
                subgraphs_items = [
                    (ns, sg) for ns, sg in all_subgraphs_list if ns == namespace
                ]
            else:
                # Get all subgraphs or just main if recurse is False
                if recurse:
                    subgraphs_items = all_subgraphs_list
                else:
                    # Just return the main subgraph if available, otherwise the first one
                    # Common subgraph names to check for main graph
                    main_graph_names = ["main", "__start__", "start", "root", "default"]
                    main_items = [
                        (ns, sg)
                        for ns, sg in all_subgraphs_list
                        if ns in main_graph_names
                    ]
                    if main_items:
                        subgraphs_items = [main_items[0]]
                    elif all_subgraphs_list:
                        subgraphs_items = [all_subgraphs_list[0]]
                    else:
                        subgraphs_items = []

            # Convert to async generator for consistency
            async def subgraphs_generator():
                for ns, subgraph in subgraphs_items:
                    yield ns, subgraph

        else:
            # Fallback: create a single subgraph with dynamic name based on graph_id
            async def subgraphs_generator():
                yield assistant.graph_id, graph

        # Cache for graph schema
        graph_schema_promise = None

        # Process each subgraph
        async for ns, subgraph in subgraphs_generator():
            # Get schema for this subgraph
            try:
                # Try runtime schema extraction first
                runtime_schema = get_graph_schema(subgraph, config)
                if runtime_schema:
                    schema = runtime_schema
                else:
                    # Lazy load cached schema if needed
                    if graph_schema_promise is None:
                        graph_schema_promise = get_graph_schema(assistant.graph_id)

                    cached_schema = graph_schema_promise
                    if cached_schema:
                        # Find root graph ID (key without pipe separator)
                        root_graph_id = next(
                            (k for k in cached_schema.keys() if "|" not in k), None
                        )
                        if not root_graph_id:
                            raise HTTPException(
                                status_code=404, detail="Failed to find root graph"
                            )

                        # Get subgraph schema with fallback to root schema
                        schema_key = f"{root_graph_id}|{ns}"
                        schema = cached_schema.get(
                            schema_key, cached_schema[root_graph_id]
                        )
                    else:
                        raise HTTPException(
                            status_code=404, detail="No cached schema available"
                        )

            except Exception as schema_error:
                logger.warning(
                    f"Error getting schema for subgraph {ns}: {schema_error}"
                )
                # This should not happen in normal operation - re-raise the error
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get schema for subgraph {ns}: {str(schema_error)}",
                )

            # Transform schema to expected format with _schema suffixes
            transformed_schema = {
                "graph_id": assistant.graph_id,
                "input_schema": schema.get("input"),
                "output_schema": schema.get("output"),
                "state_schema": schema.get("state"),
                "config_schema": schema.get("config"),
            }

            # Add to result pairs
            result_pairs.append([ns, transformed_schema])

        # Convert pairs to dictionary format
        return dict(result_pairs)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assistant subgraphs for {assistant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/assistants/debug/storage",
    response_model=Dict[str, Any],
    tags=["Debug"],
)
async def debug_storage() -> Dict[str, Any]:
    """Debug endpoint to show PostgreSQL storage contents"""
    try:
        # Get all assistants from PostgreSQL
        assistants = []
        async for result in assistant_storage.search({"limit": 100, "offset": 0}):
            assistant_data = result["assistant"]
            assistants.append(
                {
                    "assistant_id": assistant_data.assistant_id,
                    "name": assistant_data.name,
                    "graph_id": assistant_data.graph_id,
                    "version": assistant_data.version,
                    "metadata": assistant_data.metadata,
                }
            )

        return {
            "assistants": assistants,
            "graphs": GRAPHS,
            "total_assistants": len(assistants),
            "storage_type": "postgresql",
        }
    except Exception as e:
        logger.error(f"Error in debug storage: {e}")
        return {
            "error": str(e),
            "assistants": [],
            "graphs": GRAPHS,
            "total_assistants": 0,
            "storage_type": "postgresql",
        }
