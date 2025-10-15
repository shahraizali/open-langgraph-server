from __future__ import annotations

from typing import Annotated, List

from fastapi import APIRouter, Query, HTTPException

from typing import Optional, Union
from ..schemas import (
    ErrorResponse,
    Item,
    ListNamespaceResponse,
    SearchItemsResponse,
    StoreDeleteRequest,
    StoreListNamespacesRequest,
    StorePutRequest,
    StoreSearchRequest,
)
from ..storage.postgres_storage import postgres_store_storage

router = APIRouter(tags=["Store"])


def validate_namespace(namespace: List[str]) -> None:
    """
    Validate namespace format following JavaScript server rules.

    Raises HTTPException if namespace is invalid.
    """
    if not namespace or len(namespace) == 0:
        raise HTTPException(status_code=400, detail="Namespace is required")

    for label in namespace:
        if not label or "." in label:
            raise HTTPException(
                status_code=422,
                detail=f"Namespace labels cannot be empty or contain periods. Received: {'.'.join(namespace)}",
            )


def map_items_to_api(item: dict) -> dict:
    """
    Transform internal item format to API format.

    Removes internal fields and renames timestamp fields to match API spec.
    """
    if item is None:
        return None

    # Create a copy and transform field names to match API format
    api_item = {**item}

    # The StoreItem.from_db_row already returns the correct format
    # but ensure we have the right structure
    return {
        "namespace": api_item["namespace"],
        "key": api_item["key"],
        "value": api_item["value"],
        "created_at": api_item["created_at"],
        "updated_at": api_item["updated_at"],
    }


@router.put(
    "/store/items",
    response_model=None,
    status_code=204,
    responses={"422": {"model": ErrorResponse}},
    tags=["Store"],
)
async def put_item(body: StorePutRequest) -> Optional[ErrorResponse]:
    """
    Insert or Update Item
    """
    try:
        # Validate namespace if provided
        if body.namespace:
            validate_namespace(body.namespace)

        # Store the item
        await postgres_store_storage.put(body.namespace or [], body.key, body.value)

        # Return 204 No Content (FastAPI will handle the empty response)
        return None

    except HTTPException:
        # Re-raise HTTPExceptions (validation errors)
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete(
    "/store/items",
    response_model=None,
    status_code=204,
    responses={"404": {"model": ErrorResponse}, "422": {"model": ErrorResponse}},
    tags=["Store"],
)
async def delete_item(body: StoreDeleteRequest) -> Optional[ErrorResponse]:
    """
    Delete Store Item
    """
    try:
        # Validate namespace if provided
        if body.namespace:
            validate_namespace(body.namespace)

        # Delete the item
        success = await postgres_store_storage.delete(body.namespace or [], body.key)

        if not success:
            raise HTTPException(status_code=404, detail="Item not found")

        # Return 204 No Content
        return None

    except HTTPException:
        # Re-raise HTTPExceptions (validation/not found errors)
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get(
    "/store/items",
    response_model=Item,
    responses={
        "400": {"model": ErrorResponse},
        "404": {"model": ErrorResponse},
        "422": {"model": ErrorResponse},
    },
    tags=["Store"],
)
async def get_item(
    key: str, namespace: Annotated[list[str] | None, Query()] = None
) -> Union[Item, ErrorResponse]:
    """
    Get Store Item
    """
    try:
        # Validate namespace if provided
        if namespace:
            validate_namespace(namespace)

        # Get the item
        item = await postgres_store_storage.get(namespace or [], key)

        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")

        # Transform to API format and return
        return map_items_to_api(item)

    except HTTPException:
        # Re-raise HTTPExceptions (validation/not found errors)
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post(
    "/store/items/search",
    response_model=SearchItemsResponse,
    responses={"422": {"model": ErrorResponse}},
    tags=["Store"],
)
async def search_items(
    body: StoreSearchRequest,
) -> Union[SearchItemsResponse, ErrorResponse]:
    """
    Search Store Items
    """
    try:
        # Validate namespace_prefix if provided
        if body.namespace_prefix:
            validate_namespace(body.namespace_prefix)

        # Prepare search options
        search_options = {
            "filter": body.filter,
            "limit": body.limit or 10,
            "offset": body.offset or 0,
            "query": body.query,
        }

        # Search for items
        items = await postgres_store_storage.search(
            body.namespace_prefix, search_options
        )

        # Transform items to API format
        api_items = [map_items_to_api(item) for item in items]

        return SearchItemsResponse(items=api_items)

    except HTTPException:
        # Re-raise HTTPExceptions (validation errors)
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post(
    "/store/namespaces",
    response_model=ListNamespaceResponse,
    responses={"422": {"model": ErrorResponse}},
    tags=["Store"],
)
async def list_namespaces(
    body: StoreListNamespacesRequest,
) -> Union[ListNamespaceResponse, ErrorResponse]:
    """
    List namespaces
    """
    try:
        # Validate prefix and suffix if provided
        if body.prefix:
            validate_namespace(body.prefix)
        if body.suffix:
            validate_namespace(body.suffix)

        # Prepare list options
        list_options = {
            "prefix": body.prefix,
            "suffix": body.suffix,
            "max_depth": body.max_depth,
            "limit": body.limit or 100,
            "offset": body.offset or 0,
        }

        # Get namespaces
        namespaces = await postgres_store_storage.list_namespaces(list_options)

        return ListNamespaceResponse(root=namespaces)

    except HTTPException:
        # Re-raise HTTPExceptions (validation errors)
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
