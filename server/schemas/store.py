"""Store-related schemas for the LangGraph Agent Protocol."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, AwareDatetime, RootModel


class StorePutRequest(BaseModel):
    namespace: List[str] = Field(
        ...,
        description="A list of strings representing the namespace path.",
        title="Namespace",
    )
    key: str = Field(
        ...,
        description="The unique identifier for the item within the namespace.",
        title="Key",
    )
    value: Dict[str, Any] = Field(
        ..., description="A dictionary containing the item's data.", title="Value"
    )


class StoreDeleteRequest(BaseModel):
    namespace: Optional[List[str]] = Field(
        None,
        description="A list of strings representing the namespace path.",
        title="Namespace",
    )
    key: str = Field(
        ..., description="The unique identifier for the item.", title="Key"
    )


class StoreSearchRequest(BaseModel):
    namespace_prefix: Optional[List[str]] = Field(
        None,
        description="List of strings representing the namespace prefix.",
        title="Namespace Prefix",
    )
    filter: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional dictionary of key-value pairs to filter results.",
        title="Filter",
    )
    limit: Optional[int] = Field(
        10,
        description="Maximum number of items to return (default is 10).",
        title="Limit",
    )
    offset: Optional[int] = Field(
        0,
        description="Number of items to skip before returning results (default is 0).",
        title="Offset",
    )


class StoreListNamespacesRequest(BaseModel):
    prefix: Optional[List[str]] = Field(
        None,
        description="Optional list of strings representing the prefix to filter namespaces.",
        title="Prefix",
    )
    suffix: Optional[List[str]] = Field(
        None,
        description="Optional list of strings representing the suffix to filter namespaces.",
        title="Suffix",
    )
    max_depth: Optional[int] = Field(
        None,
        description="Optional integer specifying the maximum depth of namespaces to return.",
        title="Max Depth",
    )
    limit: Optional[int] = Field(
        100,
        description="Maximum number of namespaces to return (default is 100).",
        title="Limit",
    )
    offset: Optional[int] = Field(
        0,
        description="Number of namespaces to skip before returning results (default is 0).",
        title="Offset",
    )


class Item(BaseModel):
    namespace: List[str] = Field(
        ...,
        description="The namespace of the item. A namespace is analogous to a document's directory.",
    )
    key: str = Field(
        ...,
        description="The unique identifier of the item within its namespace. In general, keys needn't be globally unique.",
    )
    value: Dict[str, Any] = Field(
        ..., description="The value stored in the item. This is the document itself."
    )
    created_at: AwareDatetime = Field(
        ..., description="The timestamp when the item was created."
    )
    updated_at: AwareDatetime = Field(
        ..., description="The timestamp when the item was last updated."
    )


class SearchItemsResponse(BaseModel):
    items: List[Item]


class ListNamespaceResponse(RootModel[List[List[str]]]):
    root: List[List[str]]


class Namespace(RootModel[List[str]]):
    root: List[str]