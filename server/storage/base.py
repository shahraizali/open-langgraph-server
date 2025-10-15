"""Base storage interface and common utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional, TypeVar, Generic
from datetime import datetime
import uuid

T = TypeVar("T")


class BaseStorage(ABC, Generic[T]):
    """Abstract base class for storage operations."""

    @abstractmethod
    async def get(self, item_id: str, auth_context: Optional[Any] = None) -> T:
        """Get an item by ID."""
        pass

    @abstractmethod
    async def put(
        self, item_id: str, data: Dict[str, Any], auth_context: Optional[Any] = None
    ) -> T:
        """Create or update an item."""
        pass

    @abstractmethod
    async def delete(
        self, item_id: str, auth_context: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Delete an item."""
        pass

    @abstractmethod
    async def search(
        self, filters: Dict[str, Any], auth_context: Optional[Any] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Search for items matching filters."""
        pass


class StorageError(Exception):
    """Base exception for storage operations."""

    pass


class ItemNotFoundError(StorageError):
    """Raised when an item is not found."""

    def __init__(self, item_type: str, item_id: str):
        self.item_type = item_type
        self.item_id = item_id
        super().__init__(f"{item_type} '{item_id}' not found")


class ItemExistsError(StorageError):
    """Raised when trying to create an item that already exists."""

    def __init__(self, item_type: str, item_id: str):
        self.item_type = item_type
        self.item_id = item_id
        super().__init__(f"{item_type} '{item_id}' already exists")


def generate_id() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def validate_uuid(value: str) -> bool:
    """Validate if a string is a valid UUID."""
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False
