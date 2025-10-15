"""Authentication utilities, matching js-server/src/auth/custom.mjs."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class AuthContext:
    """
    Authentication context container.
    """

    def __init__(
        self, user: Optional[Dict[str, Any]] = None, scopes: Optional[list[str]] = None
    ):
        self.user = user or {}
        self.scopes = scopes or []

    @property
    def user_id(self) -> Optional[str]:
        """Get the user ID from user identity or id."""
        if not self.user:
            return None
        return self.user.get("identity") or self.user.get("id")


async def handle_auth_event(
    auth: Optional[AuthContext], event_type: str, payload: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Handle authentication events and return filters and mutable data.

    In a real implementation, this would integrate with your auth system.

    Args:
        auth: Authentication context
        event_type: Type of event (e.g., "threads:create", "runs:read")
        payload: Event payload

    Returns:
        Tuple of (auth_filters, mutable_payload)
    """
    # For now, we'll implement a pass-through that allows all operations
    # In a real implementation, you would:
    # 1. Check permissions based on auth.scopes
    # 2. Apply tenant/user filtering
    # 3. Validate access to specific resources

    auth_filters = None
    if auth and auth.user:
        # Add user-based filtering in production
        auth_filters = {
            "user_id": auth.user_id,
            # Add more filter criteria based on your auth model
        }

    # Return mutable copy of payload (for metadata injection, etc.)
    mutable_payload = payload.copy()

    # Inject auth metadata if needed
    if auth and auth.user:
        if "metadata" in mutable_payload:
            if mutable_payload["metadata"] is None:
                mutable_payload["metadata"] = {}
            # Could inject user info into metadata here

    return auth_filters, mutable_payload


def is_auth_matching(
    resource_metadata: Optional[Dict[str, Any]], auth_filters: Optional[Dict[str, Any]]
) -> bool:
    """
    Check if resource metadata matches auth filters.

    Args:
        resource_metadata: Metadata from the resource being accessed
        auth_filters: Filters from authentication context

    Returns:
        True if access is allowed, False otherwise
    """
    if not auth_filters:
        return True  # No filters means allow all

    if not resource_metadata:
        return True  # No metadata to check against

    # Check each filter criterion
    for key, expected_value in auth_filters.items():
        if key in resource_metadata:
            if resource_metadata[key] != expected_value:
                return False
        # If key is not in metadata, we allow it (could be configurable)

    return True


def extract_auth_headers(headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract authentication-relevant headers for injection into config.

    Args:
        headers: Request headers dictionary

    Returns:
        Dictionary of headers to inject into run config
    """
    auth_headers = {}

    for key, value in headers.items():
        key_lower = key.lower()

        # Include custom X- headers (except auth-specific ones)
        if key_lower.startswith("x-"):
            if key_lower not in ["x-api-key", "x-tenant-id", "x-service-key"]:
                auth_headers[key_lower] = value

        # Include user-agent
        elif key_lower == "user-agent":
            auth_headers[key_lower] = value

    return auth_headers


def build_run_config(
    base_config: Optional[Dict[str, Any]],
    assistant_config: Optional[Dict[str, Any]],
    thread_config: Optional[Dict[str, Any]],
    auth_headers: Dict[str, Any],
    auth_context: Optional[AuthContext],
    **additional_config,
) -> Dict[str, Any]:
    """
    Build the final run configuration by merging all config sources.

    Args:
        base_config: Base configuration from request
        assistant_config: Configuration from assistant
        thread_config: Configuration from thread
        auth_headers: Headers to inject
        auth_context: Authentication context
        **additional_config: Additional config parameters

    Returns:
        Merged configuration dictionary
    """
    # Start with empty config
    config = {}

    # Merge configurations in order of precedence
    configs_to_merge = [
        assistant_config or {},
        thread_config or {},
        base_config or {},
    ]

    for cfg in configs_to_merge:
        if cfg:
            config.update(cfg)

    # Handle configurable section separately for proper merging
    configurable = {}
    for cfg in configs_to_merge:
        if cfg and "configurable" in cfg and cfg["configurable"]:
            configurable.update(cfg["configurable"])

    # Add auth headers to configurable
    if auth_headers:
        configurable.update(auth_headers)

    # Add auth context info
    if auth_context and auth_context.user:
        configurable.update(
            {
                "langgraph_auth_user": auth_context.user,
                "langgraph_auth_user_id": auth_context.user_id,
                "langgraph_auth_permissions": auth_context.scopes,
            }
        )

    # Add any additional config
    configurable.update(additional_config)

    # Set the configurable section
    if configurable:
        config["configurable"] = configurable

    return config
