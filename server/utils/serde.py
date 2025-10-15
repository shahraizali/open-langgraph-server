from __future__ import annotations

import json
import logging
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


def serialize_as_dict(obj: Any) -> str:
    """
    Serialize an object to JSON string with special handling for objects with toDict method.
    """

    def json_serializer(obj: Any) -> Any:
        # Handle objects with toDict method
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                result = obj.to_dict()
                if isinstance(result, dict) and "type" in result and "data" in result:
                    return {**result["data"], "type": result["type"]}
                return result
            except Exception as e:
                logger.warning(f"Failed to call to_dict on object: {e}")

        # Handle Pydantic models
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            try:
                return obj.model_dump()
            except Exception as e:
                logger.warning(f"Failed to call model_dump on object: {e}")

        # Handle standard dict-like objects
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        # Fallback for other types
        return str(obj)

    # Use compact JSON format
    return json.dumps(obj, default=json_serializer, separators=(",", ":"))


def serialize_error(error: Union[Exception, Any]) -> Dict[str, str]:
    """
    Serialize an error to a consistent format.
    """
    if isinstance(error, Exception):
        return {"error": error.__class__.__name__, "message": str(error)}

    return {
        "error": "Error",
        "message": json.dumps(error) if error is not None else "Unknown error",
    }


def is_object(value: Any) -> bool:
    """Check if a value is a dict-like object."""
    return isinstance(value, dict)


def is_jsonb_contained(
    superset: Dict[str, Any] | None,
    subset: Dict[str, Any] | None,
) -> bool:
    """
    Check if subset is contained within superset (JSONB containment).
    """
    if superset is None or subset is None:
        return True

    for key, value in subset.items():
        if key not in superset:
            return False

        if is_object(value) and is_object(superset[key]):
            if not is_jsonb_contained(superset[key], value):
                return False
        elif superset[key] != value:
            return False

    return True
