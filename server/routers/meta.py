from __future__ import annotations

import os
import toml
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter

# from ..storage.database import get_database

router = APIRouter(tags=["Meta"])


def get_version() -> str:
    """Get version from pyproject.toml."""
    try:
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            data = toml.load(pyproject_path)
            return data.get("project", {}).get("version", "unknown")
    except Exception:
        pass
    return "unknown"


@router.get("/info")
async def get_info() -> Dict[str, Any]:
    """Get server information and feature flags."""

    # Check LangSmith configuration
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")

    langsmith_tracing = None
    if langsmith_api_key:
        # Check if any tracing variable is explicitly set to "false"
        tracing_vars = [
            os.getenv("LANGCHAIN_TRACING_V2"),
            os.getenv("LANGCHAIN_TRACING"),
            os.getenv("LANGSMITH_TRACING_V2"),
            os.getenv("LANGSMITH_TRACING"),
        ]

        # Return true unless explicitly disabled
        langsmith_tracing = not any(
            val in ["false", "False"] for val in tracing_vars if val
        )

    return {
        "version": get_version(),
        "context": "python",
        "flags": {
            "assistants": True,
            "crons": False,  # Not implemented yet
            "langsmith": bool(langsmith_tracing),
            "langsmith_tracing_replicas": True,
        },
    }


@router.get("/ok")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {"ok": True, "status": "healthy"}
