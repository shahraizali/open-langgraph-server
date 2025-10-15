"""Middleware for FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def setup_cors_middleware(
    app: FastAPI,
    allow_origins: Optional[List[str]] = None,
    allow_methods: Optional[List[str]] = None,
    allow_headers: Optional[List[str]] = None,
    allow_credentials: bool = False,
    allow_origin_regex: Optional[str] = None,
    expose_headers: Optional[List[str]] = None,
    max_age: int = 600,
) -> None:
    """Setup CORS middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        allow_origins: List of allowed origins (default: ["*"])
        allow_methods: List of allowed HTTP methods (default: ["*"])
        allow_headers: List of allowed headers (default: ["*"])
        allow_credentials: Whether to allow credentials (default: False)
        allow_origin_regex: Regex pattern for allowed origins
        expose_headers: List of headers to expose to the browser
        max_age: Maximum age for preflight requests cache
    """
    # Set defaults
    if allow_origins is None:
        allow_origins = ["*"]
    if allow_methods is None:
        allow_methods = ["*"]
    if allow_headers is None:
        allow_headers = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        allow_origin_regex=allow_origin_regex,
        expose_headers=expose_headers,
        max_age=max_age,
    )

    logger.info(f"CORS middleware configured with origins: {allow_origins}")


def setup_cors_from_config(app: FastAPI, cors_config: Optional[Dict] = None) -> None:
    """Setup CORS middleware from configuration dictionary.

    Args:
        app: FastAPI application instance
        cors_config: CORS configuration dictionary
    """
    if cors_config is None:
        # Default permissive CORS for development
        setup_cors_middleware(app)
        return

    setup_cors_middleware(
        app,
        allow_origins=cors_config.get("allow_origins"),
        allow_methods=cors_config.get("allow_methods"),
        allow_headers=cors_config.get("allow_headers"),
        allow_credentials=cors_config.get("allow_credentials", False),
        allow_origin_regex=cors_config.get("allow_origin_regex"),
        expose_headers=cors_config.get("expose_headers"),
        max_age=cors_config.get("max_age", 600),
    )
