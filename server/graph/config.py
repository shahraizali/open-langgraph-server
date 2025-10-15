"""Configuration schema models for LangGraph server."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import os

from pydantic import BaseModel, Field, validator


class AuthConfig(BaseModel):
    """Authentication configuration."""

    path: Optional[str] = Field(
        None, description="Path to auth module in format 'file_path:export_symbol'"
    )
    disable_studio_auth: bool = Field(
        False, description="Whether to disable studio authentication"
    )


class CorsConfig(BaseModel):
    """CORS configuration."""

    allow_origins: Optional[List[str]] = Field(None, description="Allowed origins")
    allow_methods: Optional[List[str]] = Field(None, description="Allowed methods")
    allow_headers: Optional[List[str]] = Field(None, description="Allowed headers")
    allow_credentials: Optional[bool] = Field(None, description="Allow credentials")
    allow_origin_regex: Optional[str] = Field(None, description="Allow origin regex")
    expose_headers: Optional[List[str]] = Field(None, description="Expose headers")
    max_age: Optional[int] = Field(None, description="Max age")


class HttpConfig(BaseModel):
    """HTTP configuration."""

    app: Optional[str] = Field(
        None, description="Path to HTTP app in format 'file_path:export_symbol'"
    )
    disable_assistants: bool = Field(False, description="Disable assistants endpoint")
    disable_threads: bool = Field(False, description="Disable threads endpoint")
    disable_runs: bool = Field(False, description="Disable runs endpoint")
    disable_store: bool = Field(False, description="Disable store endpoint")
    disable_meta: bool = Field(False, description="Disable meta endpoint")
    cors: Optional[CorsConfig] = Field(None, description="CORS configuration")


class UIConfig(BaseModel):
    """UI configuration."""

    shared: Optional[List[str]] = Field(None, description="Shared UI components")


class LangGraphConfig(BaseModel):
    """LangGraph server configuration schema."""

    dependencies: Optional[List[str]] = Field(
        default_factory=lambda: ["."], description="List of dependency paths"
    )
    graphs: Dict[str, str] = Field(
        description="Mapping of graph_id to 'file_path:export_symbol'"
    )
    env: Optional[Union[str, Dict[str, Any]]] = Field(
        None, description="Environment file path or environment variables dict"
    )
    auth: Optional[AuthConfig] = Field(None, description="Authentication configuration")
    ui: Optional[Dict[str, str]] = Field(None, description="UI component mappings")
    ui_config: Optional[UIConfig] = Field(None, description="UI configuration")
    http: Optional[HttpConfig] = Field(None, description="HTTP configuration")
    dockerfile_lines: Optional[List[str]] = Field(
        None, description="Additional dockerfile lines"
    )

    @validator("graphs")
    def validate_graphs(cls, v):
        """Validate graph specifications."""
        if not v:
            raise ValueError("At least one graph must be specified")

        for graph_id, spec in v.items():
            if ":" not in spec:
                raise ValueError(
                    f"Graph spec '{spec}' for '{graph_id}' must be in format 'file_path:export_symbol'"
                )

        return v

    @validator("env")
    def validate_env(cls, v):
        """Validate environment configuration."""
        if isinstance(v, str):
            # If it's a string, it should be a path to an env file
            if not v.endswith(".env"):
                raise ValueError("Environment file must have .env extension")
        elif isinstance(v, dict):
            # If it's a dict, validate it contains string values
            for key, value in v.items():
                if not isinstance(key, str):
                    raise ValueError("Environment variable keys must be strings")
        return v

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> LangGraphConfig:
        """Load configuration from langgraph.json file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if not config_path.name.endswith(".json"):
            raise ValueError("Configuration file must be a JSON file")

        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            return cls(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

    def load_environment(self, base_path: Path) -> None:
        """Load environment variables from configuration."""
        if not self.env:
            return

        if isinstance(self.env, str):
            # Load from .env file
            env_path = base_path / self.env
            if env_path.exists():
                from dotenv import load_dotenv

                load_dotenv(env_path)
        elif isinstance(self.env, dict):
            # Set environment variables directly
            for key, value in self.env.items():
                if isinstance(value, str):
                    os.environ[key] = value
                else:
                    os.environ[key] = json.dumps(value)
