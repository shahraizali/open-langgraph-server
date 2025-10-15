"""Pydantic models for storage operations."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from ..schemas import StreamMode


class RunnableConfig(Dict[str, Any]):
    """Runnable configuration"""

    pass


class RunKwargs:
    """Run execution parameters"""

    def __init__(
        self,
        input: Optional[Any] = None,
        command: Optional[Any] = None,
        stream_mode: Optional[List[StreamMode]] = None,
        interrupt_before: Optional[Union[Literal["*"], List[str]]] = None,
        interrupt_after: Optional[Union[Literal["*"], List[str]]] = None,
        config: Optional[RunnableConfig] = None,
        subgraphs: bool = False,
        resumable: bool = False,
        temporary: bool = False,
        webhook: Optional[Any] = None,
        feedback_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        self.input = input
        self.command = command
        self.stream_mode = stream_mode or ["values"]
        self.interrupt_before = interrupt_before
        self.interrupt_after = interrupt_after

        # Ensure config is always a proper dictionary with configurable key
        if config is None:
            self.config = {"configurable": {}}
        elif isinstance(config, dict):
            self.config = config.copy()
            # Ensure configurable key exists and is a dict
            if "configurable" not in self.config or self.config["configurable"] is None:
                self.config["configurable"] = {}
            elif not isinstance(self.config["configurable"], dict):
                self.config["configurable"] = {}
            else:
                # Filter out None values from configurable to prevent LangGraph errors
                self.config["configurable"] = {
                    k: v
                    for k, v in self.config["configurable"].items()
                    if v is not None
                }
        else:
            # Fallback for unexpected config types
            self.config = {"configurable": {}}

        self.subgraphs = subgraphs
        self.resumable = resumable
        self.temporary = temporary
        self.webhook = webhook
        self.feedback_keys = feedback_keys
        self.extra = kwargs
