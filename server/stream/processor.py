"""Stream processing module for LangGraph execution events."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from ..storage.streaming import RunStream

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Processes and handles LangGraph streaming events."""

    def __init__(self, run_id: str, resumable: bool = True):
        self.run_id = run_id
        self.resumable = resumable
        self.messages_state: Dict[str, Any] = {}
        self.completed_message_ids: Set[str] = set()

    async def process_stream_events(
        self, 
        graph, 
        input_data: Dict[str, Any], 
        config: Dict[str, Any], 
        stream_mode: List[str]
    ) -> None:
        """Process streaming events from LangGraph execution."""
        
        # Publish initial metadata event
        await self._publish_metadata()
        
        # Publish initial values event
        await self._publish_initial_values(input_data)

        logger.info(f"[STREAM DEBUG] --- START RUN {self.run_id} ---")
        logger.info(f"[STREAM DEBUG] Input data: {input_data}")
        logger.info(f"[STREAM DEBUG] Stream mode: {stream_mode}")

        event_count = 0
        async for event in graph.astream_events(
            input_data,
            config=config,
            version="v2",
            stream_mode=stream_mode,
        ):
            event_count += 1
            logger.info(f"[STREAM DEBUG] [EVENT #{event_count}]")

            # Skip hidden events
            if event.get("tags") and "langsmith:hidden" in event.get("tags", []):
                continue

            # Process different event types
            await self._handle_chain_stream_event(event, stream_mode)
            await self._handle_events_mode(event, stream_mode)
            await self._handle_chat_model_stream(event, stream_mode)
            await self._handle_messages_mode(event, stream_mode)

        # Handle fallback if insufficient events
        if event_count <= 1:
            await self._handle_fallback_stream(graph, input_data, config, stream_mode)

        logger.info(f"[STREAM DEBUG] --- END STREAM LOOP --- (Total events: {event_count})")

    async def _publish_metadata(self) -> None:
        """Publish metadata event."""
        await RunStream.publish({
            "runId": self.run_id,
            "event": "metadata",
            "data": {
                "run_id": self.run_id,
                "attempt": 1,
            },
            "resumable": self.resumable,
        })

    async def _publish_initial_values(self, input_data: Dict[str, Any]) -> None:
        """Publish initial values event."""
        initial_values = input_data.copy() if input_data else {}
        
        # Add default messages if not present
        if "messages" not in initial_values:
            initial_values["messages"] = []

        await RunStream.publish({
            "runId": self.run_id,
            "event": "values",
            "data": initial_values,
            "resumable": self.resumable,
        })

    async def _handle_chain_stream_event(self, event: Dict[str, Any], stream_mode: List[str]) -> None:
        """Handle on_chain_stream events."""
        if event.get("event") != "on_chain_stream":
            return

        chunk_data = event.get("data", {}).get("chunk")
        if not chunk_data:
            return

        # Unpack chunk data robustly
        ns, mode, chunk = self._unpack_chunk_data(chunk_data)
        
        logger.info(f"[STREAM DEBUG] [CHAIN_STREAM] ns: {ns}, mode: {mode}, chunk: {chunk}")

        # Handle debug events
        if mode == "debug":
            logger.info(f"[STREAM DEBUG] [CHAIN_STREAM] Debug event: {chunk}")
            return

        # Handle messages mode with messages-tuple
        if mode == "messages" and "messages-tuple" in stream_mode:
            await self._handle_messages_tuple_mode(event)
        elif stream_mode and mode in stream_mode:
            await self._emit_stream_event(mode, chunk, ns)

    def _unpack_chunk_data(self, chunk_data: Any) -> tuple[Optional[str], Optional[str], Any]:
        """Robustly unpack chunk data into namespace, mode, and chunk with proper subgraph support."""
        ns = None
        mode = None
        chunk = None
        
        if isinstance(chunk_data, (list, tuple)):
            if len(chunk_data) == 3:
                # Standard format: [namespace, mode, chunk]
                ns, mode, chunk = chunk_data
            elif len(chunk_data) == 2:
                # Two possibilities:
                # 1. Subgraph format: [namespace_list, chunk] 
                # 2. Standard format: [mode, chunk]
                first_item, second_item = chunk_data
                
                # Check if first item looks like a namespace (list or string with dots/slashes)
                if (isinstance(first_item, list) or 
                    (isinstance(first_item, str) and ('.' in first_item or '/' in first_item or ':' in first_item))):
                    # Subgraph format: namespace, chunk
                    ns = first_item
                    chunk = second_item
                    mode = "values"  # Default mode for subgraph chunks
                else:
                    # Standard format: mode, chunk
                    mode = first_item
                    chunk = second_item
            elif len(chunk_data) == 1:
                chunk = chunk_data[0]
            else:
                # Fallback for unexpected lengths
                chunk = chunk_data
        else:
            chunk = chunk_data
            
        return ns, mode, chunk

    async def _handle_messages_tuple_mode(self, event: Dict[str, Any]) -> None:
        """Handle messages mode for messages-tuple stream mode."""
        try:
            chunk_data = event.get("data", {}).get("chunk")[1][0]
            chunk_message = {
                "content": chunk_data.content,
                "additional_kwargs": {},
                "response_metadata": {},
                "type": "AIMessageChunk",
                "name": None,
                "id": chunk_data.id,
                "example": False,
                "tool_calls": [],
                "invalid_tool_calls": [],
                "usage_metadata": None,
                "tool_call_chunks": [],
            }

            await RunStream.publish({
                "runId": self.run_id,
                "event": "messages",
                "data": [chunk_message, {}],
                "resumable": self.resumable,
            })
            await asyncio.sleep(0.05)
        except (IndexError, AttributeError, KeyError) as e:
            logger.warning(f"Failed to process messages-tuple mode: {e}")

    async def _emit_stream_event(self, mode: str, chunk: Any, ns: Optional[str] = None) -> None:
        """Emit a stream event with proper namespace handling."""
        event_name = mode
        if ns:
            if isinstance(ns, list):
                # Handle list of namespace segments
                namespace_str = '|'.join(str(segment) for segment in ns)
                event_name = f"{mode}|{namespace_str}"
            elif isinstance(ns, str):
                # Handle string namespace (may contain separators)
                event_name = f"{mode}|{ns}"
            else:
                # Handle other namespace types
                event_name = f"{mode}|{str(ns)}"
            
        logger.info(f"[STREAM DEBUG] [CHAIN_STREAM] Emitting event: {event_name}")
        
        await RunStream.publish({
            "runId": self.run_id,
            "event": event_name,
            "data": chunk,
            "resumable": self.resumable,
        })

    async def _handle_events_mode(self, event: Dict[str, Any], stream_mode: List[str]) -> None:
        """Handle events mode."""
        if "events" not in stream_mode:
            return
            
        logger.info("[STREAM DEBUG] [EVENTS] Emitting events event")
        await RunStream.publish({
            "runId": self.run_id,
            "event": "events",
            "data": event,
            "resumable": self.resumable,
        })

    async def _handle_chat_model_stream(self, event: Dict[str, Any], stream_mode: List[str]) -> None:
        """Handle on_chat_model_stream events for incremental streaming."""
        if (event.get("event") != "on_chat_model_stream" or 
            "nostream" in event.get("tags", [])):
            return

        message_chunk = event.get("data", {}).get("chunk")
        if (not message_chunk or 
            not hasattr(message_chunk, "id") or 
            not message_chunk.id):
            return

        message_id = message_chunk.id
        if message_id not in self.messages_state:
            self.messages_state[message_id] = message_chunk
            logger.info(f"[STREAM DEBUG] [CHAT_MODEL] New message {message_id}")
        else:
            self.messages_state[message_id] = message_chunk

        logger.info("[STREAM DEBUG] [CHAT_MODEL] Emitting messages event for incremental chunk")

    async def _handle_messages_mode(self, event: Dict[str, Any], stream_mode: List[str]) -> None:
        """Handle messages mode with incremental streaming."""
        if "messages" not in stream_mode:
            return
            
        if (event.get("event") == "on_chain_stream" and 
            event.get("run_id") == self.run_id):
            
            chunk_data = event.get("data", {}).get("chunk")
            if chunk_data and len(chunk_data) >= 2:
                _, chunk = chunk_data
                chunk_messages = []
                
                if isinstance(chunk, dict) and "messages" in chunk:
                    chunk_messages = chunk["messages"]
                elif isinstance(chunk, list):
                    chunk_messages = chunk
                    
                if not isinstance(chunk_messages, list):
                    chunk_messages = [chunk_messages]
                    
                new_messages = []
                for message in chunk_messages:
                    if (hasattr(message, "id") and 
                        message.id and 
                        message.id not in self.completed_message_ids):
                        self.completed_message_ids.add(message.id)
                        new_messages.append(message)
                        
                if new_messages:
                    logger.info("[STREAM DEBUG] [MESSAGES] Emitting messages/complete event")
                    await RunStream.publish({
                        "runId": self.run_id,
                        "event": "messages/complete",
                        "data": new_messages,
                        "resumable": self.resumable,
                    })

    async def _handle_fallback_stream(
        self, 
        graph, 
        input_data: Dict[str, Any], 
        config: Dict[str, Any], 
        stream_mode: List[str]
    ) -> None:
        """Handle fallback streaming when astream_events doesn't yield enough events."""
        logger.warning("[STREAM DEBUG] astream_events insufficient, falling back to astream")
        
        await RunStream.publish({
            "runId": self.run_id,
            "event": "error",
            "data": {"error": "astream_events not working, falling back to astream"},
            "resumable": self.resumable,
        })

        try:
            async for chunk in graph.astream(input_data, config=config, stream_mode=stream_mode):
                logger.info(f"[STREAM DEBUG] [ASTREAM FALLBACK] Chunk: {chunk}")
                
                # Handle values event
                if self._is_values_chunk(chunk, stream_mode):
                    await self._handle_values_chunk(chunk)
                
                # Handle messages-tuple mode  
                if self._is_messages_chunk(chunk, stream_mode):
                    await self._handle_messages_chunk(chunk)
                    
        except Exception as e:
            logger.error(f"[STREAM ERROR] Error in astream fallback: {e}", exc_info=True)

    def _is_values_chunk(self, chunk: Any, stream_mode: List[str]) -> bool:
        """Check if chunk is a values chunk."""
        return (
            "values" in stream_mode and
            isinstance(chunk, (list, tuple)) and
            len(chunk) == 2 and
            isinstance(chunk[0], str) and
            chunk[0] == "values"
        )

    def _is_messages_chunk(self, chunk: Any, stream_mode: List[str]) -> bool:
        """Check if chunk is a messages chunk."""
        return (
            "messages-tuple" in stream_mode and
            isinstance(chunk, (list, tuple)) and
            len(chunk) == 2 and
            isinstance(chunk[0], str) and
            chunk[0] == "messages"
        )

    async def _handle_values_chunk(self, chunk: tuple) -> None:
        """Handle values chunk from fallback stream."""
        values_data = chunk[1]
        logger.info(f"[STREAM DEBUG] [ASTREAM FALLBACK] Values event with {len(values_data)} values")
        
        await RunStream.publish({
            "runId": self.run_id,
            "event": "values",
            "data": values_data,
            "resumable": self.resumable,
        })

    async def _handle_messages_chunk(self, chunk: tuple) -> None:
        """Handle messages chunk from fallback stream."""
        messages_data = chunk[1]
        logger.info("[STREAM DEBUG] [ASTREAM FALLBACK] Found messages chunk")
        # Messages processing can be added here if needed

    async def publish_final_values(self, final_values: Dict[str, Any]) -> None:
        """Publish final values event."""
        if final_values:
            await RunStream.publish({
                "runId": self.run_id,
                "event": "values",
                "data": final_values,
                "resumable": self.resumable,
            })

    async def publish_completion(self, status: str = "success") -> None:
        """Publish run completion event."""
        await RunStream.publish({
            "runId": self.run_id,
            "event": "end",
            "data": {"run_id": self.run_id, "status": status},
            "resumable": self.resumable,
        })

    async def publish_error(self, error_data: Any) -> None:
        """Publish error event."""
        await RunStream.publish({
            "runId": self.run_id,
            "event": "error",
            "data": error_data,
            "resumable": self.resumable,
        })