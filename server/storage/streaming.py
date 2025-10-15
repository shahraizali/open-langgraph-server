from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when stream operations timeout."""

    pass


class AbortError(Exception):
    """Raised when stream operations are aborted."""

    pass


@dataclass
class Message:
    """Stream message data structure."""

    topic: str
    data: Any

    def __init__(self, topic: str, data: Any):
        self.topic = topic
        self.data = data


class Queue:
    """
    Message queue implementation
    Supports resumable streaming with message logging and listener management.
    """

    def __init__(self, resumable: bool = False):
        self.log: List[Message] = []
        self.listeners: List[asyncio.Future] = []
        self.next_id = 0
        self.resumable = resumable
        self._lock = threading.Lock()

    def push(self, item: Message) -> None:
        """Add a message to the queue and notify listeners."""
        with self._lock:
            self.log.append(item)
            # Notify all waiting listeners
            for listener in self.listeners[
                :
            ]:  # Copy list to avoid modification during iteration
                if not listener.done():
                    listener.set_result(self.next_id)
            self.listeners.clear()
            self.next_id += 1

    async def get(
        self,
        timeout: float = 0.5,
        last_event_id: Optional[str] = None,
        signal: Optional[asyncio.Event] = None,
    ) -> tuple[str, Message]:
        """
        Get the next message from the queue.
        Supports resumable streaming using last_event_id.
        """
        if self.resumable and last_event_id is not None:
            try:
                target_id = int(last_event_id) + 1
                if 0 <= target_id < len(self.log):
                    return str(target_id), self.log[target_id]
            except (ValueError, IndexError):
                pass

        # Check if we have messages available immediately
        with self._lock:
            if not self.resumable and self.log:
                next_id = self.next_id - len(self.log)
                message = self.log.pop(0)
                return str(next_id), message

        # Wait for new messages
        future = asyncio.get_event_loop().create_future()

        try:
            with self._lock:
                self.listeners.append(future)

            # Wait with timeout and cancellation support
            if signal:
                done, pending = await asyncio.wait(
                    [future, asyncio.create_task(signal.wait())],
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()

                if signal.is_set():
                    raise AbortError("Operation was cancelled")

                if not done:
                    raise TimeoutError("Timeout waiting for message")
            else:
                try:
                    await asyncio.wait_for(future, timeout=timeout)
                except asyncio.TimeoutError:
                    raise TimeoutError("Timeout waiting for message")

            # Get the result
            if future.done():
                idx = future.result()

                with self._lock:
                    if self.resumable:
                        if idx < len(self.log):
                            return str(idx), self.log[idx]
                    else:
                        if self.log:
                            next_id = self.next_id - len(self.log)
                            message = self.log.pop(0)
                            return str(next_id), message

            raise TimeoutError("No message available")

        except Exception:
            # Clean up listener on any error
            with self._lock:
                if future in self.listeners:
                    self.listeners.remove(future)
            raise


class CancellationAbortController:
    """
    Abortion controller for run cancellation
    """

    def __init__(self):
        self._aborted = False
        self._abort_reason: Optional[str] = None
        self._signal = asyncio.Event()

    def abort(self, reason: str = "interrupt") -> None:
        """Abort the operation with the given reason."""
        self._aborted = True
        self._abort_reason = reason
        self._signal.set()

    @property
    def signal(self) -> asyncio.Event:
        """Get the abort signal."""
        return self._signal

    @property
    def aborted(self) -> bool:
        """Check if the operation has been aborted."""
        return self._aborted

    @property
    def reason(self) -> Optional[str]:
        """Get the abort reason."""
        return self._abort_reason


class StreamManagerImpl:
    """
    Stream manager implementation
    Manages message queues and control signals for runs.
    """

    def __init__(self):
        self.readers: Dict[str, Queue] = {}
        self.control: Dict[str, CancellationAbortController] = {}
        self._lock = threading.Lock()

    def get_queue(
        self, run_id: str, if_not_found: str = "create", resumable: bool = False
    ) -> Optional[Queue]:
        """Get or create a queue for the given run ID."""
        with self._lock:
            if run_id not in self.readers:
                if if_not_found == "create":
                    self.readers[run_id] = Queue(resumable=resumable)
                else:
                    return None
            return self.readers[run_id]

    def get_control(self, run_id: str) -> Optional[CancellationAbortController]:
        """Get the control signal for the given run ID."""
        with self._lock:
            return self.control.get(run_id)

    def is_locked(self, run_id: str) -> bool:
        """Check if a run is currently locked (has active control)."""
        with self._lock:
            return run_id in self.control

    def lock(self, run_id: str) -> asyncio.Event:
        """Lock a run and return its abort signal."""
        with self._lock:
            if run_id in self.control:
                logger.warning(f"Run {run_id} already locked")
            controller = CancellationAbortController()
            self.control[run_id] = controller
            return controller.signal

    def unlock(self, run_id: str) -> None:
        """Unlock a run by removing its control signal."""
        with self._lock:
            self.control.pop(run_id, None)

    def cleanup_run(self, run_id: str) -> None:
        """Clean up all resources for a run."""
        with self._lock:
            self.readers.pop(run_id, None)
            self.control.pop(run_id, None)


class RunStreamImpl:
    """
    Run stream implementation
    Handles streaming of run execution events.
    """

    @staticmethod
    async def join(
        run_id: str,
        thread_id: Optional[str],
        options: Dict[str, Any],
        auth: Optional[Any] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Join a run stream and yield events.
        """
        ignore404 = options.get("ignore404", False)
        last_event_id = options.get("lastEventId")
        cancel_on_disconnect = options.get("cancelOnDisconnect")

        # Get the queue for this run
        queue = stream_manager.get_queue(
            run_id, if_not_found="create", resumable=last_event_id is not None
        )

        if not queue:
            if not ignore404:
                yield {
                    "event": "error",
                    "data": {
                        "error": "Run not found",
                        "message": f"Run {run_id} not found",
                    },
                }
            return

        # Set up cancellation signal if provided
        abort_signal = None
        if cancel_on_disconnect:
            if isinstance(cancel_on_disconnect, asyncio.Event):
                abort_signal = cancel_on_disconnect

        try:
            while True:
                try:
                    # Get next message from queue
                    event_id, message = await queue.get(
                        timeout=0.5, last_event_id=last_event_id, signal=abort_signal
                    )

                    last_event_id = event_id

                    # Check for control messages
                    if message.topic == f"run:{run_id}:control":
                        if message.data == "done":
                            break
                        continue

                    # Extract event name from topic
                    if message.topic.startswith(f"run:{run_id}:stream:"):
                        event_name = message.topic[len(f"run:{run_id}:stream:") :]
                        yield {
                            "id": event_id,
                            "event": event_name,
                            "data": message.data,
                        }

                except TimeoutError:
                    # Check if run still exists and is active
                    # For now, we'll continue - in a full implementation,
                    # we'd check the run status from storage
                    continue

                except AbortError:
                    logger.info(f"Stream for run {run_id} was aborted")
                    break

        except Exception as e:
            logger.error(f"Error in run stream for {run_id}: {e}")
            yield {
                "event": "error",
                "data": {"error": str(e), "message": f"Stream error for run {run_id}"},
            }

    @staticmethod
    async def publish(payload: Dict[str, Any]) -> None:
        """
        Publish an event to a run stream.
        """
        run_id = payload["runId"]
        event = payload["event"]
        data = payload["data"]
        resumable = payload.get("resumable", False)

        # Get or create queue for this run
        queue = stream_manager.get_queue(
            run_id, if_not_found="create", resumable=resumable
        )

        if queue:
            # Create message with proper topic format
            topic = f"run:{run_id}:stream:{event}"
            message = Message(topic=topic, data=data)
            queue.push(message)

            logger.debug(f"Published event '{event}' for run {run_id}")


# Global instances
stream_manager = StreamManagerImpl()
RunStream = RunStreamImpl()


def cleanup_stream(run_id: str) -> None:
    """Clean up stream resources for a run."""
    stream_manager.cleanup_run(run_id)


def send_control_message(run_id: str, message: str) -> None:
    """Send a control message to end or control a stream."""
    queue = stream_manager.get_queue(run_id, if_not_found="create", resumable=False)
    if queue:
        control_message = Message(topic=f"run:{run_id}:control", data=message)
        queue.push(control_message)
