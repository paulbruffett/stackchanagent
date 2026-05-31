"""Live log + voice-transaction feeds for the web UI.

`Broadcaster` keeps a bounded scrollback `deque` and fans new items out
to any number of subscriber `asyncio.Queue`s. Publishing is thread-safe:
log records may be emitted from worker threads (the brain runs STT/TTS
via `asyncio.to_thread`), so we hop onto the event loop with
`call_soon_threadsafe` before touching the asyncio queues.

Two instances are exported: `LOGS` (fed by `WebUILogHandler`) and
`TURNS` (fed by the agent-turn recorder in agent_server).
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any


class Broadcaster:
    def __init__(self, maxlen: int = 500) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the event loop so thread-emitted items can be delivered."""
        self._loop = loop

    def scrollback(self) -> list[dict[str, Any]]:
        return list(self._buffer)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def publish(self, item: dict[str, Any]) -> None:
        """Append to scrollback and deliver to subscribers. Safe to call
        from any thread."""
        self._buffer.append(item)
        if not self._subscribers:
            return
        loop = self._loop
        if loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._fanout(item)
        else:
            loop.call_soon_threadsafe(self._fanout, item)

    def _fanout(self, item: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                # Slow consumer — drop the oldest, then enqueue.
                try:
                    q.get_nowait()
                    q.put_nowait(item)
                except Exception:
                    pass


LOGS = Broadcaster(maxlen=500)
TURNS = Broadcaster(maxlen=200)


class WebUILogHandler(logging.Handler):
    """Logging handler that publishes records to the LOGS broadcaster."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            item = {
                "ts": record.created,
                "level": record.levelname,
                "name": record.name,
                "msg": record.getMessage(),
            }
        except Exception:
            return
        LOGS.publish(item)


def publish_turn(turn: dict[str, Any]) -> None:
    TURNS.publish(turn)
