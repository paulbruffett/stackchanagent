"""Supervised background-task spawning (M6.3).

A bare ``asyncio.create_task`` is two footguns at once: the loop keeps only a
*weak* reference to the task (so a fire-and-forget task can be garbage-collected
mid-flight), and any exception it raises is swallowed into a silent
"Task exception was never retrieved" — which is exactly how a background turn
(proactive greeting, summarizer fold) can vanish or wedge the session loop.

``spawn(coro, name)`` wraps ``create_task`` with a strong reference plus a
done-callback that logs any exception, so no background work fails silently.
Route every fire-and-forget through it; tasks that are explicitly tracked and
awaited/cancelled by their caller don't need it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine

log = logging.getLogger("brain.tasks")

# Strong refs to in-flight background tasks. asyncio only holds a weak
# reference, so without this a task with no other referent can be collected
# before it finishes. Discarded in the done-callback.
_BACKGROUND: set[asyncio.Task[Any]] = set()


def spawn(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
    """Schedule ``coro`` as a supervised background task. Any exception it
    raises is logged (never silently dropped); normal completion and
    cancellation are quiet. Returns the Task so the caller may still
    track/cancel it."""
    task = asyncio.create_task(coro, name=name)
    _BACKGROUND.add(task)
    task.add_done_callback(_on_done)
    return task


def _on_done(task: asyncio.Task[Any]) -> None:
    _BACKGROUND.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error("background task %r failed", task.get_name(), exc_info=exc)
