"""Claude Buddy — robot-as-approver for Claude tool permissions (Phase: Buddy
Option C, Milestone 0).

A Claude Code `PreToolUse` hook POSTs a pending tool-permission request to the
brain's `/buddy/permission` endpoint (see webui/app.py + buddy_hook_client.py).
That call blocks here in `request_permission()` until the user **taps the head to
approve**. There is no deny gesture and no timeout: an un-tapped prompt stays open
until it's handled in the Claude session — at which point the hook process goes
away, the web route cancels this wait, and the robot UI clears. While pending, the
robot shows a "waiting" face + speech bubble, glances at the user, and (optionally)
speaks the tool name.

This is the transport-agnostic experience layer the plan calls for: it knows
nothing about HTTP or BLE. The HTTP route calls `request_permission()`; the
WebSocket event loop (agent_server) calls `attach`/`detach` and `approve()` when a
head tap arrives. A later BLE ingress (Option B) can drive the same methods.

Milestone 0 handles one pending permission at a time (serialized by a lock); the
priority model vs. live conversations is a Phase-1 concern.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from config import get_config

log = logging.getLogger("brain.buddy")

# Decisions we hand back to the Claude Code hook (its permissionDecision values).
# Only ALLOW (tap) and ASK (defer to the Claude session) are used — there is no
# deny gesture by design.
ALLOW = "allow"
ASK = "ask"

SpeakFn = Callable[[str], Awaitable[None]]


class Buddy:
    """Process-wide singleton (lifecycle like Memory / McpClient): built in
    agent_server.main(), attached to the active robot connection, queried by the
    web endpoint."""

    def __init__(self) -> None:
        self._ws: Any = None
        self._state: Any = None
        self._speak: SpeakFn | None = None
        self._pending: asyncio.Future[str] | None = None
        # One permission at a time in Milestone 0. Created lazily so
        # constructing Buddy() at import time (no running loop on py3.9) is safe.
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # -- connection lifecycle (called by agent_server.handle) --
    def attach(self, ws: Any, state: Any, speak: SpeakFn) -> None:
        self._ws, self._state, self._speak = ws, state, speak

    def detach(self, ws: Any) -> None:
        if self._ws is ws:
            self._ws = self._state = self._speak = None
            # Robot dropped mid-prompt: can't be tapped, so defer to the
            # Claude session rather than leaving the hook hanging.
            self._resolve(ASK)

    @property
    def connected(self) -> bool:
        return self._ws is not None

    @property
    def pending(self) -> bool:
        return self._pending is not None and not self._pending.done()

    # -- robot input → decision (called by agent_server event loop) --
    def approve(self) -> bool:
        """Approve the pending permission (a head tap). Returns True if there
        was something to resolve (so the caller swallows the tap instead of
        starting a voice turn). There is no deny gesture by design — an
        un-tapped prompt stays open until the Claude session is dealt with."""
        return self._resolve(ALLOW)

    def _resolve(self, decision: str) -> bool:
        if self._pending is not None and not self._pending.done():
            self._pending.set_result(decision)
            return True
        return False

    # -- HTTP/BLE ingress → blocking request (called by webui route) --
    async def request_permission(self, tool: str, hint: str) -> str:
        """Surface a pending tool permission on the robot and block until the
        user taps the head to approve. There is **no timeout and no deny** — the
        prompt stays open until a tap (→ allow) or until the caller goes away
        (the Claude session is dealt with directly): the awaiting task is then
        cancelled by the web route on client disconnect, which clears the robot
        UI. If Buddy is off or the robot is disconnected, returns 'ask'
        immediately so the normal Claude Code prompt handles it."""
        cfg = get_config()
        if not cfg.get("BUDDY_MODE") or not self.connected:
            log.info(
                "buddy: %s → ask (mode=%s connected=%s)",
                tool, cfg.get("BUDDY_MODE"), self.connected,
            )
            return ASK

        # Serialize: if a prompt is already up, queue behind it.
        async with self._get_lock():
            if not self.connected:  # robot dropped while we waited for the lock
                return ASK
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[str] = loop.create_future()
            self._pending = fut
            ws, state = self._ws, self._state
            log.info("buddy: awaiting tap-to-approve for %r (%s)", tool, hint[:80])
            try:
                await self._enter_waiting(ws, state, tool, hint)
                # Wait indefinitely for a tap; cancellation (client disconnect)
                # propagates through the finally, which clears the robot UI.
                decision = await fut
                log.info("buddy: %r → %s", tool, decision)
                return decision
            finally:
                self._pending = None
                await asyncio.shield(self._exit_waiting(ws, state))

    # -- robot experience (face / bubble / glance / voice) --
    async def _enter_waiting(
        self, ws: Any, state: Any, tool: str, hint: str
    ) -> None:
        cfg = get_config()
        await self._send(ws, {
            "cmd": "set_expression",
            "value": cfg.get("BUDDY_WAITING_EXPRESSION"),
        })
        await self._send(ws, {"cmd": "set_busy", "on": True})
        # A single glance toward the user to draw attention — reuse the idle
        # head choreography rather than reinventing servo math.
        try:
            if state is not None and hasattr(state, "behavior"):
                await state.behavior.glance_at_user(ws)
        except Exception:
            log.exception("buddy glance failed")
        # Speak the tool name so it's actionable without looking — but only if
        # the robot isn't mid-conversation (Milestone 0 keeps arbitration simple).
        if (
            cfg.get("BUDDY_SPEAK_PROMPTS")
            and self._speak is not None
            and not getattr(state, "listening", False)
            and not getattr(state, "speaking", False)
        ):
            try:
                await self._speak(f"Tap to approve {tool}.")
            except Exception:
                log.exception("buddy speak failed")

    async def _exit_waiting(self, ws: Any, state: Any) -> None:
        await self._send(ws, {"cmd": "set_busy", "on": False})
        await self._send(ws, {"cmd": "set_expression", "value": "neutral"})
        # The tap/wake word locally flips the firmware into LISTENING; since we
        # consumed the event for a decision (not a voice turn), put it back to
        # idle so the wake word re-arms.
        await self._send(ws, {"cmd": "stop_listening"})

    @staticmethod
    async def _send(ws: Any, cmd: dict[str, Any]) -> None:
        if ws is None:
            return
        try:
            await ws.send(json.dumps(cmd))
        except Exception:
            log.exception("buddy send failed: %s", cmd.get("cmd"))

    # -- web UI / status --
    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(get_config().get("BUDDY_MODE")),
            "connected": self.connected,
            "pending": self.pending,
        }
