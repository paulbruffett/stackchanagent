"""MCP client — a second tool source for the Claude tool-use loop (Phase 9b).

Connects to the MCP servers in the registry (`memory.mcp_servers`), lists
their tools, and exposes them to the agent namespaced as
`mcp__<server>__<tool>`. The agent loop merges `tool_defs()` into its
`tools=` list and routes `tool_use` blocks whose name `is_mcp_tool()` to
`dispatch()`.

Concurrency model — one **owning task per server** ("connection actor"):
the MCP SDK (anyio) requires that a transport/session context be entered
and exited in the *same* task, or you get cross-task cancel-scope errors.
So each server's full lifecycle (open transport → initialize → serve calls
→ close) lives in a single dedicated task. Other tasks (agent turns, the
web UI Reload button) interact only by putting call requests on a queue
and by signalling stop — never by entering/exiting the contexts
themselves. `reload()`/`aclose()` just signal + await the owning tasks,
and cancel one that outstays STOP_TIMEOUT_S: cancellation is still
delivered *in* the owning task, so the contexts exit where they were
entered.

Secrets: `env_ref` names a single environment variable (loaded from
`.env`); for stdio servers only that one var is added back to an env that
has had secret-looking keys stripped, so a child server can't read the
brain's other secrets (e.g. ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import get_config
from memory import McpServer, Memory

log = logging.getLogger("brain.mcp")

CONNECT_TIMEOUT_S = 20.0
CALL_TIMEOUT_S = 30.0
# `_serve` already bounds each call; this bounds the *wait* for one, so a
# request the owning task never gets to (queued behind a reload's _STOP, or
# behind another session's call) fails the tool instead of parking the turn.
DISPATCH_TIMEOUT_S = CALL_TIMEOUT_S + 15.0
# How long stop() gives a server to wind down before it stops waiting on it.
STOP_TIMEOUT_S = 10.0
# The Messages API constrains a tool name to ^[a-zA-Z0-9_-]{1,64}$, and it
# rejects the whole `tools` array — every turn — if one name is too long.
MAX_TOOL_NAME = 64
_MAX_SERVER_FRAGMENT = 24
# A tool result is sized by an outside process, replayed on every later
# request in the session, and committed to memory.db.
MAX_RESULT_CHARS = 8000

_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
_SECRET_SUFFIXES = ("_TOKEN", "_KEY", "_SECRET", "_PASSWORD", "_PASS")
_SECRET_KEYS = {"ANTHROPIC_API_KEY"}


def _sanitize(part: str) -> str:
    """Make a name fragment safe for an Anthropic tool name."""
    return _NAME_RE.sub("_", part)


def _tool_name(server: str, tool: str) -> str:
    """Compose `mcp__<server>__<tool>` within the API's tool-name limit.

    The budget belongs to the whole name, not to each fragment: capping the
    fragments at 48 each let a 27-char server plus a 47-char tool compose an
    81-char name, which the API rejects for the entire request. The turn
    then lands in `_recover_api_error`, which only ever truncates *messages*
    — so every turn 400s twice and speaks the canned fallback until the
    operator disables that server."""
    sname = _sanitize(server)[:_MAX_SERVER_FRAGMENT]
    prefix = f"mcp__{sname}__"
    tname = _sanitize(tool)
    budget = MAX_TOOL_NAME - len(prefix)
    if len(tname) > budget or len(server) > _MAX_SERVER_FRAGMENT:
        log.warning(
            "mcp %s: tool name %r shortened to fit the API's %d-char limit",
            server, tool, MAX_TOOL_NAME,
        )
    return prefix + tname[:budget]


def _unique_name(name: str, taken: dict[str, Any]) -> str:
    """Two tools can land on one name after shortening (or after the
    character scrub); overwriting would make the first silently unreachable."""
    n = 2
    while True:
        suffix = f"_{n}"
        candidate = name[: MAX_TOOL_NAME - len(suffix)] + suffix
        if candidate not in taken:
            return candidate
        n += 1


def _child_env(env_ref: str | None) -> dict[str, str]:
    """A child environment with secret-looking vars stripped, plus the one
    env var this server is allowed to see re-added."""
    base = {
        k: v for k, v in os.environ.items()
        if k not in _SECRET_KEYS and not k.endswith(_SECRET_SUFFIXES)
    }
    if env_ref and env_ref in os.environ:
        base[env_ref] = os.environ[env_ref]
    elif env_ref:
        log.warning("env_ref %r not set in environment", env_ref)
    # Hot knobs a bundled server reads from its environment. Config lives in
    # SQLite, never in os.environ, so without this the weather server keeps
    # reporting its hardcoded default after the operator changes
    # DEFAULT_LOCATION in the console (mcp_servers/README.md documents the
    # knob as the source of truth).
    base["DEFAULT_LOCATION"] = str(get_config().get("DEFAULT_LOCATION"))
    return base


class _ServerConn:
    """Owns one server connection in its own task."""

    def __init__(
        self, spec: McpServer, on_ready: Callable[[], None] | None = None
    ) -> None:
        self.spec = spec
        self._on_ready = on_ready
        self.connected = False
        self.error: str | None = None
        self.tools: list[Any] = []          # mcp Tool objects
        self._requests: asyncio.Queue[Any] = asyncio.Queue()
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()
        self._task: asyncio.Task | None = None

    # -- lifecycle (called by McpClient) --
    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"mcp:{self.spec.name}")

    async def wait_ready(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
        except asyncio.TimeoutError:
            self.error = self.error or "connect timeout"

    async def stop(self) -> None:
        self._stop.set()
        self._requests.put_nowait(_STOP)
        task = self._task
        if task is None:
            return
        if await self._join(task, STOP_TIMEOUT_S):
            return
        # A call wedged inside the SDK (or a child that ignores its stdin
        # closing) must not hold /api/mcp/reload — or shutdown — open for
        # however long it takes. Cancellation is delivered *in* the owning
        # task, so the anyio transport still tears the child down from the
        # task that entered it, which is the one rule this module has.
        log.warning(
            "mcp %s did not stop within %.0fs; cancelling its task",
            self.spec.name, STOP_TIMEOUT_S,
        )
        task.cancel()
        if not await self._join(task, STOP_TIMEOUT_S):
            log.error(
                "mcp %s ignored cancellation; its child may still be running",
                self.spec.name,
            )

    async def _join(self, task: asyncio.Task, timeout: float) -> bool:
        """Await `task` for at most `timeout`s; True if it finished. Shielded
        so a timeout leaves the task running — the caller decides when to
        cancel, and an unshielded wait_for would cancel here and then block
        unbounded waiting for that cancellation to be honoured."""
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except asyncio.TimeoutError:
            return False
        except asyncio.CancelledError:
            if not task.cancelled():
                raise  # our own cancellation, not the server task's
        except Exception:
            log.exception("mcp server task %s errored on stop", self.spec.name)
        return True

    # -- the owning task --
    async def _run(self) -> None:
        try:
            async with self._open_session() as session:
                await asyncio.wait_for(session.initialize(), CONNECT_TIMEOUT_S)
                listed = await asyncio.wait_for(
                    session.list_tools(), CONNECT_TIMEOUT_S
                )
                self.tools = list(listed.tools)
                self.connected = True
                self.error = None   # else status() shows connected + stale error
                log.info(
                    "mcp %s connected: %d tools (%s)",
                    self.spec.name, len(self.tools),
                    ", ".join(t.name for t in self.tools),
                )
                self._ready.set()
                # A server that misses start()'s deadline still arrives here
                # with a live child and live tools; start() built its index
                # already, so without this callback those tools stay invisible
                # to the model until someone clicks Reload.
                if self._on_ready is not None:
                    self._on_ready()
                await self._serve(session)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("mcp %s connection failed: %s", self.spec.name, self.error)
        finally:
            self.connected = False
            self._ready.set()
            self._fail_pending()

    def _open_session(self):
        spec = self.spec
        if spec.transport == "stdio":
            if not spec.command:
                raise ValueError("stdio server needs a command")
            params = StdioServerParameters(
                command=spec.command,
                args=list(spec.args),
                env=_child_env(spec.env_ref),
            )
            return _StdioSession(params)
        if spec.transport == "http":
            if not spec.url:
                raise ValueError("http server needs a url")
            return _HttpSession(spec.url)
        raise ValueError(f"unknown transport: {spec.transport}")

    async def _serve(self, session: ClientSession) -> None:
        while not self._stop.is_set():
            req = await self._requests.get()
            if req is _STOP:
                break
            tool_name, arguments, fut = req
            try:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments), CALL_TIMEOUT_S
                )
                if not fut.done():
                    fut.set_result(result)
            except asyncio.CancelledError:
                # stop() gave up on us mid-call. The queue drain in _run can't
                # see this one — it is already dequeued — so fail it here, and
                # with a plain error: handing a caller CancelledError would
                # read as *its* task being cancelled.
                if not fut.done():
                    fut.set_exception(
                        RuntimeError(f"server {self.spec.name} stopped")
                    )
                raise
            except Exception as exc:  # noqa: BLE001 — surface to caller
                if not fut.done():
                    fut.set_exception(exc)

    def _fail_pending(self) -> None:
        """Fail every queued request as the owning task dies. Nothing else can
        ever complete those futures — `_serve` drops whatever is still queued
        when it exits — and an abandoned `await fut` in `call()` parks the
        whole agent turn: `_run_loop` never reaches its finally, so busy is
        never cleared and the firmware sits in the thinking state (the M6.8
        turn-state disagreement, reached from the brain side)."""
        while True:
            try:
                req = self._requests.get_nowait()
            except asyncio.QueueEmpty:
                return
            if req is _STOP:
                continue
            fut = req[2]
            if not fut.done():
                fut.set_exception(
                    RuntimeError(f"server {self.spec.name} stopped")
                )

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self.connected:
            raise RuntimeError(f"server {self.spec.name} not connected")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._requests.put_nowait((tool_name, arguments, fut))
        # `_serve` bounds the call itself; this bounds the queue wait, so a
        # request enqueued behind a reload's _STOP can only ever be slow.
        try:
            return await asyncio.wait_for(fut, DISPATCH_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"server {self.spec.name} did not answer within "
                f"{DISPATCH_TIMEOUT_S:.0f}s"
            ) from None


_STOP = object()


class _StdioSession:
    """async-cm: stdio transport + ClientSession entered/exited together."""

    def __init__(self, params: StdioServerParameters) -> None:
        self._params = params

    async def __aenter__(self) -> ClientSession:
        self._transport = stdio_client(self._params)
        read, write = await self._transport.__aenter__()
        self._session = ClientSession(read, write)
        try:
            return await self._session.__aenter__()
        except BaseException as exc:
            # `async with` only calls __aexit__ on a context manager whose
            # __aenter__ *returned*, so a failure here would leave nobody
            # owning the transport — and for stdio that means the child we
            # just spawned is never terminated or reaped.
            await self._transport.__aexit__(type(exc), exc, exc.__traceback__)
            raise

    async def __aexit__(self, *exc: Any) -> None:
        try:
            await self._session.__aexit__(*exc)
        finally:
            await self._transport.__aexit__(*exc)


class _HttpSession:
    def __init__(self, url: str) -> None:
        self._url = url

    async def __aenter__(self) -> ClientSession:
        from mcp.client.streamable_http import streamablehttp_client
        self._transport = streamablehttp_client(self._url)
        read, write, *_ = await self._transport.__aenter__()
        self._session = ClientSession(read, write)
        try:
            return await self._session.__aenter__()
        except BaseException as exc:  # see _StdioSession.__aenter__
            await self._transport.__aexit__(type(exc), exc, exc.__traceback__)
            raise

    async def __aexit__(self, *exc: Any) -> None:
        try:
            await self._session.__aexit__(*exc)
        finally:
            await self._transport.__aexit__(*exc)


class McpClient:
    """Process-wide singleton (lifecycle like Memory): built in
    agent_server.main(), started before serving, reloaded from the web UI."""

    def __init__(self, memory: Memory) -> None:
        self._memory = memory
        self._conns: list[_ServerConn] = []
        # namespaced tool name -> (conn, raw tool name)
        self._index: dict[str, tuple[_ServerConn, str]] = {}

    async def start(self) -> None:
        specs = [s for s in self._memory.list_mcp_servers() if s.enabled]
        self._conns = [_ServerConn(s, self._rebuild_index) for s in specs]
        for c in self._conns:
            c.start()
        await asyncio.gather(*(c.wait_ready(CONNECT_TIMEOUT_S) for c in self._conns))
        self._rebuild_index()
        log.info(
            "mcp client started: %d/%d servers connected, %d tools",
            sum(c.connected for c in self._conns), len(self._conns),
            len(self._index),
        )

    async def aclose(self) -> None:
        await asyncio.gather(*(c.stop() for c in self._conns), return_exceptions=True)
        self._conns = []
        self._index = {}

    async def reload(self) -> None:
        await self.aclose()
        await self.start()

    def _rebuild_index(self) -> None:
        index: dict[str, tuple[_ServerConn, str]] = {}
        for c in self._conns:
            if not c.connected:
                continue
            for tool in c.tools:
                full = _tool_name(c.spec.name, tool.name)
                if full in index:
                    full = _unique_name(full, index)
                    log.warning(
                        "mcp %s: tool %r collides with another name; "
                        "exposing it as %r", c.spec.name, tool.name, full,
                    )
                index[full] = (c, tool.name)
        self._index = index

    # -- agent-facing --
    def tool_defs(self) -> list[dict[str, Any]]:
        defs = []
        for full, (conn, raw) in self._index.items():
            tool = next((t for t in conn.tools if t.name == raw), None)
            if tool is None:
                continue
            schema = tool.inputSchema or {"type": "object", "properties": {}}
            defs.append({
                "name": full,
                "description": (tool.description or "")[:1024]
                + f"  (via MCP server '{conn.spec.name}')",
                "input_schema": schema,
            })
        return defs

    def is_mcp_tool(self, name: str) -> bool:
        return name in self._index

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        entry = self._index.get(name)
        if entry is None:
            return f"Unknown MCP tool {name}"
        conn, raw = entry
        try:
            out = _flatten_result(await conn.call(raw, arguments))
        except Exception as exc:  # noqa: BLE001
            log.warning("mcp call %s failed: %s", name, exc)
            out = f"The {conn.spec.name} tool failed: {exc}"
        return _clamp_result(name, out)

    # -- web UI --
    def status(self) -> list[dict[str, Any]]:
        out = []
        for c in self._conns:
            out.append({
                "name": c.spec.name,
                "enabled": c.spec.enabled,
                "connected": c.connected,
                "error": c.error,
                "tools": [t.name for t in c.tools],
            })
        return out


def _clamp_result(name: str, text: str) -> str:
    """Bound what a tool hands back. The size is chosen entirely by an outside
    process, and the string is replayed on every later request of the session
    *and* committed to memory.db — so one multi-MB dump blows the context
    window (and `_truncate_to_current_exchange` keeps the exchange holding it,
    costing two turns) or, if merely large, inflates every future prompt."""
    if len(text) <= MAX_RESULT_CHARS:
        return text
    dropped = len(text) - MAX_RESULT_CHARS
    log.warning("%s returned %d chars; truncated %d away", name, len(text), dropped)
    return text[:MAX_RESULT_CHARS] + f"\n... [truncated, {dropped} chars omitted]"


def _flatten_result(result: Any) -> str:
    """CallToolResult -> a plain string for the tool_result content."""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    text = "\n".join(parts).strip()
    if getattr(result, "isError", False):
        return f"[tool error] {text or 'the tool reported an error'}"
    return text or "(the tool returned no text)"
