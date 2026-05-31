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
themselves. `reload()`/`aclose()` just signal + await the owning tasks.

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
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from memory import McpServer, Memory

log = logging.getLogger("brain.mcp")

CONNECT_TIMEOUT_S = 20.0
CALL_TIMEOUT_S = 30.0

_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
_SECRET_SUFFIXES = ("_TOKEN", "_KEY", "_SECRET", "_PASSWORD", "_PASS")
_SECRET_KEYS = {"ANTHROPIC_API_KEY"}


def _sanitize(part: str) -> str:
    """Make a name fragment safe for an Anthropic tool name."""
    return _NAME_RE.sub("_", part)[:48]


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
    return base


class _ServerConn:
    """Owns one server connection in its own task."""

    def __init__(self, spec: McpServer) -> None:
        self.spec = spec
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
        if self._task is not None:
            try:
                await self._task
            except Exception:
                log.exception("mcp server task %s errored on stop", self.spec.name)

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
                log.info(
                    "mcp %s connected: %d tools (%s)",
                    self.spec.name, len(self.tools),
                    ", ".join(t.name for t in self.tools),
                )
                self._ready.set()
                await self._serve(session)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("mcp %s connection failed: %s", self.spec.name, self.error)
        finally:
            self.connected = False
            self._ready.set()

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
            except Exception as exc:  # noqa: BLE001 — surface to caller
                if not fut.done():
                    fut.set_exception(exc)

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self.connected:
            raise RuntimeError(f"server {self.spec.name} not connected")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._requests.put_nowait((tool_name, arguments, fut))
        return await fut


_STOP = object()


class _StdioSession:
    """async-cm: stdio transport + ClientSession entered/exited together."""

    def __init__(self, params: StdioServerParameters) -> None:
        self._params = params

    async def __aenter__(self) -> ClientSession:
        self._transport = stdio_client(self._params)
        read, write = await self._transport.__aenter__()
        self._session = ClientSession(read, write)
        return await self._session.__aenter__()

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
        return await self._session.__aenter__()

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
        self._conns = [_ServerConn(s) for s in specs]
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
        self._index = {}
        for c in self._conns:
            if not c.connected:
                continue
            for tool in c.tools:
                full = f"mcp__{_sanitize(c.spec.name)}__{_sanitize(tool.name)}"
                self._index[full] = (c, tool.name)

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
            result = await conn.call(raw, arguments)
        except Exception as exc:  # noqa: BLE001
            log.warning("mcp call %s failed: %s", name, exc)
            return f"The {conn.spec.name} tool failed: {exc}"
        return _flatten_result(result)

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
