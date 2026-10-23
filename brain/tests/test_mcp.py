"""The MCP client: namespacing, result size, and the bounds that keep one bad
server from parking an agent turn.

Everything here runs against a scripted MCP session object, so no child process
is spawned and no socket is opened.
"""

from __future__ import annotations

import asyncio

import pytest

import mcp_client
from memory import McpServer


# --- fakes -----------------------------------------------------------------

class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = ""
        self.inputSchema = {"type": "object", "properties": {}}


class _Listed:
    def __init__(self, tools: list[_FakeTool]) -> None:
        self.tools = tools


class _FakeSession:
    """An MCP session whose call_tool blocks until `release` is set."""

    def __init__(self, tools: list[str]) -> None:
        self._tools = [_FakeTool(t) for t in tools]
        self.release = asyncio.Event()

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> _Listed:
        return _Listed(self._tools)

    async def call_tool(self, name: str, arguments: dict) -> str:
        await self.release.wait()
        return f"{name} done"


class _FakeCM:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *exc) -> bool:
        return False


def _spec(name: str) -> McpServer:
    return McpServer(
        id=1, name=name, transport="stdio", command="python", args=[],
        url=None, env_ref=None, enabled=True,
    )


def _conn(name: str, tools: list[str], on_ready=None) -> mcp_client._ServerConn:
    conn = mcp_client._ServerConn(_spec(name), on_ready)
    conn.connected = True
    conn.tools = [_FakeTool(t) for t in tools]
    return conn


# --- MCP tool naming -------------------------------------------------------

def test_namespaced_names_are_unchanged_for_the_bundled_servers():
    assert mcp_client._tool_name("weather", "get_weather") == "mcp__weather__get_weather"


def test_a_long_server_and_tool_still_fit_the_api_name_limit():
    name = mcp_client._tool_name(
        "home-assistant-local-bridge",
        "get_climate_entity_current_temperature_setpoint",
    )
    assert len(name) <= mcp_client.MAX_TOOL_NAME
    assert name.startswith("mcp__home-assistant-local-b")


async def test_colliding_tool_names_are_disambiguated_not_overwritten():
    long_a = "a" * 60 + "_one"
    long_b = "a" * 60 + "_two"
    client = mcp_client.McpClient(memory=None)
    client._conns = [_conn("srv", [long_a, long_b])]
    client._rebuild_index()
    assert sorted(raw for _, raw in client._index.values()) == sorted([long_a, long_b])
    assert all(len(n) <= mcp_client.MAX_TOOL_NAME for n in client._index)


# --- results are bounded ---------------------------------------------------

def test_an_oversized_result_is_truncated_with_a_marker():
    out = mcp_client._clamp_result("tool", "x" * (mcp_client.MAX_RESULT_CHARS + 5000))
    assert len(out) < mcp_client.MAX_RESULT_CHARS + 200
    assert "[truncated, 5000 chars omitted]" in out


def test_a_normal_result_is_passed_through_untouched():
    assert mcp_client._clamp_result("tool", "it is 62 degrees") == "it is 62 degrees"


# --- the config knob actually reaches the child ----------------------------

def test_the_child_env_carries_the_default_location_knob(mem):
    from config import get_config

    get_config().set("DEFAULT_LOCATION", "Austin, Texas")
    assert mcp_client._child_env(None)["DEFAULT_LOCATION"] == "Austin, Texas"


# --- no abandoned futures --------------------------------------------------

async def test_a_call_queued_behind_a_reload_fails_instead_of_hanging():
    """The F29 path: request B is queued behind an in-flight A when the web UI
    reloads, so `_serve` exits without ever reaching B."""
    ready: list[bool] = []
    conn = _conn("weather", [], on_ready=lambda: ready.append(True))
    conn.connected = False
    session = _FakeSession(["get_weather"])
    conn._open_session = lambda: _FakeCM(session)

    conn.start()
    await conn.wait_ready(2.0)
    assert conn.connected
    assert ready == [True]      # a late connect re-indexes (F33)

    a = asyncio.create_task(conn.call("get_weather", {}))
    await asyncio.sleep(0.05)   # let the actor pick A up and block in call_tool
    b = asyncio.create_task(conn.call("get_weather", {}))
    await asyncio.sleep(0.05)

    stopping = asyncio.create_task(conn.stop())
    session.release.set()       # A finishes; _serve then sees _stop and exits

    assert await asyncio.wait_for(a, 2.0) == "get_weather done"
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(b, 2.0)
    await asyncio.wait_for(stopping, 2.0)


async def test_a_call_to_a_dead_server_is_rejected_immediately():
    conn = _conn("weather", ["get_weather"])
    conn.connected = False
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(conn.call("get_weather", {}), 2.0)
