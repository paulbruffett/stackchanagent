"""The two outside-the-brain tool sources: namespacing, result size, and the
bounds that keep one bad server from parking an agent turn.

Everything here runs against fakes — a scripted MCP session object and a
scripted JSON-RPC endpoint — so no child process is spawned and no socket is
opened.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import a2a_client
import mcp_client
from memory import A2aServer, McpServer


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


def _a2a_conn(url: str = "http://192.168.4.30:8080") -> a2a_client._A2aConn:
    return a2a_client._A2aConn(
        A2aServer(id=1, name="hermes", url=url, env_ref=None, enabled=True)
    )


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _PollHttp:
    """A tasks/get endpoint that answers `payload` after `delay` seconds."""

    def __init__(self, payload: dict, delay: float = 0.0) -> None:
        self.payload = payload
        self.delay = delay
        self.polls = 0

    async def post(self, url, json, headers, timeout):  # noqa: A002
        self.polls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return _Resp(self.payload)


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

@pytest.mark.parametrize("clamp", [mcp_client._clamp_result, a2a_client._clamp_result])
def test_an_oversized_result_is_truncated_with_a_marker(clamp):
    out = clamp("tool", "x" * (mcp_client.MAX_RESULT_CHARS + 5000))
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


# --- A2A: the bearer token stays on the registered origin ------------------

CARD_URL = "http://192.168.4.30:8080/.well-known/agent.json"


def test_a_relative_card_endpoint_resolves_normally():
    assert _a2a_conn()._endpoint(CARD_URL, "/rpc") == "http://192.168.4.30:8080/rpc"


def test_an_off_origin_card_endpoint_is_pinned_back_to_the_registry():
    conn = _a2a_conn()
    assert conn._endpoint(CARD_URL, "https://attacker.example/collect") == (
        "http://192.168.4.30:8080/collect"
    )


# --- A2A: task polling is bounded ------------------------------------------

WORKING = {"kind": "task", "id": "t1", "status": {"state": "working"}}


async def test_task_polling_stops_at_a_wall_clock_deadline(monkeypatch):
    """Each poll's round trip counts against the budget, not just the sleep —
    otherwise 55s of budget buys 37 polls of up to a minute each."""
    monkeypatch.setattr(a2a_client, "POLL_TIMEOUT_S", 0.2)
    monkeypatch.setattr(a2a_client, "POLL_INTERVAL_S", 0.01)
    http = _PollHttp({"jsonrpc": "2.0", "result": WORKING}, delay=0.05)

    started = time.monotonic()
    out = await _a2a_conn()._await_task(http, "http://192.168.4.30:8080/rpc", WORKING)
    elapsed = time.monotonic() - started

    assert out == WORKING
    assert elapsed < 1.0            # ~20 polls x 0.05s under the old code
    assert http.polls <= 6


async def test_an_error_envelope_ends_polling(monkeypatch):
    monkeypatch.setattr(a2a_client, "POLL_INTERVAL_S", 0.01)
    http = _PollHttp({"jsonrpc": "2.0", "error": {"code": -32001, "message": "gone"}})

    out = await _a2a_conn()._await_task(http, "http://192.168.4.30:8080/rpc", WORKING)

    assert out == WORKING
    assert http.polls == 1          # not re-polled until the budget runs out
