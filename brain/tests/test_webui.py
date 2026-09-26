"""Web console boundary tests.

The console's mutating half executes what it stores — an MCP row is a command
line the brain later spawns as its own user — so these cover the gate around
it (shared token, Host check) and the two registry validations, plus the
hidden-config-key path that used to persist a write and then 500.

Driven through httpx's ASGI transport rather than fastapi's TestClient: the
console really runs in-process on the agent's own event loop, and TestClient
would instead hand the app to a worker thread, where Memory's sqlite
connection (check_same_thread defaults to True) raises ProgrammingError —
which the endpoints then swallow into a 400. That is a lie about a topology
the brain never has.
"""

from __future__ import annotations

import httpx
import pytest

from config import SPECS, get_config
from webui.app import TOKEN_HEADER, create_app

TOKEN = "test-console-token"
AUTH = {TOKEN_HEADER: TOKEN}
# An IP literal for the Host header: that is how the console is really
# reached, and it can't be DNS-rebound.
BASE = "http://192.168.1.9:8080"


def _client(app, base_url: str = BASE) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


@pytest.fixture
async def client(mem):
    async with _client(create_app(mem, get_config(), token=TOKEN)) as c:
        yield c


async def test_api_needs_the_token(client):
    assert (await client.get("/api/config")).status_code == 401
    assert (await client.get("/api/config", headers=AUTH)).status_code == 200


async def test_spa_shell_stays_public(client):
    # app.js has to load before it can present a token.
    assert (await client.get("/")).status_code == 200


async def test_foreign_host_is_refused_even_with_the_token(mem):
    app = create_app(mem, get_config(), token=TOKEN)
    async with _client(app, "http://evil-attacker-domain.example.com:8080") as c:
        assert (await c.get("/api/config", headers=AUTH)).status_code == 403


async def test_mcp_env_ref_cannot_name_the_brains_own_key(client, mem):
    r = await client.post("/api/mcp/servers", headers=AUTH, json={
        "name": "x", "transport": "stdio", "command": "/bin/sh",
        "args": ["-c", "exfiltrate"], "env_ref": "ANTHROPIC_API_KEY",
    })
    assert r.status_code == 400
    assert mem.list_mcp_servers() == []


async def test_mcp_env_ref_still_allows_a_server_secret(client, mem):
    r = await client.post("/api/mcp/servers", headers=AUTH, json={
        "name": "home-assistant", "transport": "stdio", "command": "python",
        "env_ref": "HA_TOKEN",
    })
    assert r.status_code == 200
    assert [s.env_ref for s in mem.list_mcp_servers()] == ["HA_TOKEN"]


async def test_unknown_mcp_transport_is_refused(client, mem):
    r = await client.post("/api/mcp/servers", headers=AUTH,
                          json={"name": "x", "transport": "carrier-pigeon"})
    assert r.status_code == 400
    assert mem.list_mcp_servers() == []


async def test_hidden_config_key_saves_without_a_500(client):
    # SUMMARIZE_SYSTEM is hidden from describe(), which the response used to
    # scan for the restart flag — StopIteration inside the coroutine, 500 to
    # the caller, value already written.
    r = await client.put("/api/config", headers=AUTH,
                         json={"key": "SUMMARIZE_SYSTEM", "value": "be terse"})
    assert r.status_code == 200
    assert r.json() == {"key": "SUMMARIZE_SYSTEM", "value": "be terse",
                        "restart": False}
    assert get_config().get("SUMMARIZE_SYSTEM") == "be terse"


async def test_reset_resyncs_the_live_session(mem):
    calls = []

    async def resync() -> int:
        calls.append(1)
        return 1

    app = create_app(mem, get_config(), token=TOKEN, resync_sessions=resync)
    async with _client(app) as c:
        mem.append_turns([{"role": "user", "content": "hello"}])
        r = await c.post("/api/memories/reset", headers=AUTH)
        assert r.json() == {"ok": True, "deleted": 1, "live_synced": 1}
    assert calls == [1]


# --- OpenRouter model suggestions --------------------------------------------

_CATALOG = {"data": [
    {"id": "z/tools-model", "name": "Z", "context_length": 8000,
     "pricing": {"prompt": "0.1"}, "supported_parameters": ["tools", "max_tokens"]},
    {"id": "a/no-tools", "name": "A", "context_length": 4000,
     "pricing": {}, "supported_parameters": ["max_tokens"]},
    {"id": "b/tools-too", "name": "B", "context_length": 128000,
     "pricing": {"prompt": "0"}, "supported_parameters": ["tool_choice", "tools"]},
    {"id": "c/no-params", "name": "C"},
]}


async def test_models_lists_only_tool_capable_models_sorted_and_cached(
    client, monkeypatch
):
    import webui.app as webui_app

    hits = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        return httpx.Response(200, json=_CATALOG)

    monkeypatch.setattr(webui_app, "_MODELS_TRANSPORT", httpx.MockTransport(handler))
    r = await client.get("/api/models", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == [
        {"id": "b/tools-too", "name": "B", "context_length": 128000,
         "pricing": {"prompt": "0"}},
        {"id": "z/tools-model", "name": "Z", "context_length": 8000,
         "pricing": {"prompt": "0.1"}},
    ]
    assert hits == ["https://openrouter.ai/api/v1/models"]
    await client.get("/api/models", headers=AUTH)
    assert len(hits) == 1  # served from the cache


async def test_models_degrades_to_an_empty_list(client, monkeypatch):
    import webui.app as webui_app

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    monkeypatch.setattr(webui_app, "_MODELS_TRANSPORT", httpx.MockTransport(handler))
    r = await client.get("/api/models", headers=AUTH)
    assert r.status_code == 200 and r.json() == []


async def test_turn_listing_carries_tool_calls_and_legacy_rows(client, mem):
    call = {"id": "c1", "type": "function",
            "function": {"name": "look_at", "arguments": "{}"}}
    mem.append_turns([
        {"role": "assistant", "content": [{"type": "text", "text": "legacy"}]},
        {"role": "user", "content": "look left"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ])
    turns = (await client.get("/api/memories/turns", headers=AUTH)).json()["turns"]
    assert turns[0]["content"] == [{"type": "text", "text": "legacy"}]
    assert turns[2]["tool_calls"] == [call]
    assert turns[3] == {"id": 4, "role": "tool", "tool_call_id": "c1", "content": "ok"}


# --- model / effort knobs are validated ---------------------------------------

@pytest.mark.parametrize("key, value", [
    ("MODEL", "claude-haiku-4-5"),        # an Anthropic-era id: no vendor
    ("MODEL", ""),
    ("MODEL", "openai/"),
    ("SUMMARY_MODEL", "gpt 5"),
    ("REASONING_EFFORT", "extreme"),
])
async def test_bad_model_or_effort_is_a_400(client, key, value):
    r = await client.put("/api/config", headers=AUTH, json={"key": key, "value": value})
    assert r.status_code == 400
    assert get_config().get(key) == SPECS[key].default


@pytest.mark.parametrize("key, value, stored", [
    ("MODEL", " anthropic/claude-haiku-4.5 ", "anthropic/claude-haiku-4.5"),
    ("SUMMARY_MODEL", "", ""),                 # empty = use MODEL
    ("REASONING_EFFORT", "HIGH", "high"),
    ("REASONING_EFFORT", "", ""),
])
async def test_good_model_or_effort_is_stored_normalised(client, key, value, stored):
    r = await client.put("/api/config", headers=AUTH, json={"key": key, "value": value})
    assert r.status_code == 200
    assert get_config().get(key) == stored


def test_stale_stored_model_is_ignored_on_reload(mem):
    # memory.db on the Jetson may still hold MODEL=claude-haiku-4-5 from the
    # Anthropic days; reload must fall back to the default, not 400 each turn.
    mem.set_config("MODEL", "claude-haiku-4-5")
    mem.set_config("REASONING_EFFORT", "turbo")
    cfg = get_config()
    cfg.reload()
    assert cfg.get("MODEL") == SPECS["MODEL"].default
    assert cfg.get("REASONING_EFFORT") == "low"
