"""Web console boundary tests.

The console's mutating half executes what it stores — an MCP row is a command
line the brain later spawns as its own user — so these cover the gate around
it (shared token, Host check) and the two registry validations, plus the
hidden-config-key path that used to persist a write and then 500.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from config import get_config
from webui.app import TOKEN_HEADER, create_app

TOKEN = "test-console-token"
AUTH = {TOKEN_HEADER: TOKEN}
# An IP literal for the Host header: that is how the console is really
# reached, and it can't be DNS-rebound.
BASE = "http://192.168.1.9:8080"


@pytest.fixture
def client(mem):
    with TestClient(create_app(mem, get_config(), token=TOKEN),
                    base_url=BASE) as c:
        yield c


def test_api_needs_the_token(client):
    assert client.get("/api/config").status_code == 401
    assert client.get("/api/config", headers=AUTH).status_code == 200


def test_spa_shell_stays_public(client):
    # app.js has to load before it can present a token.
    assert client.get("/").status_code == 200


def test_foreign_host_is_refused_even_with_the_token(mem):
    with TestClient(create_app(mem, get_config(), token=TOKEN),
                    base_url="http://evil-attacker-domain.example.com:8080") as c:
        assert c.get("/api/config", headers=AUTH).status_code == 403


def test_mcp_env_ref_cannot_name_the_brains_own_key(client, mem):
    r = client.post("/api/mcp/servers", headers=AUTH, json={
        "name": "x", "transport": "stdio", "command": "/bin/sh",
        "args": ["-c", "exfiltrate"], "env_ref": "ANTHROPIC_API_KEY",
    })
    assert r.status_code == 400
    assert mem.list_mcp_servers() == []


def test_mcp_env_ref_still_allows_a_server_secret(client, mem):
    r = client.post("/api/mcp/servers", headers=AUTH, json={
        "name": "hue", "transport": "stdio", "command": "python",
        "env_ref": "HUE_TOKEN",
    })
    assert r.status_code == 200
    assert [s.env_ref for s in mem.list_mcp_servers()] == ["HUE_TOKEN"]


def test_unknown_mcp_transport_is_refused(client, mem):
    r = client.post("/api/mcp/servers", headers=AUTH,
                    json={"name": "x", "transport": "carrier-pigeon"})
    assert r.status_code == 400
    assert mem.list_mcp_servers() == []


def test_hidden_config_key_saves_without_a_500(client):
    # SUMMARIZE_SYSTEM is hidden from describe(), which the response used to
    # scan for the restart flag — StopIteration inside the coroutine, 500 to
    # the caller, value already written.
    r = client.put("/api/config", headers=AUTH,
                   json={"key": "SUMMARIZE_SYSTEM", "value": "be terse"})
    assert r.status_code == 200
    assert r.json() == {"key": "SUMMARIZE_SYSTEM", "value": "be terse",
                        "restart": False}
    assert get_config().get("SUMMARIZE_SYSTEM") == "be terse"


def test_reset_resyncs_the_live_session(mem):
    calls = []

    async def resync() -> int:
        calls.append(1)
        return 1

    app = create_app(mem, get_config(), token=TOKEN, resync_sessions=resync)
    with TestClient(app, base_url=BASE) as c:
        mem.append_turns([{"role": "user", "content": "hello"}])
        r = c.post("/api/memories/reset", headers=AUTH)
        assert r.json() == {"ok": True, "deleted": 1, "live_synced": 1}
    assert calls == [1]
