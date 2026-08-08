"""FastAPI app for the brain web console (Phase 9a).

Mounted in-process by agent_server.main() and served on a separate port.
Shares the live `memory` and `Config` singletons, so edits take effect on
the running agent (hot knobs immediately; restart knobs on next start).

Authenticated with a shared secret. This is not a login system — it is the
recognition that POST /api/mcp/servers persists a command line that the MCP
client then *executes* as the brain's user, so an open console is a remote
shell for anything that can reach port 8080. agent_server passes the token
(CONSOLE_TOKEN in .env, else one minted per run and logged); the browser
sends it as a header, the log/turn WebSockets as a query param.
"""

from __future__ import annotations

import ipaddress
import logging
import secrets
import socket
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from anthropic import AsyncAnthropic

from claude_agent import (
    DEFAULT_SYSTEM_PROMPT,
    consolidate_facts,
    repair_memory,
    summarize_backlog,
)
from config import SPECS, Config
from memory import Memory
from webui.logbuf import LOGS, TURNS, Broadcaster

STATIC_DIR = Path(__file__).parent / "static"

log = logging.getLogger("brain.webui")

TOKEN_HEADER = "x-stackchan-token"

# `env_ref` names the one .env variable a child MCP server / A2A endpoint is
# allowed to see, which is the point of the field (HUE_TOKEN for the Hue
# server). What it must never name is a credential the *brain* itself holds:
# mcp_client._child_env strips secret-looking vars from the child environment
# and then re-adds exactly the named one, so `env_ref: ANTHROPIC_API_KEY`
# hands our own key to whatever command that registry row launches.
PROTECTED_ENV = {"ANTHROPIC_API_KEY", "HUME_API_KEY", "CONSOLE_TOKEN"}

# The only two mcp_client._open_session knows how to open. Anything else was
# accepted by the registry and then failed at connect time with a ValueError
# the operator only ever saw as a red dot.
MCP_TRANSPORTS = ("stdio", "http")


def _host_allowed(host_header: str) -> bool:
    """DNS-rebinding backstop for the Host header.

    The token is the real control; this stops an attacker's domain, rebound
    to the Jetson's address, from being *same-origin* with the console at all.
    So it rejects registrable names while passing every way the console is
    actually reached: an IP literal can't be rebound, nor can localhost or an
    mDNS `.local` name, and a router-assigned name (`orin.lan`) still matches
    on this host's own name.
    """
    host = host_header.strip().lower()
    if not host:
        return True
    if host.startswith("["):                       # [::1]:8080
        host = host[1:host.find("]")] if "]" in host else host[1:]
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    if host in ("localhost", "::1"):
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    return (host.endswith(".local")
            or host.split(".")[0] == socket.gethostname().lower().split(".")[0])


def _token_ok(presented: str | None, token: str) -> bool:
    # Compare as bytes: compare_digest refuses non-ASCII str, and a pasted
    # token with a stray character should be a 401, not a 500.
    return bool(presented) and secrets.compare_digest(
        presented.encode("utf-8", "replace"), token.encode("utf-8", "replace")
    )


def _check_env_ref(env_ref: Any) -> None:
    if isinstance(env_ref, str) and env_ref.strip().upper() in PROTECTED_ENV:
        raise HTTPException(
            400, f"env_ref {env_ref!r} is one of the brain's own credentials"
        )


def _check_transport(transport: Any) -> None:
    if transport not in MCP_TRANSPORTS:
        raise HTTPException(400, f"transport must be one of {MCP_TRANSPORTS}")


def create_app(
    memory: Memory, config: Config, mcp: Any = None, a2a: Any = None,
    *,
    token: str | None = None,
    resync_sessions: Callable[[], Awaitable[int]] | None = None,
) -> FastAPI:
    app = FastAPI(title="Stack-Chan brain console")

    if not token:
        log.warning("web console running WITHOUT a token — every API route, "
                    "including MCP server registration, is open")

    # Lazy Anthropic client for operator-triggered LLM jobs (summarize now,
    # fact compaction). Created on first use inside the app's event loop and
    # reused; the agent has its own per-session client.
    _llm: dict[str, AsyncAnthropic] = {}

    def llm_client() -> AsyncAnthropic:
        if "c" not in _llm:
            _llm["c"] = AsyncAnthropic()
        return _llm["c"]

    async def resync_live() -> int:
        """Push a durable-history rewrite into the live conversation(s).

        AgentSession hydrates its in-memory thread once, at construction, and
        only appends to it — a session lives for the whole firmware WebSocket
        connection, i.e. days. So editing turns in SQLite alone leaves the
        connected device replaying the very turns the operator just deleted
        until the brain restarts (or a summarizer fold re-hydrates hours
        later, landing the reset at an unpredictable moment). Returns how many
        live sessions were re-synced; 0 means "nothing connected", or that the
        app was built without the hook (tests)."""
        return await resync_sessions() if resync_sessions else 0

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        """Shared-secret gate on the whole API.

        Only the SPA shell and its assets are public — app.js has to load
        before it can present a token. Everything else needs the header (or
        `?token=` for clients that can't set one), because the mutating half
        of this API runs processes as the brain's user and the read half
        streams every transcript."""
        if not _host_allowed(request.headers.get("host", "")):
            return JSONResponse({"detail": "unrecognised Host header"},
                                status_code=403)
        path = request.url.path
        public = path == "/" or path.startswith("/static")
        if token and not public:
            presented = (request.headers.get(TOKEN_HEADER)
                         or request.query_params.get("token"))
            if not _token_ok(presented, token):
                return JSONResponse({"detail": "console token required"},
                                    status_code=401)
        return await call_next(request)

    async def authorize_ws(ws: WebSocket) -> bool:
        """Same gate for the live feeds. Browsers can't set headers on a
        WebSocket handshake, so the token rides in the query string; the log
        feed carries every transcript, so it is not left open."""
        if not _host_allowed(ws.headers.get("host", "")):
            await ws.close(code=1008)
            return False
        if token and not _token_ok(ws.query_params.get("token"), token):
            await ws.close(code=1008)
            return False
        return True

    @app.middleware("http")
    async def revalidate_static(request: Request, call_next):
        """Force the browser to revalidate the SPA assets on every load
        instead of serving a stale cached copy across deploys. `no-cache`
        means "always revalidate", not "never store" — StaticFiles' etag /
        last-modified still yield cheap 304s when nothing changed."""
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # --- config -------------------------------------------------------
    @app.get("/api/config")
    async def get_config_api() -> dict[str, Any]:
        return {"items": config.describe()}

    @app.put("/api/config")
    async def set_config_api(body: dict[str, Any]) -> dict[str, Any]:
        key = body.get("key")
        if not isinstance(key, str):
            raise HTTPException(400, "missing 'key'")
        try:
            value = config.set(key, body.get("value"))
        except KeyError:
            raise HTTPException(404, f"unknown config key: {key}")
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"bad value: {exc}")
        # Read the flag off SPECS, not describe(): describe() omits hidden
        # knobs (SYSTEM_PROMPT, the two prompt templates), so a scan of it
        # raised StopIteration — which CPython turns into a 500 — *after*
        # config.set had already persisted the value, telling the caller the
        # write failed when it had landed.
        return {"key": key, "value": value, "restart": SPECS[key].restart}

    # --- persona / system prompt --------------------------------------
    # The persona is a hot config knob (SYSTEM_PROMPT) but too large for the
    # generic config grid, so it gets its own panel. An empty override means
    # "use the built-in default"; we echo the default so the UI can prefill
    # the editor and offer a reset.
    @app.get("/api/system_prompt")
    async def get_system_prompt() -> dict[str, Any]:
        override = (config.get("SYSTEM_PROMPT") or "").strip()
        return {
            "prompt": override or DEFAULT_SYSTEM_PROMPT,
            "default": DEFAULT_SYSTEM_PROMPT,
            "overridden": bool(override),
        }

    @app.put("/api/system_prompt")
    async def set_system_prompt(body: dict[str, Any]) -> dict[str, Any]:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise HTTPException(400, "missing 'prompt'")
        prompt = prompt.strip()
        # Empty, or verbatim-equal to the default, both collapse to "no
        # override" so the device tracks future default changes.
        if prompt == DEFAULT_SYSTEM_PROMPT.strip():
            prompt = ""
        config.set("SYSTEM_PROMPT", prompt)
        return {
            "prompt": prompt or DEFAULT_SYSTEM_PROMPT,
            "overridden": bool(prompt),
        }

    # --- memories -----------------------------------------------------
    @app.get("/api/memories/facts")
    async def list_facts() -> dict[str, Any]:
        return {
            "facts": [
                {"id": f.id, "ts": f.ts, "fact": f.fact}
                for f in memory.list_fact_rows()
            ]
        }

    @app.put("/api/memories/facts/{fact_id}")
    async def update_fact(fact_id: int, body: dict[str, Any]) -> dict[str, Any]:
        fact = body.get("fact")
        if not isinstance(fact, str) or not fact.strip():
            raise HTTPException(400, "missing 'fact'")
        if not memory.update_fact(fact_id, fact.strip()):
            raise HTTPException(404, "no such fact")
        return {"ok": True}

    @app.post("/api/memories/facts")
    async def add_fact(body: dict[str, Any]) -> dict[str, Any]:
        fact = (body.get("fact") or "").strip() if isinstance(body.get("fact"), str) else ""
        if not fact:
            raise HTTPException(400, "missing 'fact'")
        return {"id": memory.add_fact(fact)}

    @app.delete("/api/memories/facts/{fact_id}")
    async def delete_fact(fact_id: int) -> dict[str, Any]:
        if not memory.delete_fact(fact_id):
            raise HTTPException(404, "no such fact")
        return {"ok": True}

    # LLM fact compaction: propose a consolidated list (no write), then the
    # client POSTs the approved set to /apply.
    @app.post("/api/memories/facts/compact")
    async def compact_facts() -> dict[str, Any]:
        facts = memory.list_facts()
        proposed = await consolidate_facts(
            llm_client(), config.get("MODEL"), facts
        )
        return {"original": facts, "proposed": proposed}

    @app.post("/api/memories/facts/apply")
    async def apply_facts(body: dict[str, Any]) -> dict[str, Any]:
        facts = body.get("facts")
        if not isinstance(facts, list) or not all(isinstance(f, str) for f in facts):
            raise HTTPException(400, "'facts' must be a list of strings")
        cleaned = [f.strip() for f in facts if f.strip()]
        memory.replace_facts(cleaned)
        return {"ok": True, "count": len(cleaned)}

    @app.get("/api/memories/summaries")
    async def list_summaries() -> dict[str, Any]:
        return {
            "summaries": [
                {
                    "id": s.id,
                    "summary": s.summary,
                    "span_from": s.span_from,
                    "span_to": s.span_to,
                }
                for s in memory.list_summaries()
            ]
        }

    @app.put("/api/memories/summaries/{summary_id}")
    async def edit_summary(summary_id: int, body: dict[str, Any]) -> dict[str, Any]:
        text = body.get("summary")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(400, "missing 'summary'")
        if not memory.update_summary(summary_id, text.strip()):
            raise HTTPException(404, "no such summary")
        return {"ok": True}

    @app.delete("/api/memories/summaries/{summary_id}")
    async def delete_summary(summary_id: int, unmark: bool = True) -> dict[str, Any]:
        # `unmark` (default true) un-summarizes the covered turns so they
        # replay verbatim again instead of being silently dropped.
        if not memory.delete_summary(summary_id, unmark_turns=unmark):
            raise HTTPException(404, "no such summary")
        return {"ok": True, "unmarked_turns": unmark}

    @app.post("/api/memories/summarize")
    async def summarize_now() -> dict[str, Any]:
        # Force a fold of the current backlog (down to the keep-recent tail),
        # ignoring the trigger threshold.
        result, reason = await summarize_backlog(
            memory,
            llm_client(),
            config.get("MODEL"),
            keep_recent=int(config.get("KEEP_RECENT_TURNS")),
            force=True,
        )
        if result is None:
            return {"ok": False, "reason": reason}
        # summarize_backlog marks the folded turns summarized; callers holding
        # live turn state have to re-hydrate (its docstring says so, and the
        # automatic path does it).
        live = await resync_live()
        return {
            "ok": True,
            "live_synced": live,
            "summary": {
                "id": result.id, "summary": result.summary,
                "span_from": result.span_from, "span_to": result.span_to,
            },
        }

    @app.get("/api/memories/turns")
    async def list_turns(limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        return {
            "turns": [
                {"id": t.id, "role": t.role, "content": t.content}
                for t in memory.recent_turns(limit)
            ]
        }

    @app.post("/api/memories/repair")
    async def repair_conversation() -> dict[str, Any]:
        """Run the M6.5 integrity pass on demand: heal dangling tool_use /
        orphan tool_result corruption in the unsummarized tail and report the
        counts. Safe to run anytime; a clean DB changes nothing."""
        counts = repair_memory(memory)
        live = await resync_live()
        return {"ok": True, "counts": counts, "live_synced": live}

    @app.post("/api/memories/reset")
    async def reset_conversation() -> dict[str, Any]:
        """Hard-reset the live conversation tail: delete every unsummarized
        turn so the next turn starts from summaries + facts only. Summaries and
        durable facts are kept. The escape hatch when a thread is wedged and a
        repair isn't enough — so it has to reach the *connected* device's
        in-memory thread too, not just SQLite."""
        turns = memory.list_unsummarized_turns()
        deleted = memory.delete_turns_from(turns[0].id) if turns else 0
        live = await resync_live()
        log.warning("conversation reset: deleted %d unsummarized turn(s), "
                    "re-synced %d live session(s)", deleted, live)
        return {"ok": True, "deleted": deleted, "live_synced": live}

    # --- MCP servers (Phase 9b) ---------------------------------------
    @app.get("/api/mcp/servers")
    async def list_mcp() -> dict[str, Any]:
        # Merge persisted registry rows with live connection status.
        status = {s["name"]: s for s in (mcp.status() if mcp else [])}
        servers = []
        for s in memory.list_mcp_servers():
            live = status.get(s.name, {})
            servers.append({
                "id": s.id, "name": s.name, "transport": s.transport,
                "command": s.command, "args": s.args, "url": s.url,
                "env_ref": s.env_ref, "enabled": s.enabled,
                "connected": live.get("connected", False),
                "error": live.get("error"),
                "tools": live.get("tools", []),
            })
        return {"servers": servers, "mcp_available": mcp is not None}

    @app.post("/api/mcp/servers")
    async def add_mcp(body: dict[str, Any]) -> dict[str, Any]:
        name = (body.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "missing 'name'")
        transport = body.get("transport", "stdio")
        _check_transport(transport)
        _check_env_ref(body.get("env_ref"))
        try:
            sid = memory.add_mcp_server(
                name=name,
                transport=transport,
                command=body.get("command"),
                args=body.get("args") or [],
                url=body.get("url"),
                env_ref=body.get("env_ref") or None,
                enabled=bool(body.get("enabled", True)),
            )
        except Exception as exc:
            raise HTTPException(400, f"could not add server: {exc}")
        return {"id": sid}

    @app.put("/api/mcp/servers/{server_id}")
    async def edit_mcp(server_id: int, body: dict[str, Any]) -> dict[str, Any]:
        allowed = {"name", "transport", "command", "args", "url",
                   "env_ref", "enabled"}
        fields = {k: v for k, v in body.items() if k in allowed}
        if not fields:
            raise HTTPException(400, "no editable fields")
        if "transport" in fields:
            _check_transport(fields["transport"])
        if "env_ref" in fields:
            _check_env_ref(fields["env_ref"])
        if not memory.update_mcp_server(server_id, **fields):
            raise HTTPException(404, "no such server")
        return {"ok": True}

    @app.delete("/api/mcp/servers/{server_id}")
    async def remove_mcp(server_id: int) -> dict[str, Any]:
        if not memory.delete_mcp_server(server_id):
            raise HTTPException(404, "no such server")
        return {"ok": True}

    @app.post("/api/mcp/reload")
    async def reload_mcp() -> dict[str, Any]:
        if mcp is None:
            raise HTTPException(503, "MCP client not available")
        await mcp.reload()
        return {"servers": mcp.status()}

    # --- A2A servers (Phase 9c) ---------------------------------------
    @app.get("/api/a2a/servers")
    async def list_a2a() -> dict[str, Any]:
        # Merge persisted registry rows with live connection status.
        status = {s["name"]: s for s in (a2a.status() if a2a else [])}
        servers = []
        for s in memory.list_a2a_servers():
            live = status.get(s.name, {})
            servers.append({
                "id": s.id, "name": s.name, "url": s.url,
                "env_ref": s.env_ref, "enabled": s.enabled,
                "connected": live.get("connected", False),
                "error": live.get("error"),
                "agent": live.get("agent"),
                "delegates": live.get("delegates", []),
            })
        return {"servers": servers, "a2a_available": a2a is not None}

    @app.post("/api/a2a/servers")
    async def add_a2a(body: dict[str, Any]) -> dict[str, Any]:
        name = (body.get("name") or "").strip()
        url = (body.get("url") or "").strip()
        if not name:
            raise HTTPException(400, "missing 'name'")
        if not url:
            raise HTTPException(400, "missing 'url'")
        _check_env_ref(body.get("env_ref"))
        try:
            sid = memory.add_a2a_server(
                name=name,
                url=url,
                env_ref=body.get("env_ref") or None,
                enabled=bool(body.get("enabled", True)),
            )
        except Exception as exc:
            raise HTTPException(400, f"could not add server: {exc}")
        return {"id": sid}

    @app.put("/api/a2a/servers/{server_id}")
    async def edit_a2a(server_id: int, body: dict[str, Any]) -> dict[str, Any]:
        allowed = {"name", "url", "env_ref", "enabled"}
        fields = {k: v for k, v in body.items() if k in allowed}
        if not fields:
            raise HTTPException(400, "no editable fields")
        if "env_ref" in fields:
            _check_env_ref(fields["env_ref"])
        if not memory.update_a2a_server(server_id, **fields):
            raise HTTPException(404, "no such server")
        return {"ok": True}

    @app.delete("/api/a2a/servers/{server_id}")
    async def remove_a2a(server_id: int) -> dict[str, Any]:
        if not memory.delete_a2a_server(server_id):
            raise HTTPException(404, "no such server")
        return {"ok": True}

    @app.post("/api/a2a/reload")
    async def reload_a2a() -> dict[str, Any]:
        if a2a is None:
            raise HTTPException(503, "A2A client not available")
        await a2a.reload()
        return {"servers": a2a.status()}

    # --- live feeds ---------------------------------------------------
    @app.websocket("/ws/logs")
    async def ws_logs(ws: WebSocket) -> None:
        if await authorize_ws(ws):
            await _stream(ws, LOGS)

    @app.websocket("/ws/turns")
    async def ws_turns(ws: WebSocket) -> None:
        if await authorize_ws(ws):
            await _stream(ws, TURNS)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


async def _stream(ws: WebSocket, feed: Broadcaster) -> None:
    """Send scrollback, then live items, until the client disconnects."""
    await ws.accept()
    queue = feed.subscribe()
    try:
        await ws.send_json({"type": "scrollback", "items": feed.scrollback()})
        while True:
            item = await queue.get()
            await ws.send_json({"type": "item", "item": item})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        feed.unsubscribe(queue)
