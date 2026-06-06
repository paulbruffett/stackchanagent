"""FastAPI app for the brain web console (Phase 9a).

Mounted in-process by agent_server.main() and served on a separate port.
Shares the live `memory` and `Config` singletons, so edits take effect on
the running agent (hot knobs immediately; restart knobs on next start).
No auth — the LAN is trusted.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from anthropic import AsyncAnthropic

from claude_agent import (
    DEFAULT_SYSTEM_PROMPT,
    consolidate_facts,
    summarize_backlog,
)
from config import Config
from memory import Memory
from webui.logbuf import LOGS, TURNS, Broadcaster

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    memory: Memory, config: Config, mcp: Any = None, a2a: Any = None,
    buddy: Any = None,
) -> FastAPI:
    app = FastAPI(title="Stack-Chan brain console")

    # Lazy Anthropic client for operator-triggered LLM jobs (summarize now,
    # fact compaction). Created on first use inside the app's event loop and
    # reused; the agent has its own per-session client.
    _llm: dict[str, AsyncAnthropic] = {}

    def llm_client() -> AsyncAnthropic:
        if "c" not in _llm:
            _llm["c"] = AsyncAnthropic()
        return _llm["c"]

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
        spec = next(i for i in config.describe() if i["key"] == key)
        return {"key": key, "value": value, "restart": spec["restart"]}

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
        return {
            "ok": True,
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
        try:
            sid = memory.add_mcp_server(
                name=name,
                transport=body.get("transport", "stdio"),
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

    # --- Claude Buddy (Option C, Milestone 0) -------------------------
    @app.post("/buddy/permission")
    async def buddy_permission(body: dict[str, Any]) -> dict[str, Any]:
        """Blocking permission gate for a Claude Code PreToolUse hook. Surfaces
        the pending tool on the robot and waits for tap (allow) / wake word
        (deny) / timeout. Returns {"decision": allow|deny|ask}. Never errors —
        a missing buddy or a failure falls back to a safe default so the
        caller's Claude session never hangs."""
        tool = (body.get("tool") or "a tool").strip()
        hint = (body.get("hint") or "").strip()
        if buddy is None:
            return {"decision": config.get("BUDDY_PERMISSION_FALLBACK")}
        decision = await buddy.request_permission(tool, hint)
        return {"decision": decision}

    @app.get("/buddy/status")
    async def buddy_status() -> dict[str, Any]:
        if buddy is None:
            return {"available": False}
        return {"available": True, **buddy.status()}

    # --- live feeds ---------------------------------------------------
    @app.websocket("/ws/logs")
    async def ws_logs(ws: WebSocket) -> None:
        await _stream(ws, LOGS)

    @app.websocket("/ws/turns")
    async def ws_turns(ws: WebSocket) -> None:
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
