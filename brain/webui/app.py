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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import Config
from memory import Memory
from webui.logbuf import LOGS, TURNS, Broadcaster

STATIC_DIR = Path(__file__).parent / "static"


def create_app(memory: Memory, config: Config) -> FastAPI:
    app = FastAPI(title="Stack-Chan brain console")

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

    @app.delete("/api/memories/facts/{fact_id}")
    async def delete_fact(fact_id: int) -> dict[str, Any]:
        if not memory.delete_fact(fact_id):
            raise HTTPException(404, "no such fact")
        return {"ok": True}

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

    @app.get("/api/memories/turns")
    async def list_turns(limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        return {
            "turns": [
                {"id": t.id, "role": t.role, "content": t.content}
                for t in memory.recent_turns(limit)
            ]
        }

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
