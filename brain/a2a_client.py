"""A2A client — a third tool source for the Claude tool-use loop (Phase 9c).

Connects to the Agent2Agent (A2A) servers in the registry
(`memory.a2a_servers`), fetches each one's agent card
(`<url>/.well-known/agent.json`), and exposes its sub-agents to the Claude
loop as delegation tools namespaced `a2a__<server>__<agent_id>`. The agent
loop merges `tool_defs()` into its `tools=` list and routes `tool_use`
blocks whose name `is_a2a_tool()` to `dispatch()`.

Why one tool per *sub-agent* (not per skill): A2A is message-based — you
send a sub-agent a natural-language message and it self-routes among its
own skills. So each sub-agent becomes a single delegation tool taking a
free-text `request`, and its advertised skills go into the tool
description so Claude knows what to ask for. A card with no `agents[]`
(plain A2A) collapses to one tool for the root agent.

Unlike the MCP client there is no long-lived session or anyio task-owner
constraint: A2A is stateless HTTP request/response (JSON-RPC 2.0), so a
shared httpx.AsyncClient is all we need. "Connecting" just means fetching
the card; `reload()` re-fetches.

Secrets: `env_ref` optionally names a single environment variable (loaded
from `.env`) holding a bearer token, sent as `Authorization: Bearer ...`.
The value is never stored in the registry — only the variable name.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from typing import Any
from urllib.parse import urljoin

import httpx

from memory import A2aServer, Memory

log = logging.getLogger("brain.a2a")

CONNECT_TIMEOUT_S = 15.0
CALL_TIMEOUT_S = 60.0
# When a server returns a still-running task (non-terminal state) we poll
# tasks/get until it finishes or this budget is exhausted.
POLL_TIMEOUT_S = 55.0
POLL_INTERVAL_S = 1.5

_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
_TERMINAL = {"completed", "failed", "canceled", "cancelled", "rejected", "input-required"}
_WELL_KNOWN = "/.well-known/agent.json"


def _sanitize(part: str) -> str:
    """Make a name fragment safe for an Anthropic tool name."""
    return _NAME_RE.sub("_", part)[:48]


def _card_url(url: str) -> str:
    """Normalize a registry URL to the agent-card URL. Accepts either the
    agent base (http://host:port) or the full /.well-known/agent.json URL."""
    u = url.strip()
    if u.endswith(_WELL_KNOWN) or "/.well-known/" in u:
        return u
    return u.rstrip("/") + _WELL_KNOWN


def _parts_text(parts: Any) -> str:
    """Join the text from an A2A `parts` list, tolerating both the 0.2.x
    `kind` and older `type` discriminators."""
    out: list[str] = []
    for p in parts or []:
        if not isinstance(p, dict):
            continue
        if p.get("kind") == "text" or p.get("type") == "text":
            t = p.get("text")
            if t:
                out.append(t)
    return "\n".join(out).strip()


def _result_text(result: Any) -> str:
    """Pull spoken text out of a message/send (or tasks/get) result, which
    may be a Message or a Task."""
    if not isinstance(result, dict):
        return ""
    kind = result.get("kind")
    # A bare Message result.
    if kind == "message" or ("parts" in result and "status" not in result):
        return _parts_text(result.get("parts"))
    # A Task: prefer artifacts, then the status message, then last history.
    artifacts = result.get("artifacts") or []
    texts = [_parts_text(a.get("parts")) for a in artifacts if isinstance(a, dict)]
    texts = [t for t in texts if t]
    if texts:
        return "\n".join(texts)
    status = result.get("status") or {}
    msg = status.get("message") or {}
    t = _parts_text(msg.get("parts"))
    if t:
        return t
    history = result.get("history") or []
    for h in reversed(history):
        if isinstance(h, dict):
            t = _parts_text(h.get("parts"))
            if t:
                return t
    return ""


class _Delegate:
    """One sub-agent surfaced as a single delegation tool."""

    def __init__(
        self, tool_name: str, agent_id: str, agent_name: str, url: str,
        description: str,
    ) -> None:
        self.tool_name = tool_name
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.url = url
        self.description = description


class _A2aConn:
    """One A2A server: its fetched card + the delegation tools it yields."""

    def __init__(self, spec: A2aServer) -> None:
        self.spec = spec
        self.connected = False
        self.error: str | None = None
        self.card: dict[str, Any] = {}
        self.delegates: list[_Delegate] = []
        # Send method that this server actually accepts. Some servers (e.g.
        # Hermes) advertise protocol 0.2.0 but only implement the pre-0.2
        # "tasks/send"; discovered on first call and remembered.
        self.method: str | None = None

    def _headers(self) -> dict[str, str]:
        ref = self.spec.env_ref
        if ref and os.environ.get(ref):
            return {"Authorization": f"Bearer {os.environ[ref]}"}
        if ref:
            log.warning("a2a %s: env_ref %r not set", self.spec.name, ref)
        return {}

    async def connect(self, http: httpx.AsyncClient) -> None:
        """Fetch the agent card and build delegation tools."""
        try:
            card_url = _card_url(self.spec.url)
            r = await http.get(
                card_url, headers=self._headers(), timeout=CONNECT_TIMEOUT_S
            )
            r.raise_for_status()
            self.card = r.json()
            self.delegates = self._build_delegates(card_url)
            self.connected = True
            log.info(
                "a2a %s connected: %s, %d delegate(s) (%s)",
                self.spec.name, self.card.get("name", "?"), len(self.delegates),
                ", ".join(d.agent_id for d in self.delegates),
            )
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("a2a %s connect failed: %s", self.spec.name, self.error)

    def _build_delegates(self, card_url: str) -> list[_Delegate]:
        card = self.card
        skills = card.get("skills") or []
        # Group skills by their owning sub-agent for the descriptions.
        by_agent: dict[str | None, list[dict]] = {}
        for s in skills:
            by_agent.setdefault(s.get("agent"), []).append(s)

        agents = card.get("agents") or []
        sname = _sanitize(self.spec.name)
        out: list[_Delegate] = []
        if agents:
            for a in agents:
                aid = a.get("id") or a.get("name") or "agent"
                aurl = urljoin(card_url, a.get("url") or card.get("url") or "")
                desc = self._describe(a.get("name", aid), a.get("description", ""),
                                      by_agent.get(aid, []))
                out.append(_Delegate(
                    f"a2a__{sname}__{_sanitize(str(aid))}",
                    str(aid), a.get("name", str(aid)), aurl, desc,
                ))
        else:
            # Plain A2A card: one tool for the whole agent.
            aurl = urljoin(card_url, card.get("url") or "")
            desc = self._describe(card.get("name", self.spec.name),
                                  card.get("description", ""), skills)
            out.append(_Delegate(
                f"a2a__{sname}", "root", card.get("name", self.spec.name),
                aurl, desc,
            ))
        return out

    @staticmethod
    def _describe(name: str, description: str, skills: list[dict]) -> str:
        lines = [f"Delegate a request to the '{name}' agent."]
        if description:
            lines.append(description)
        if skills:
            lines.append("It can: " + "; ".join(
                f"{s.get('name', s.get('id', '?'))}"
                + (f" ({s['description']})" if s.get("description") else "")
                for s in skills
            ) + ".")
        lines.append(
            "Pass a clear natural-language `request`; the agent routes it to "
            "the right skill and returns a text answer."
        )
        return " ".join(lines)[:1024]

    async def call(self, http: httpx.AsyncClient, url: str, request: str) -> str:
        method = self.method or "message/send"
        env = await self._rpc(http, url, method, request)
        err = env.get("error")
        if (
            err and err.get("code") == -32601 and method == "message/send"
        ):
            log.info(
                "a2a %s: message/send unsupported, retrying with tasks/send",
                self.spec.name,
            )
            method = "tasks/send"
            env = await self._rpc(http, url, method, request)
            err = env.get("error")
        if err:
            log.warning("a2a %s %s error: %s", self.spec.name, method, err)
            return f"[a2a error] {err.get('message', err)}"
        self.method = method
        result = env.get("result")
        result = await self._await_task(http, url, result)
        text = _result_text(result)
        if text:
            log.info("a2a %s %s ok: %d chars", self.spec.name, method, len(text))
        else:
            log.warning(
                "a2a %s: no text in %s result (keys=%s)", self.spec.name, method,
                sorted(result.keys()) if isinstance(result, dict)
                else type(result).__name__,
            )
        return text or "(the agent returned no text)"

    async def _rpc(
        self, http: httpx.AsyncClient, url: str, method: str, request: str
    ) -> dict[str, Any]:
        message = {
            "role": "user",
            # Both part discriminators, for 0.2.x ("kind") and legacy ("type")
            # servers alike.
            "parts": [{"kind": "text", "type": "text", "text": request}],
            "messageId": uuid.uuid4().hex,
        }
        params: dict[str, Any] = {"message": message}
        if method == "tasks/send":
            params["id"] = uuid.uuid4().hex  # legacy servers require a task id
        payload = {
            "jsonrpc": "2.0", "id": uuid.uuid4().hex,
            "method": method, "params": params,
        }
        r = await http.post(
            url, json=payload, headers=self._headers(), timeout=CALL_TIMEOUT_S
        )
        r.raise_for_status()
        return r.json()

    async def _await_task(
        self, http: httpx.AsyncClient, url: str, result: Any
    ) -> Any:
        """If the result is a still-running Task, poll tasks/get until it
        reaches a terminal state or the poll budget runs out."""
        if not isinstance(result, dict):
            return result
        # 0.2.x tasks carry kind=="task"; legacy (tasks/send) results have no
        # kind but are tasks iff they carry an id + status.state.
        is_task = result.get("kind") == "task" or (
            result.get("id") and isinstance(result.get("status"), dict)
        )
        if not is_task:
            return result
        task_id = result.get("id")
        waited = 0.0
        while task_id and waited < POLL_TIMEOUT_S:
            state = (result.get("status") or {}).get("state")
            if state in _TERMINAL or state is None:
                return result
            await asyncio.sleep(POLL_INTERVAL_S)
            waited += POLL_INTERVAL_S
            poll = {
                "jsonrpc": "2.0", "id": uuid.uuid4().hex,
                "method": "tasks/get", "params": {"id": task_id},
            }
            try:
                pr = await http.post(
                    url, json=poll, headers=self._headers(), timeout=CALL_TIMEOUT_S
                )
                pr.raise_for_status()
                result = pr.json().get("result", result)
            except Exception as exc:  # noqa: BLE001
                log.warning("a2a %s tasks/get failed: %s", self.spec.name, exc)
                return result
        return result


class A2aClient:
    """Process-wide singleton (lifecycle like Memory / McpClient): built in
    agent_server.main(), started before serving, reloaded from the web UI."""

    def __init__(self, memory: Memory) -> None:
        self._memory = memory
        self._conns: list[_A2aConn] = []
        self._http: httpx.AsyncClient | None = None
        # tool name -> (conn, delegate)
        self._index: dict[str, tuple[_A2aConn, _Delegate]] = {}

    async def start(self) -> None:
        self._http = httpx.AsyncClient()
        specs = [s for s in self._memory.list_a2a_servers() if s.enabled]
        self._conns = [_A2aConn(s) for s in specs]
        for c in self._conns:
            await c.connect(self._http)
        self._rebuild_index()
        log.info(
            "a2a client started: %d/%d servers connected, %d delegate tools",
            sum(c.connected for c in self._conns), len(self._conns),
            len(self._index),
        )

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
        self._http = None
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
            for d in c.delegates:
                self._index[d.tool_name] = (c, d)

    # -- agent-facing --
    def tool_defs(self) -> list[dict[str, Any]]:
        defs = []
        for name, (conn, d) in self._index.items():
            defs.append({
                "name": name,
                "description": d.description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "string",
                            "description": "The natural-language request to "
                            "send the agent.",
                        }
                    },
                    "required": ["request"],
                },
            })
        return defs

    def is_a2a_tool(self, name: str) -> bool:
        return name in self._index

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        entry = self._index.get(name)
        if entry is None or self._http is None:
            return f"Unknown A2A tool {name}"
        conn, d = entry
        request = (arguments or {}).get("request", "").strip()
        if not request:
            return "No request provided to delegate."
        try:
            return await conn.call(self._http, d.url, request)
        except Exception as exc:  # noqa: BLE001
            log.warning("a2a call %s failed: %s", name, exc)
            return f"The {d.agent_name} agent failed: {exc}"

    # -- web UI --
    def status(self) -> list[dict[str, Any]]:
        out = []
        for c in self._conns:
            out.append({
                "name": c.spec.name,
                "enabled": c.spec.enabled,
                "connected": c.connected,
                "error": c.error,
                "agent": c.card.get("name"),
                "delegates": [d.agent_name for d in c.delegates],
            })
        return out
