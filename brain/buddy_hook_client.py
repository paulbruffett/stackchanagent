#!/usr/bin/env python3
"""Claude Code PreToolUse hook → Stack-Chan Buddy (Option C, Milestone 0).

Reads the PreToolUse hook payload on stdin, asks the brain to surface the
pending tool on the robot, blocks until the user taps the head to approve (→
allow), the brain's permission window lapses (→ ask), or the brain is
unreachable (→ ask), then emits a PreToolUse permission decision.

There is no deny: tap to approve. Otherwise the robot reverts to normal after
BUDDY_PERMISSION_TIMEOUT_S (on the brain) and Claude Code falls back to its
normal permission prompt, i.e. you handle it in the session.

Install (no repo dependency — just point a hook at this file). In
~/.claude/settings.json (user-wide) or a project's .claude/settings.json:

    {
      "hooks": {
        "PreToolUse": [
          {
            "matcher": "Bash",
            "hooks": [
              {
                "type": "command",
                "command": "python3 /Users/paul/code/stackchan/brain/buddy_hook_client.py",
                "timeout": 300
              }
            ]
          }
        ]
      }
    }

Use "matcher": "*" to gate every tool, or e.g. "Bash|Edit|Write" for a subset.
The hook `timeout` is how long the prompt stays open on the robot before Claude
Code gives up waiting and shows its own prompt — set it to taste.

Env:
  BUDDY_BRAIN_URL    brain base URL (default http://192.168.4.150:8080)
  BUDDY_HTTP_TIMEOUT seconds to wait on the brain (default 300). The brain has
                     its own BUDDY_PERMISSION_TIMEOUT_S and returns 'ask' when
                     that lapses, so this is just a backstop against an
                     unreachable/hung brain — keep it >= the brain timeout so a
                     late tap still resolves. Set 0 to wait indefinitely (not
                     recommended: a never-closing connection leaves a stale
                     prompt on the robot, which is exactly what this avoids).

On anything but a tap (brain unreachable, timeout, coalesced, mode off) it
emits NO decision, so Claude Code's normal permission flow applies unchanged —
allowlisted/auto-mode commands run, gated ones get the usual terminal prompt.
It never auto-approves, and the bounded timeout means it never blocks the
session or orphans a process.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

FALLBACK = "ask"  # safest: defer to the normal Claude Code prompt


def _hint(tool: str, tool_input: dict) -> str:
    """A short, human-readable summary of what the tool will do."""
    if not isinstance(tool_input, dict):
        return ""
    for key in ("command", "file_path", "path", "url", "pattern", "query"):
        val = tool_input.get(key)
        if isinstance(val, str) and val:
            return val[:200]
    return ""


def _decide(tool: str, hint: str, session_id: str) -> str:
    base = os.environ.get("BUDDY_BRAIN_URL", "http://192.168.4.150:8080").rstrip("/")
    raw = os.environ.get("BUDDY_HTTP_TIMEOUT", "").strip()
    # Bounded by default (300s) so a hung/unreachable brain can never block the
    # session or orphan this process holding a connection open. The brain
    # resolves within BUDDY_PERMISSION_TIMEOUT_S, so a healthy brain always
    # answers first; this only bites if the brain is dead. 0 = wait forever.
    timeout: float | None = float(raw) if raw else 300.0
    if timeout is not None and timeout <= 0:
        timeout = None
    payload = json.dumps(
        {"tool": tool, "hint": hint, "session_id": session_id}
    ).encode("utf-8")
    req = urllib.request.Request(
        base + "/buddy/permission", data=payload,
        headers={"content-type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        decision = json.loads(resp.read()).get("decision", FALLBACK)
    return decision if decision in ("allow", "deny", "ask", "defer") else FALLBACK


def main() -> None:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except Exception:
        data = {}
    tool = data.get("tool_name", "a tool")
    hint = _hint(tool, data.get("tool_input", {}) or {})
    session_id = str(data.get("session_id", "") or "")

    try:
        decision = _decide(tool, hint, session_id)
    except Exception as exc:  # brain down / timeout / bad response
        print(f"buddy hook: falling back to {FALLBACK} ({exc})", file=sys.stderr)
        decision = FALLBACK

    if decision == "allow":
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": "Stack-Chan Buddy: tap-approved",
            }
        }))
    else:
        # No decision: defer to Claude Code's normal permission flow. Emitting
        # "ask" here would FORCE a terminal prompt even for allowlisted /
        # auto-mode commands — an un-tapped robot prompt must leave the
        # session's behavior unchanged, not make it stricter.
        print("{}")
    sys.exit(0)


if __name__ == "__main__":
    main()
