#!/usr/bin/env python3
"""Claude Code PreToolUse hook → Stack-Chan Buddy (Option C, Milestone 0).

Reads the PreToolUse hook payload on stdin, asks the brain to surface the
pending tool on the robot, blocks until the user taps (approve) / says the wake
word (deny) / it times out, then emits a PreToolUse permission decision.

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
                "timeout": 120
              }
            ]
          }
        ]
      }
    }

Use "matcher": "*" to gate every tool, or e.g. "Bash|Edit|Write" for a subset.
Set the hook `timeout` larger than the brain's BUDDY_PERMISSION_TIMEOUT_S.

Env:
  BUDDY_BRAIN_URL   brain base URL (default http://127.0.0.1:8080)
  BUDDY_HTTP_TIMEOUT seconds to wait on the brain (default 110)

On any failure (brain unreachable, bad response) it emits "ask" so the normal
Claude Code permission prompt appears — it never auto-approves and never hangs.
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


def _decide(tool: str, hint: str) -> str:
    base = os.environ.get("BUDDY_BRAIN_URL", "http://127.0.0.1:8080").rstrip("/")
    timeout = float(os.environ.get("BUDDY_HTTP_TIMEOUT", "110"))
    payload = json.dumps({"tool": tool, "hint": hint}).encode("utf-8")
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

    try:
        decision = _decide(tool, hint)
    except Exception as exc:  # brain down / timeout / bad response
        print(f"buddy hook: falling back to {FALLBACK} ({exc})", file=sys.stderr)
        decision = FALLBACK

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": f"Stack-Chan Buddy: {decision}",
        }
    }))
    sys.exit(0)


if __name__ == "__main__":
    main()
