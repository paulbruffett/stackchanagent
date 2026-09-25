"""Tool definitions for the Claude tool-use loop.

Each tool maps to either a JSON command the firmware understands or a
brain-local action (like saving a fact). Handlers take a context
object plus the tool input and return a brief acknowledgement string
for the next assistant turn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from websockets.asyncio.server import ServerConnection

from memory import Memory

log = logging.getLogger("brain.tools")


@dataclass
class ToolContext:
    """Per-connection handles a tool handler may need."""
    ws: ServerConnection
    memory: Memory
    # MCP client (Phase 9b), shared across connections. Tools it exposes
    # are namespaced `mcp__<server>__<tool>` and routed here before the
    # native tool ladder below. None if MCP is disabled/unavailable.
    mcp: Any = None
    # Set by the end_conversation tool, reset by AgentSession at the start of
    # each exchange. Without it the tool is inert: the caller decides whether
    # to hold the mic open purely on "the reply was non-empty", and a goodbye
    # is non-empty, so the follow-up window opened anyway.
    conversation_ended: bool = False


# Schemas exposed to Claude. Keep tight — descriptions are what drive
# tool selection, so be explicit about when to call each.
TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "set_expression",
        "description": (
            "Change the robot's facial expression. Use sparingly to react to the "
            "conversation: 'happy' on good news, 'sad' on bad news, 'surprised' "
            "on unexpected information, 'sleepy' when asked to wind down, "
            "'celebrate' for a genuine win or milestone (a brief flourish). "
            "Defaults to 'neutral'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "enum": [
                        "neutral",
                        "happy",
                        "sad",
                        "sleepy",
                        "angry",
                        "surprised",
                        "celebrate",
                    ],
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "look_at",
        "description": (
            "Point the head at a target. Yaw is left/right in degrees "
            "(-128 to +128, negative is left, +90 is fully right, 0 is "
            "straight ahead). Pitch is up/down in degrees (3 to 87); "
            "~3 is looking down at the desk, ~30 is a neutral resting "
            "pose (slightly up, eyes at user height), ~50 is looking up "
            "at the user, ~85 is chin-to-ceiling. For 'look down' use "
            "5–15, for 'look up' use 60–80, for resting use ~30."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "yaw_deg": {"type": "number", "minimum": -128, "maximum": 128},
                "pitch_deg": {"type": "number", "minimum": 3, "maximum": 87},
            },
            "required": ["yaw_deg", "pitch_deg"],
        },
    },
    {
        "name": "remember_fact",
        "description": (
            "Save a single fact about the user or your shared context that "
            "should persist across conversations (their name, preferences, "
            "ongoing projects, pets, etc.). Use sparingly — only for things "
            "worth recalling later. Don't use for transient conversation "
            "state. Phrase the fact concisely in third person, e.g. 'The "
            "user's name is Paul' or 'The user prefers tea over coffee'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "Concise third-person fact to remember.",
                }
            },
            "required": ["fact"],
        },
    },
    {
        "name": "end_conversation",
        "description": (
            "End the conversation gracefully. Use when the user says goodbye or "
            "indicates they're done talking."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


# Ceiling on a single tool_result. An MCP server, or anything upstream of
# it, can hand back an arbitrarily large blob, and it is not just
# one turn's problem: the result is committed to durable history, replayed on
# every following turn, and rendered again into the summarizer transcript — so
# an oversized one wedges the very fold that would have cleared it. 8k chars is
# far more than a reply this robot speaks aloud can use.
MAX_TOOL_RESULT_CHARS = 8000


async def dispatch(
    name: str, input_: dict[str, Any], ctx: ToolContext
) -> str:
    """Send a tool's JSON command to the firmware (or run the brain-local
    handler). Returns the tool_result content string for the next agent
    turn, clamped to MAX_TOOL_RESULT_CHARS."""
    result = await _dispatch(name, input_, ctx)
    if len(result) > MAX_TOOL_RESULT_CHARS:
        log.warning(
            "tool %s returned %d chars — truncated to %d",
            name, len(result), MAX_TOOL_RESULT_CHARS,
        )
        return result[:MAX_TOOL_RESULT_CHARS] + "\n… (truncated)"
    return result


async def _dispatch(
    name: str, input_: dict[str, Any], ctx: ToolContext
) -> str:
    # MCP tools (Phase 9b) take priority — they're namespaced (`mcp__…`)
    # so they can't collide with the native tools below.
    if ctx.mcp is not None and ctx.mcp.is_mcp_tool(name):
        return await ctx.mcp.dispatch(name, input_)
    if name == "set_expression":
        await ctx.ws.send(
            json.dumps({"cmd": "set_expression", "value": input_["expression"]})
        )
        return f"Expression set to {input_['expression']}."
    if name == "look_at":
        yaw_deg = input_["yaw_deg"]
        pitch_deg = input_["pitch_deg"]
        await ctx.ws.send(
            json.dumps(
                {"cmd": "look_at", "yaw_deg": yaw_deg, "pitch_deg": pitch_deg}
            )
        )
        return f"Looking at yaw={yaw_deg}, pitch={pitch_deg}."
    if name == "remember_fact":
        fact = input_["fact"].strip()
        if not fact:
            return "Empty fact — nothing saved."
        ctx.memory.add_fact(fact)
        log.info("remembered: %r", fact)
        return f"Remembered: {fact}"
    if name == "end_conversation":
        # No firmware-side cmd needed; the agent's reply is the goodbye. The
        # flag is what actually ends the conversation — the caller reads it
        # after the turn and skips the follow-up window.
        ctx.conversation_ended = True
        return "Conversation ended."
    log.warning("unknown tool: %s", name)
    return f"Unknown tool {name}"
