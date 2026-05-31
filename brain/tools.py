"""Tool definitions for the Claude tool-use loop.

Each tool maps to either a JSON command the firmware understands or a
brain-local action (like a vision-model call). Handlers take a context
object plus the tool input and return a brief acknowledgement string
for the next assistant turn.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from anthropic import AsyncAnthropic
from websockets.asyncio.server import ServerConnection

from memory import Memory

log = logging.getLogger("brain.tools")

# Sonnet for vision Q&A — Haiku 4.5 can do vision too but Sonnet gives
# noticeably better scene descriptions and the tool is invoked rarely.
VISION_MODEL = "claude-sonnet-4-6"


@dataclass
class ToolContext:
    """Per-connection handles a tool handler may need."""
    ws: ServerConnection
    client: AsyncAnthropic
    memory: Memory
    # Callable returning the most recent JPEG frame the firmware has
    # sent us on this connection, or None if we haven't received one
    # yet. Used by describe_view.
    get_latest_jpeg: Callable[[], bytes | None]
    # Hook fired whenever the agent commands an absolute head move
    # (look_at). Lets the gaze controller sync to the new pose so it
    # doesn't immediately yank the head back.
    on_external_head_move: Callable[[float, float], None] = (
        lambda yaw_deg, pitch_deg: None
    )
    # MCP client (Phase 9b), shared across connections. Tools it exposes
    # are namespaced `mcp__<server>__<tool>` and routed here before the
    # native tool ladder below. None if MCP is disabled/unavailable.
    mcp: Any = None


# Schemas exposed to Claude. Keep tight — descriptions are what drive
# tool selection, so be explicit about when to call each.
TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "set_expression",
        "description": (
            "Change the robot's facial expression. Use sparingly to react to the "
            "conversation: 'happy' on good news, 'sad' on bad news, 'surprised' "
            "on unexpected information, 'sleepy' when asked to wind down. Defaults "
            "to 'neutral'."
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
        "name": "describe_view",
        "description": (
            "Look at what the camera currently sees and answer a question "
            "about it. Use when the user asks 'what do you see?', 'what's in "
            "front of you?', 'who's there?', or any visual question. The "
            "optional prompt steers what to report on; if omitted, returns a "
            "short scene description."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Question to answer about the scene, e.g. 'who is "
                        "this?' or 'is the cat in the picture?'. Default: a "
                        "short description."
                    ),
                }
            },
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


async def _describe_view(ctx: ToolContext, prompt: str) -> str:
    jpeg = ctx.get_latest_jpeg()
    if jpeg is None:
        return "No camera frame is available yet."
    img_b64 = base64.standard_b64encode(jpeg).decode("ascii")
    msg = await ctx.client.messages.create(
        model=VISION_MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    text = " ".join(b.text for b in msg.content if b.type == "text").strip()
    log.info(
        "describe_view: %d-byte jpeg → %r (in=%d out=%d)",
        len(jpeg), text[:120],
        msg.usage.input_tokens, msg.usage.output_tokens,
    )
    return text or "I couldn't make out the scene."


async def dispatch(
    name: str, input_: dict[str, Any], ctx: ToolContext
) -> str:
    """Send a tool's JSON command to the firmware (or run the brain-local
    handler). Returns the tool_result content string for the next agent
    turn."""
    # MCP tools (Phase 9b) take priority — they're namespaced so they
    # can't collide with the native tools below.
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
        ctx.on_external_head_move(yaw_deg, pitch_deg)
        return f"Looking at yaw={yaw_deg}, pitch={pitch_deg}."
    if name == "describe_view":
        prompt = input_.get("prompt") or "Describe what you see in one sentence."
        return await _describe_view(ctx, prompt)
    if name == "remember_fact":
        fact = input_["fact"].strip()
        if not fact:
            return "Empty fact — nothing saved."
        ctx.memory.add_fact(fact)
        log.info("remembered: %r", fact)
        return f"Remembered: {fact}"
    if name == "end_conversation":
        # No firmware-side cmd needed; the agent's reply is the goodbye.
        return "Conversation ended."
    log.warning("unknown tool: %s", name)
    return f"Unknown tool {name}"
