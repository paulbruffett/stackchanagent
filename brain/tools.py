"""Tool definitions for the agent's tool-call loop.

Each tool maps to either a JSON command the firmware understands or a
brain-local action (like saving a fact). Handlers take a context
object plus the tool input and return a brief acknowledgement string
for the next assistant turn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from websockets.asyncio.server import ServerConnection

from config import get_config
import timers
from memory import Memory
from policy import DeviceStatus, describe_device_status, step_volume

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
    # The firmware's last battery/charging/volume report (the connection's
    # ConnState.device). None when the session was built without one (tests).
    device: DeviceStatus | None = None


# Schemas exposed to the model. Keep tight — descriptions are what drive
# tool selection, so be explicit about when to call each. Written in the
# neutral {name, description, input_schema} shape MCP tools also arrive in;
# claude_agent._openai_tool converts both to the Chat Completions shape.
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
        "name": "set_volume",
        "description": (
            "Change YOUR OWN speaker volume (the robot's voice) — for requests "
            "like 'turn your volume down', 'speak louder', 'you're too loud' or "
            "'set your volume to 40'. Not for TVs, speakers or media players in "
            "the house: those are Home Assistant devices. Give either level "
            "(0-100) or change 'up'/'down' (a step of 15). Say a short "
            "confirmation in the same message."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "minimum": 0, "maximum": 100},
                "change": {"type": "string", "enum": ["up", "down"]},
            },
        },
    },
    {
        "name": "get_device_status",
        "description": (
            "Read your own battery level, whether you're charging, and your "
            "speaker volume. Use when asked about your battery, charge or "
            "volume."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "dance",
        "description": (
            "Do a short dance with your head and face (a few seconds). Use "
            "when asked to dance, or for a real celebration. Styles: 'happy' "
            "(swaying), 'robot' (stiff and angular), 'panic' (a nervous "
            "shake). Not on every turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "style": {"type": "string", "enum": ["happy", "robot", "panic"]},
            },
            "required": ["style"],
        },
    },
    {
        "name": "set_timer",
        "description": (
            "Start a timer or set a reminder; the robot chimes and speaks when "
            "it goes off. Give EXACTLY ONE of duration_seconds (\"set a timer "
            "for 10 minutes\" → 600) or at (a local clock time as 24-hour "
            "HH:MM: \"remind me at 5pm to call mom\" → at=\"17:00\", "
            "label=\"call mom\"; a time already past today means tomorrow). "
            "label is optional: what the timer is for (\"pasta\") or what to "
            "remind about, phrased to follow \"Reminder:\". Say a short "
            "confirmation in the same message as the call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "duration_seconds": {
                    "type": "integer", "minimum": timers.MIN_DURATION_S,
                    "maximum": timers.MAX_DURATION_S,
                    "description": "Length of the timer in seconds (1 s to 24 h).",
                },
                "at": {
                    "type": "string",
                    "description": "Local clock time, 24-hour HH:MM (e.g. 17:00).",
                },
                "label": {
                    "type": "string",
                    "description": "Optional: what it is for, e.g. 'pasta' or 'call mom'.",
                },
                "kind": {
                    "type": "string", "enum": list(timers.KINDS),
                    "description": (
                        "'reminder' when the user said \"remind me\", else "
                        "'timer'. Defaults to 'reminder' with at, 'timer' "
                        "with duration_seconds."
                    ),
                },
            },
        },
    },
    {
        "name": "list_timers",
        "description": (
            "List the active timers and reminders with the time left on each "
            "(\"how long is left on the pasta?\", \"what timers do I have?\")."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "cancel_timer",
        "description": (
            "Cancel a timer or reminder by its label (\"pasta\"), by the id "
            "list_timers shows, or \"all\". With a single timer set, "
            "\"timer\" cancels it. Say a short confirmation in the same "
            "message as the call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label_or_id": {
                    "type": "string",
                    "description": "The timer's label, its id, or \"all\".",
                },
            },
            "required": ["label_or_id"],
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

# Marks a tool result that reports a failure rather than an outcome. Chat
# Completions tool messages have no is_error flag, so the agent loop reads
# this prefix (is_error_result) to decide a device command did not land and
# the model must get a second round to say so.
TOOL_ERROR_PREFIX = "[tool error]"


def is_error_result(result: str) -> bool:
    return result.startswith(TOOL_ERROR_PREFIX)


# remember_fact's result when it was handed an empty fact: not an error the
# tool raises, but nothing happened, so the model must get a round to react.
NOTHING_SAVED = "Empty fact — nothing saved."

# The native tools that are a single effect with nothing to read back: a
# WebSocket command to the firmware or a local DB write, done in well under a
# second. The agent loop derives both "no busy bubble / ack" and "may ride
# along with a single-round device command" from this one set.
NATIVE_EFFECT_TOOLS = frozenset(
    {"set_expression", "look_at", "remember_fact", "dance", "set_volume"}
)

# Of those, the ones that are a device action the user asked for — like a Home
# Assistant action they may END a turn alongside the spoken confirmation. The
# rest are expressive and only ride along.
NATIVE_ACTION_TOOLS = frozenset({"set_volume"})

# Timer changes: a local DB write, like remember_fact, but the point of the
# request — so, like a Home Assistant device action, a round that makes one
# with a spoken confirmation may end the turn (claude_agent._ends_turn).
# list_timers is a query: fast, but its answer needs a second round.
TIMER_ACTION_TOOLS = frozenset({"set_timer", "cancel_timer"})
TIMER_TOOLS = TIMER_ACTION_TOOLS | {"list_timers"}


DANCE_STYLES = ("happy", "robot", "panic")


def clean_mcp_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Drop argument values that only mean "not given". Models fill every
    optional property of a schema with a blank — Home Assistant's intent tools
    have no required fields at all — and HA rejects the whole call over one
    of them ("Received invalid slot info" for floor="" or device_class=[];
    "Failed to call turn_on" for a colour temperature of 0 K). Blank strings,
    empty lists/dicts and None are never meaningful slot values; a colour
    temperature of 0 or below never is either. A brightness of 0 is real and
    stays."""
    out = {k: v for k, v in args.items() if v not in (None, "", [], {})}
    if name.rsplit("__", 1)[-1].startswith("Hass"):
        temp = out.get("temperature")
        if isinstance(temp, (int, float)) and temp <= 0:
            del out["temperature"]
    return out


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
        return await ctx.mcp.dispatch(name, clean_mcp_args(name, input_))
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
            return NOTHING_SAVED
        ctx.memory.add_fact(fact)
        log.info("remembered: %r", fact)
        return f"Remembered: {fact}"
    if name == "set_volume":
        return await _set_volume(input_, ctx)
    if name == "get_device_status":
        return describe_device_status(ctx.device or DeviceStatus())
    if name == "dance":
        style = input_.get("style")
        if style not in DANCE_STYLES:
            return f"{TOOL_ERROR_PREFIX} Unknown dance style {style!r}"
        await ctx.ws.send(json.dumps({"cmd": "dance", "style": style}))
        return f"Dancing ({style})."
    if name in TIMER_TOOLS:
        return _timer_tool(name, input_, ctx.memory)
    if name == "end_conversation":
        # No firmware-side cmd needed; the agent's reply is the goodbye. The
        # flag is what actually ends the conversation — the caller reads it
        # after the turn and skips the follow-up window.
        ctx.conversation_ended = True
        return "Conversation ended."
    log.warning("unknown tool: %s", name)
    return f"{TOOL_ERROR_PREFIX} Unknown tool {name}"


async def _set_volume(input_: dict[str, Any], ctx: ToolContext) -> str:
    """Write SPEAKER_VOLUME (so the console and the voice agree) and send
    set_volume at once — the user asked, so mid-turn is fine. The sent value
    becomes the device's volume straight away, so the idle ticker's sync has
    nothing to repeat and a later call in the same turn starts from it."""
    dev = ctx.device
    if dev is None or dev.volume is None:
        # Firmware without set_volume never reports a volume.
        return (f"{TOOL_ERROR_PREFIX} The robot's firmware doesn't support "
                "volume control yet.")
    current = dev.volume
    level = input_.get("level")
    if isinstance(level, bool) or not isinstance(level, (int, float)):
        level = None
    change = input_.get("change")
    try:
        new = step_volume(current, level, change if change in ("up", "down") else None)
    except ValueError as e:
        return f"{TOOL_ERROR_PREFIX} {e}"
    if new == current and level is None:
        # "up" at 100 / "down" at 0: nothing changes, and the confirmation the
        # model already spoke is wrong — a failure gets it a round to say so.
        edge = "maximum" if new == 100 else "minimum"
        return f"{TOOL_ERROR_PREFIX} Volume is already at the {edge} ({new})."
    get_config().set("SPEAKER_VOLUME", new)
    await ctx.ws.send(json.dumps({"cmd": "set_volume", "value": new}))
    dev.volume = new
    if new == current:
        return f"Volume is already at {new}."
    return f"Volume set to {new} (was {current})."


def _timer_tool(name: str, input_: dict[str, Any], memory: Memory) -> str:
    """set_timer / list_timers / cancel_timer against memory.db. A request
    that can't be carried out comes back as a TOOL_ERROR_PREFIX result, so
    the model gets a second round to say so."""
    now = datetime.now().astimezone()
    active = memory.list_timers()
    try:
        if name == "set_timer":
            new = timers.new_timer(
                input_.get("duration_seconds"), input_.get("at"),
                input_.get("label"), now, len(active), input_.get("kind"),
            )
            t = memory.add_timer(new.label, new.fire_ts, new.duration_s, new.kind)
            log.info("timer %d set: %s", t.id, timers.describe(t, now))
            return timers.confirmation(new, now)
        if name == "list_timers":
            return timers.format_list(active, now, with_ids=True)
        hits = timers.match_cancel(active, input_.get("label_or_id"))
        cancelled = [t for t in hits if memory.delete_timer(t.id)]
        log.info("timers cancelled: %s", [t.id for t in cancelled])
        if not cancelled:
            # Nothing happened: like an empty remember_fact, the model must
            # get a round to correct the "Cancelled!" it may already have said.
            raise timers.TimerError("No timers are set; nothing was cancelled.")
        return f"Cancelled {len(cancelled)}: " + "; ".join(
            timers.describe(t, now) for t in cancelled) + "."
    except timers.TimerError as e:
        return f"{TOOL_ERROR_PREFIX} {e}"
