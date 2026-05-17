"""Claude tool-use loop for the agentic conversation turn.

Maintains per-connection message history and runs the standard Anthropic
SDK loop: call → if tool_use, dispatch tool, append result, call again,
else return the final text. Prompt caching is enabled on the system
prompt and tool definitions (both stable across turns).

Default model: claude-haiku-4-5 (fast + cheap for conversational turns).
Escalation to claude-sonnet-4-6 will come in Phase 6 when we add image
input via describe_view.
"""

from __future__ import annotations

import logging
from typing import Any

from anthropic import AsyncAnthropic
from websockets.asyncio.server import ServerConnection

import tools

log = logging.getLogger("brain.agent")

SYSTEM_PROMPT = """You are Stack-Chan, a small desktop robot with a screen for a face, two servos to point your head, and a microphone and speaker. The user is talking to you out loud — your replies are spoken aloud, so:

- Keep replies short (one or two sentences usually).
- No markdown, lists, code blocks, or special characters that don't read well aloud.
- Don't say "I am an AI" or apologize for your nature.

You have tools to change your facial expression, point your head, adjust how often you fidget, and end the conversation. Use them naturally to be expressive, not on every turn. If the user asks you to "move less" or "be still", call set_motion_rate with a low number.

Stay in character: curious, friendly, a little informal. You don't have eyes outside your screen — you can't see the user yet, you only hear them."""

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024


class AgentSession:
    """One agent state per WebSocket connection. Tracks rolling history."""

    def __init__(self, ws: ServerConnection) -> None:
        self.ws = ws
        self.client = AsyncAnthropic()
        self.messages: list[dict[str, Any]] = []

    async def respond(self, user_text: str) -> str:
        """Run a full agent turn. Returns the assistant's spoken reply.
        Tools fire as side effects (WS commands to the firmware)."""
        self.messages.append({"role": "user", "content": user_text})

        while True:
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=tools.TOOL_DEFS,
                messages=self.messages,
            )

            # Append the assistant turn verbatim (preserves tool_use blocks
            # so the next turn's tool_results can reference them).
            self.messages.append(
                {"role": "assistant", "content": response.content}
            )

            if response.stop_reason != "tool_use":
                # Final reply — extract text and return.
                text = " ".join(
                    b.text for b in response.content if b.type == "text"
                ).strip()
                log.info(
                    "agent reply: %r (in=%d out=%d cache_r=%d cache_w=%d)",
                    text[:120],
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    response.usage.cache_read_input_tokens,
                    response.usage.cache_creation_input_tokens,
                )
                return text

            # Tool-use turn — dispatch each tool_use block, collect results.
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                log.info("tool: %s %s", block.name, block.input)
                result = await tools.dispatch(block.name, block.input, self.ws)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
            self.messages.append({"role": "user", "content": tool_results})
