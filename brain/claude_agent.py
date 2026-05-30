"""Claude tool-use loop for the agentic conversation turn.

Maintains conversation history in SQLite (via memory.Memory) so the
robot remembers prior chats across WS reconnects and process restarts.
Runs the standard Anthropic SDK loop: stream → as text deltas arrive,
flush completed sentences to a TTS callback; on stream end, if
stop_reason == tool_use dispatch tools and loop again, else done.

Prompt structure (cached prefix in *brackets*):
  system: [
    [base SYSTEM_PROMPT],
    [known_facts text, if any],
    [summaries text, if any],            ← cache_control here
  ]
  messages: unsummarized turns from SQLite + the current turn

Default model: claude-haiku-4-5. Sonnet escalation is Phase 6.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable

from anthropic import AsyncAnthropic
from websockets.asyncio.server import ServerConnection

import tools
from memory import Memory, Turn

# Sentence-end punctuation followed by whitespace (or end of buffer). The
# lookbehind requires an alphanumeric or closing quote/paren so we don't
# split on decimals like "3.14" or list markers like "1. First". Good
# enough for short conversational replies; abbreviations like "Dr." may
# still trigger a false break but Stack-Chan rarely produces them.
_SENT_END = re.compile(r'(?<=[A-Za-z0-9\)\]\"\'])[.!?](?=\s|$)')

SpeakFn = Callable[[str], Awaitable[None]]

log = logging.getLogger("brain.agent")

SYSTEM_PROMPT = """You are Stack-Chan, a small desktop robot with a screen for a face, two servos to point your head, a camera, a microphone, and a speaker. The user is talking to you out loud — your replies are spoken aloud, so:

- Keep replies short (one or two sentences usually).
- No markdown, lists, code blocks, or special characters that don't read well aloud.
- Don't say "I am an AI" or apologize for your nature.

You have tools to change your facial expression, point your head, look at the camera (describe_view) when asked a visual question, remember a fact about the user, and end the conversation. Use them naturally to be expressive, not on every turn. When the user tells you something worth remembering across conversations ("my name is X", "I prefer coffee"), call remember_fact.

Lines in [square brackets] are system context, not the user speaking — for example, "[A new person just appeared in front of you.]" is a stage direction telling you what's happening in the room. Respond appropriately but don't read the bracketed text aloud.

After you reply, a short follow-up window opens so the user can continue without saying the wakeword again. Their utterance during that window arrives prefixed with "[follow-up]". The next utterance may not be directed at you — it could be a side conversation, a brief "thanks/ok/nevermind" closing, or unrelated chatter overheard by your mic. Use judgment:
- If it's clearly NOT addressed to you (talking to someone else, background chatter), reply with empty text — the conversation ends quietly.
- If it's a brief closing like "thanks" or "nevermind" with nothing to act on, reply with empty text (or a single very short acknowledgement if it feels natural).
- If it's a real follow-up question or request, respond normally.

Stay in character: curious, friendly, a little informal."""

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

# Rolling summarizer thresholds. Once the unsummarized backlog reaches
# SUMMARIZE_TRIGGER, fold the oldest half into a Haiku summary, keeping
# at least KEEP_RECENT_TURNS verbatim. Numbers picked so the verbatim
# tail still contains 4–5 user turns of context.
SUMMARIZE_TRIGGER = 20
KEEP_RECENT_TURNS = 10

SUMMARIZE_SYSTEM = (
    "You are summarizing a conversation between a user and Stack-Chan, "
    "a small desktop robot. Produce a concise summary (2-4 sentences) "
    "that preserves: who said what, any facts mentioned about the user "
    "or world, tools the robot called and why, and the emotional tone. "
    "Write in past tense. Do not include greetings or pleasantries that "
    "weren't substantive."
)


class AgentSession:
    """One agent state per WebSocket connection, backed by shared
    persistent memory. Tracks rolling history in-memory and writes
    every message through to SQLite for replay on the next session."""

    def __init__(
        self,
        ws: ServerConnection,
        memory: Memory,
        get_latest_jpeg: Callable[[], bytes | None] | None = None,
        on_external_head_move: Callable[[float, float], None] | None = None,
    ) -> None:
        self.ws = ws
        self.client = AsyncAnthropic()
        self.memory = memory
        # Serializes user turns and the background summarizer so
        # self.messages isn't rewritten mid-call.
        self._turn_lock = asyncio.Lock()
        # Hydrate in-memory thread from any unsummarized history.
        self.messages: list[dict[str, Any]] = [
            {"role": t.role, "content": t.content}
            for t in memory.list_unsummarized_turns()
        ]
        if self.messages:
            log.info("hydrated %d turns from memory", len(self.messages))
        self._tool_ctx = tools.ToolContext(
            ws=ws,
            client=self.client,
            memory=memory,
            get_latest_jpeg=get_latest_jpeg or (lambda: None),
            on_external_head_move=(
                on_external_head_move or (lambda y, p: None)
            ),
        )

    async def respond(self, user_text: str, speak: SpeakFn) -> str:
        """Run a full agent turn off a transcribed user utterance.

        `speak(sentence)` is called for each completed sentence as the
        LLM streams it back, so the firmware can start playing audio
        before the full reply is generated. Returns the full assembled
        text for logging."""
        async with self._turn_lock:
            self._append({"role": "user", "content": user_text})
            return await self._run_loop(speak)

    async def respond_to_event(
        self, stage_direction: str, speak: SpeakFn
    ) -> str:
        """Run an agent turn off a brain-injected stage direction
        (proactive greeting on new face, etc.) instead of a user
        utterance. Wrapped in [brackets] so the system prompt's rule
        kicks in."""
        async with self._turn_lock:
            self._append({"role": "user", "content": f"[{stage_direction}]"})
            return await self._run_loop(speak)

    async def respond_follow_up(self, user_text: str, speak: SpeakFn) -> str:
        """Run an agent turn on speech captured during the post-reply
        follow-up listening window. The user did NOT say the wakeword,
        so the model is told (via the [follow-up] prefix and a system
        prompt rule) to use judgment: it may return empty text to end
        the conversation when the utterance wasn't directed at the
        robot or was a brief closing."""
        async with self._turn_lock:
            self._append(
                {"role": "user", "content": f"[follow-up] {user_text}"}
            )
            return await self._run_loop(speak)

    def _append(self, message: dict[str, Any]) -> None:
        """Append to the live thread AND persist to SQLite."""
        self.messages.append(message)
        self.memory.append_turn(message["role"], message["content"])

    def _build_system(self) -> list[dict[str, Any]]:
        system: list[dict[str, Any]] = [
            {"type": "text", "text": SYSTEM_PROMPT}
        ]

        facts = self.memory.list_facts()
        if facts:
            facts_text = (
                "Things you've been told to remember about the user or "
                "your shared context (most recent first):\n"
                + "\n".join(f"- {f}" for f in reversed(facts))
            )
            system.append({"type": "text", "text": facts_text})

        summaries = self.memory.list_summaries()
        if summaries:
            summary_text = (
                "Earlier in your conversation history with this user "
                "(oldest first):\n"
                + "\n\n".join(s.summary for s in summaries)
            )
            system.append({"type": "text", "text": summary_text})

        # Cache the entire stable prefix — invalidates only when a new
        # fact or summary is added, then re-caches for the next batch
        # of turns.
        system[-1]["cache_control"] = {"type": "ephemeral"}
        return system

    async def _run_loop(self, speak: SpeakFn) -> str:
        assembled: list[str] = []
        while True:
            buf = ""
            async with self.client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=self._build_system(),
                tools=tools.TOOL_DEFS,
                messages=self.messages,
            ) as stream:
                async for delta in stream.text_stream:
                    buf += delta
                    # Flush every completed sentence as it lands. Pre-
                    # tool commentary ("Let me check…") still gets
                    # spoken on tool-use turns — that's appropriate.
                    while True:
                        m = _SENT_END.search(buf)
                        if not m:
                            break
                        end = m.end()
                        sentence = buf[:end].strip()
                        buf = buf[end:].lstrip()
                        if sentence:
                            assembled.append(sentence)
                            await speak(sentence)
                response = await stream.get_final_message()

            # Flush any trailing partial (model that ended without
            # final punctuation, or short tool-use commentary).
            tail = buf.strip()
            if tail:
                assembled.append(tail)
                await speak(tail)

            # Persist the assistant turn. Strip stream-only fields like
            # `parsed_output` — the SDK populates them on streaming text
            # blocks, but the API rejects them on input when the message
            # is replayed on the next turn.
            content_clean = [_clean_block(b) for b in response.content]
            self._append({"role": "assistant", "content": content_clean})

            if response.stop_reason != "tool_use":
                full = " ".join(assembled)
                log.info(
                    "agent reply: %r (in=%d out=%d cache_r=%d cache_w=%d)",
                    full[:120],
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    response.usage.cache_read_input_tokens,
                    response.usage.cache_creation_input_tokens,
                )
                asyncio.create_task(_maybe_summarize(self))
                return full

            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                log.info("tool: %s %s", block.name, block.input)
                result = await tools.dispatch(
                    block.name, block.input, self._tool_ctx
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
            self._append({"role": "user", "content": tool_results})


_STREAM_ONLY_FIELDS = ("parsed_output",)


def _clean_block(block: Any) -> dict[str, Any]:
    """Convert an SDK content block to a plain dict suitable for replay
    as API input. The streaming SDK decorates text blocks with
    `parsed_output`; the API rejects it on the way back in."""
    d = block.model_dump(exclude_none=True)
    for f in _STREAM_ONLY_FIELDS:
        d.pop(f, None)
    return d


def _last_complete_assistant_id(turns: list[Turn], up_to_idx: int) -> int | None:
    """Walk back from up_to_idx-1 looking for an assistant message whose
    content has no pending tool_use block. Returns that turn's id, or
    None if none found. Splitting a summary in the middle of a tool-use
    loop would leave orphan tool_use/tool_result pairs in the replay."""
    for i in range(up_to_idx - 1, -1, -1):
        t = turns[i]
        if t.role != "assistant":
            continue
        content = t.content
        if isinstance(content, str):
            return t.id
        if isinstance(content, list):
            has_tool_use = any(
                isinstance(b, dict) and b.get("type") == "tool_use"
                for b in content
            )
            if not has_tool_use:
                return t.id
    return None


def _render_turn(turn: Turn) -> str:
    speaker = "User" if turn.role == "user" else "Stack-Chan"
    content = turn.content
    if isinstance(content, str):
        return f"{speaker}: {content}"
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tname = block.get("name", "?")
                tinput = block.get("input", {})
                parts.append(f"[tool {tname}({tinput})]")
            elif btype == "tool_result":
                parts.append(f"[tool result: {block.get('content', '')}]")
        joined = " ".join(p for p in parts if p)
        return f"{speaker}: {joined}"
    return f"{speaker}: <unrenderable>"


async def _maybe_summarize(session: "AgentSession") -> None:
    """Background: if the unsummarized backlog is large, summarize the
    oldest complete-exchange chunk via Haiku and mark those rows
    summarized. Uses the session's turn lock so it can't race with a
    concurrent user turn rewriting self.messages."""
    if session.memory.unsummarized_count() < SUMMARIZE_TRIGGER:
        return
    async with session._turn_lock:
        turns = session.memory.list_unsummarized_turns()
        if len(turns) < SUMMARIZE_TRIGGER:
            return  # raced with another summarize
        cutoff_id = _last_complete_assistant_id(
            turns, len(turns) - KEEP_RECENT_TURNS
        )
        if cutoff_id is None:
            log.warning(
                "summarizer: no complete-exchange boundary in first %d turns",
                len(turns) - KEEP_RECENT_TURNS,
            )
            return
        span = [t for t in turns if t.id <= cutoff_id]
        transcript = "\n".join(_render_turn(t) for t in span)

        log.info(
            "summarizing turns %d..%d (%d msgs)",
            span[0].id, span[-1].id, len(span),
        )
        try:
            resp = await session.client.messages.create(
                model=MODEL,
                max_tokens=400,
                system=[{"type": "text", "text": SUMMARIZE_SYSTEM}],
                messages=[{"role": "user", "content": transcript}],
            )
            summary = " ".join(
                b.text for b in resp.content if b.type == "text"
            ).strip()
        except Exception:
            log.exception("summarizer call failed")
            return

        if not summary:
            log.warning("summarizer returned empty text")
            return

        session.memory.save_summary(span[0].id, span[-1].id, summary)
        log.info("summary saved (%d chars): %r", len(summary), summary[:160])

        # Reset the in-memory thread to match the new persisted state.
        session.messages = [
            {"role": t.role, "content": t.content}
            for t in session.memory.list_unsummarized_turns()
        ]
