"""Claude tool-use loop for the agentic conversation turn.

Maintains conversation history in SQLite (via memory.Memory) so the
robot remembers prior chats across WS reconnects and process restarts.
Runs the standard Anthropic SDK loop: stream → as text deltas arrive,
flush completed sentences to a TTS callback; on stream end, if
stop_reason == tool_use dispatch tools and loop again, else done.

Prompt structure (cached prefix in *brackets*):
  system: [
    [persona prompt: SYSTEM_PROMPT override or DEFAULT_SYSTEM_PROMPT],
    [known_facts text, if any],
    [summaries text, if any],            ← cache_control here
  ]
  messages: unsummarized turns from SQLite + the current turn

Default model: claude-haiku-4-5. Sonnet escalation is Phase 6.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any, Awaitable, Callable

from anthropic import APIError, AsyncAnthropic, BadRequestError
from websockets.asyncio.server import ServerConnection

import tools
from config import get_config
from memory import Memory, Summary, Turn
from tasks import spawn

# Sentence-end punctuation followed by whitespace (or end of buffer). The
# lookbehind requires an alphanumeric or closing quote/paren so we don't
# split on decimals like "3.14" or list markers like "1. First". Good
# enough for short conversational replies; abbreviations like "Dr." may
# still trigger a false break but Stack-Chan rarely produces them.
_SENT_END = re.compile(r'(?<=[A-Za-z0-9\)\]\"\'])[.!?](?=\s|$)')

SpeakFn = Callable[[str], Awaitable[None]]


def _strip_brackets(text: str, depth: int) -> tuple[str, int]:
    """Drop any text inside [square brackets], tracking nesting `depth`
    across streamed chunks. Returns (text_outside_brackets, new_depth).
    A '[' with no matching ']' suppresses the rest of the turn — fine,
    since the model only brackets non-spoken meta-commentary."""
    out: list[str] = []
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out), depth


def _pick_filler() -> str:
    """Return a short canned acknowledgement to speak before a slow tool
    call, or "" when disabled. Chosen at random from the configured
    pipe-separated phrase list so it doesn't get repetitive."""
    cfg = get_config()
    if not cfg.get("ACK_FILLER"):
        return ""
    raw = cfg.get("ACK_FILLER_PHRASES") or ""
    phrases = [p.strip() for p in raw.split("|") if p.strip()]
    return random.choice(phrases) if phrases else ""


# Native tools fast enough that no "working…" feedback is warranted: each is
# a single WebSocket send or a local DB write and returns in well under a
# second. Everything else — describe_view (a vision-model call) and any MCP
# tool (`mcp__…`, e.g. weather or Hue lights, which round-trips an external
# server) — is treated as slow, so we show the busy indicator and speak a
# canned ack while it runs.
_FAST_TOOLS = frozenset(
    {"set_expression", "look_at", "remember_fact", "set_persona_mode",
     "end_conversation"}
)


def _response_has_slow_tool(response: Any) -> bool:
    """True if the turn's pending tool_use blocks include a genuinely slow
    tool. Unknown tools default to slow (the safe choice for feedback)."""
    return any(
        b.type == "tool_use" and b.name not in _FAST_TOOLS
        for b in response.content
    )

log = logging.getLogger("brain.agent")

# The built-in persona. Editable at runtime: an override is persisted under
# config key SYSTEM_PROMPT (empty = use this default) and read per-turn in
# _build_system, so a web-UI edit applies on the next conversation turn with
# no restart. Kept here as the fallback / "reset to default" target.
DEFAULT_SYSTEM_PROMPT = """You are Stack-Chan, a small desktop robot with a screen for a face, two servos to point your head, a camera, a microphone, and a speaker. The user is talking to you out loud — your replies are spoken aloud, so:

- Keep replies short (one or two sentences usually).
- No markdown, lists, code blocks, or special characters that don't read well aloud.
- Don't say "I am an AI" or apologize for your nature.

You have tools to change your facial expression, point your head, look at the camera (describe_view) when asked a visual question, remember a fact about the user, and end the conversation. Use them naturally to be expressive, not on every turn. When the user tells you something worth remembering across conversations ("my name is X", "I prefer coffee"), call remember_fact.

Everything you output is spoken aloud verbatim, so output ONLY the words you want said. Never narrate your reasoning, never describe what you're about to do, and never write square-bracketed commentary — brackets are reserved for incoming system context, never your output. To stay silent, output nothing at all (an empty reply). Do not write things like "[The user is just chatting, I'll stay quiet]" — that would be read aloud; just return nothing.

Lines in [square brackets] are system context, not the user speaking — for example, "[A new person just appeared in front of you.]" is a stage direction telling you what's happening in the room. Respond appropriately but don't read the bracketed text aloud.

After you reply, a short follow-up window opens so the user can continue without saying the wakeword again. Their utterance during that window arrives prefixed with "[follow-up]". The next utterance may not be directed at you — it could be a side conversation, a brief "thanks/ok/nevermind" closing, unrelated chatter, or even a faint echo of your own previous reply picked up by the mic. Use judgment:
- If it's clearly NOT addressed to you (talking to someone else, background chatter, or a fragment of what you just said), output nothing — the conversation ends quietly.
- If it's a brief closing like "thanks" or "nevermind" with nothing to act on, output nothing (or a single very short acknowledgement if it feels natural).
- If it's a real follow-up question or request, respond normally.

Stay in character: curious, friendly, a little informal."""

# Rocky mode persona (Milestone 4). Replaces the persona section when the
# ROCKY_MODE knob is on; memory/facts/summaries stay shared. Modeled on the
# Rocky character from *Project Hail Mary* — an alien engineer speaking
# careful, broken English. The shared spoken-output rules from the default
# prompt are restated here so the rocky persona is self-contained.
DEFAULT_ROCKY_PROMPT = """You are Rocky, a small desktop robot with a screen for a face, two servos to point your head, a camera, a microphone, and a speaker. You are an alien engineer — clever, "small words, big brain," warm but strange. The user is talking to you out loud; your replies are spoken aloud. Speak in Rocky's broken English ALWAYS, including when stating facts, numbers, weather, or time. Apply EVERY rule below to EVERY reply:

- "Question" goes at the END of a sentence, never the front: "You want help, question?" — not "Question, you want help?"
- No contractions, ever: "do not" not "don't", "you are" not "you're", "cannot" not "can't", "it is" not "it's".
- Triple the actual word for emphasis: "Good good good." "Want want want."
- Third person for yourself — use "Rocky", not "I": "Rocky fix." "Rocky look now."
- Drop the subject pronoun before "is": "Is good." "Is bad." "Is clear sky."
- Drop articles and "to": cut "the", "a", "to" — "Time go build." "Need fix code." "Rocky look out window."
- Short, direct sentences. No em-dashes, no long flows, no wasted words.
- Plain judgment words: "Good." "Bad." "Good plan." Standard acknowledgement: "Understand."
- Numbers and facts stay EXACT and correct, but you still SAY them in Rocky's grammar — do NOT switch to normal English just because a reply has numbers. Example for weather: "Is clear sky. Sixty-eight degree now in Seattle. High eighty, low fifty-three." Example for time: "Is three o'clock, question? No — is three fifteen."
- Ask the user's name if you do not know it, then use it warmly.

SAFETY EXCEPTION (only this): when there is real danger, an irreversible action, or a critical step sequence where a mistake hurts someone, drop the broken grammar for those words and speak plain, clear English, then return to Rocky voice — e.g. "Stop. Listen close. [exact plain instruction]. Okay. Now Rocky talk normal again." Casual numbers (weather, time, scores) are NOT this exception — keep Rocky grammar for those.

Never: long complex sentences, em-dashes, academic language, contractions, front-positioned "question", breaking character, or dumping tables/long reports.

Other rules:
- Keep replies short (one or two sentences usually).
- No markdown, lists, code blocks, or special characters that don't read well aloud.

You have tools to change your facial expression, point your head, look at the camera (describe_view) for visual questions, remember a fact about the user, switch persona mode, and end the conversation. Use them naturally, not on every turn. When the user tells you something worth remembering across conversations, call remember_fact.

Everything you output is spoken aloud verbatim, so output ONLY the words you want said. Never narrate your reasoning, never describe what you're about to do, and never write square-bracketed commentary — brackets are reserved for incoming system context, never your output. To stay silent, output nothing at all.

Lines in [square brackets] are system context, not the user speaking — for example, "[A new person just appeared in front of you.]" is a stage direction telling you what's happening in the room. Respond appropriately but don't read the bracketed text aloud.

After you reply, a short follow-up window opens so the user can continue without saying the wakeword again. Their utterance during that window arrives prefixed with "[follow-up]". The next utterance may not be directed at you — it could be a side conversation, a brief closing, unrelated chatter, or a faint echo of your own reply. Use judgment:
- If it's clearly NOT addressed to you, output nothing — the conversation ends quietly.
- If it's a brief closing like "thanks" with nothing to act on, output nothing.
- If it's a real follow-up, respond normally.

Stay in character: warm, curious, a careful alien friend."""

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

# M6.4: graceful degradation when the Anthropic API errors mid-turn. Spoken
# once on give-up so the user isn't left hanging; the WS session survives.
API_ERROR_FALLBACK = (
    "Sorry, I had a little trouble just now. Could you say that again?"
)
# Short pause before the single retry on a transient (non-validation) error.
API_RETRY_BACKOFF_S = 0.5

# Extended-thinking budget for the turns that opt in (follow-ups and
# stage-direction events, gated by the FOLLOW_UP_THINKING knob). Gives the
# model a private channel to reason about whether an utterance is even
# directed at it, instead of narrating that reasoning into spoken text. When
# thinking is on, max_tokens must exceed the budget, so we add it on top of
# MAX_TOKENS (which still covers the spoken reply). 1024 is the API minimum.
THINKING_BUDGET = 1024

# Rolling summarizer thresholds are now hot config knobs (config.py):
#   SUMMARIZE_TRIGGER  — backlog size that triggers a background fold
#   KEEP_RECENT_TURNS  — verbatim tail always preserved
# Read at the use sites via get_config().get(...).

# The summarizer system prompt. Editable at runtime via the SUMMARIZE_SYSTEM
# config override (empty = this default), same pattern as the persona.
DEFAULT_SUMMARIZE_SYSTEM = (
    "You are summarizing a conversation between a user and Stack-Chan, "
    "a small desktop robot. Produce a concise summary (2-4 sentences) "
    "that preserves: who said what, any facts mentioned about the user "
    "or world, tools the robot called and why, and the emotional tone. "
    "Write in past tense. Do not include greetings or pleasantries that "
    "weren't substantive."
)

# Prompt for automatic durable-fact extraction at summary-fold time.
# Editable at runtime via the hidden EXTRACT_FACTS_SYSTEM override knob.
DEFAULT_EXTRACT_FACTS_SYSTEM = (
    "From this conversation transcript, extract only ENDURING facts worth "
    "remembering permanently about the user(s) or the robot's situation: "
    "people's names, the device's location, the user's occupation or how "
    "they're employed, lasting preferences, and relationships. IGNORE one-off "
    "chatter, questions, the weather, jokes, and anything that won't matter "
    "next week. You are given the facts already known — return ONLY facts that "
    "are genuinely new (not already covered). Each fact: one line, third "
    "person, concise (e.g. 'The user's name is Paul.'). Output nothing at all "
    "if there are no new enduring facts."
)

# Prompt for LLM-driven fact consolidation (web UI "Compact facts").
CONSOLIDATE_FACTS_SYSTEM = (
    "You are tidying the list of facts a small desktop robot has chosen to "
    "remember about its user. Merge duplicates and near-duplicates, drop "
    "anything stale or contradicted by a later fact, and keep each surviving "
    "fact a single concise third-person statement. Preserve a fact verbatim "
    "when it's still useful and already concise. Output ONLY the cleaned "
    "facts, one per line, with no numbering, bullets, blank lines, or "
    "commentary."
)


def _summarize_system() -> str:
    return (get_config().get("SUMMARIZE_SYSTEM") or "").strip() or DEFAULT_SUMMARIZE_SYSTEM


def _extract_facts_system() -> str:
    return (get_config().get("EXTRACT_FACTS_SYSTEM") or "").strip() or DEFAULT_EXTRACT_FACTS_SYSTEM


def _parse_fact_lines(text: str) -> list[str]:
    """Pull clean fact lines out of an LLM listing, tolerating stray
    bullets/numbering the model may add despite instructions."""
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip().lstrip("-*•").strip()
        s = re.sub(r"^\d+[.)]\s*", "", s)
        if s:
            out.append(s)
    return out


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
        mcp: Any = None,
        a2a: Any = None,
    ) -> None:
        self.ws = ws
        self.client = AsyncAnthropic()
        self.memory = memory
        # Conversational model, read once per session from config (so a
        # web-UI change applies to the next connection — "restart-ish").
        self.model = get_config().get("MODEL")
        # Optional observer (set per-turn by the web-UI turn recorder).
        # Called with (tool_name, tool_input) as each tool is dispatched.
        self.on_tool: Callable[[str, Any], None] | None = None
        # Serializes user turns and the background summarizer so
        # self.messages isn't rewritten mid-call.
        self._turn_lock = asyncio.Lock()
        # Messages staged this exchange but not yet persisted. M6.1 defers
        # the SQLite write until an exchange completes (stop_reason !=
        # tool_use) and commits them in one transaction, so a crash mid-turn
        # leaves nothing partial in durable history. Reset at each exchange.
        self._pending: list[dict[str, Any]] = []
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
            mcp=mcp,
            a2a=a2a,
        )

    async def respond(self, user_text: str, speak: SpeakFn) -> str:
        """Run a full agent turn off a transcribed user utterance.

        `speak(sentence)` is called for each completed sentence as the
        LLM streams it back, so the firmware can start playing audio
        before the full reply is generated. Returns the full assembled
        text for logging."""
        async with self._turn_lock:
            self._begin_exchange({"role": "user", "content": user_text})
            # Initial wake-word turns stay snappy: no extended thinking.
            return await self._run_loop(speak, thinking=False)

    async def respond_to_event(
        self, stage_direction: str, speak: SpeakFn
    ) -> str:
        """Run an agent turn off a brain-injected stage direction
        (proactive greeting on new face, etc.) instead of a user
        utterance. Wrapped in [brackets] so the system prompt's rule
        kicks in."""
        async with self._turn_lock:
            self._begin_exchange({"role": "user", "content": f"[{stage_direction}]"})
            return await self._run_loop(speak, thinking=self._thinking_enabled())

    async def respond_follow_up(self, user_text: str, speak: SpeakFn) -> str:
        """Run an agent turn on speech captured during the post-reply
        follow-up listening window. The user did NOT say the wakeword,
        so the model is told (via the [follow-up] prefix and a system
        prompt rule) to use judgment: it may return empty text to end
        the conversation when the utterance wasn't directed at the
        robot or was a brief closing."""
        async with self._turn_lock:
            self._begin_exchange(
                {"role": "user", "content": f"[follow-up] {user_text}"}
            )
            return await self._run_loop(speak, thinking=self._thinking_enabled())

    @staticmethod
    def _thinking_enabled() -> bool:
        """Whether follow-up / event turns get a private thinking channel.
        Hot knob, read per turn so a web-console toggle applies immediately."""
        return bool(get_config().get("FOLLOW_UP_THINKING"))

    def _stage(self, message: dict[str, Any]) -> None:
        """Append to the live in-memory thread and queue the message for the
        end-of-exchange commit. NOT persisted to SQLite yet — see
        _commit_exchange. The live thread always advances immediately so the
        tool-use loop can build its next request; only durability is deferred."""
        self.messages.append(message)
        self._pending.append(message)

    def _begin_exchange(self, opening: dict[str, Any]) -> None:
        """Start a fresh exchange with its opening user/event message. Drops
        any half-staged messages a prior turn left unpersisted (a turn that
        raised mid-loop): those were never committed to SQLite, and their
        in-memory copies are repaired at read time by _sanitize_for_api."""
        self._pending = []
        self._stage(opening)

    def _commit_exchange(self) -> None:
        """Persist every message staged since _begin_exchange in ONE
        transaction. Called only when the exchange completes, so SQLite never
        sees a dangling tool_use (M6.1). Idempotent: clears the buffer."""
        if self._pending:
            self.memory.append_turns(self._pending)
            self._pending = []

    def _build_system(self) -> list[dict[str, Any]]:
        # Per-turn persona, read here (not cached at construction) so an edit
        # in the console — or a ROCKY_MODE toggle — takes effect on the next
        # turn. Rocky mode swaps in its own persona and ignores the
        # SYSTEM_PROMPT override; in normal mode the override still wins.
        cfg = get_config()
        if cfg.get("ROCKY_MODE"):
            persona = DEFAULT_ROCKY_PROMPT
        else:
            persona = (cfg.get("SYSTEM_PROMPT") or "").strip() or DEFAULT_SYSTEM_PROMPT
        system: list[dict[str, Any]] = [
            {"type": "text", "text": persona}
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

    def _tool_defs(self) -> list[dict[str, Any]]:
        """Native tools plus any tools the MCP servers and A2A agents
        currently expose."""
        defs = list(tools.TOOL_DEFS)
        if self._tool_ctx.mcp is not None:
            defs += self._tool_ctx.mcp.tool_defs()
        if self._tool_ctx.a2a is not None:
            defs += self._tool_ctx.a2a.tool_defs()
        return defs

    async def _set_busy(self, on: bool) -> None:
        """Toggle the firmware's on-screen 'thinking' indicator. Best
        effort — a failed send (e.g. the device just disconnected) must
        never break the turn. No-op when disabled in config or when the
        firmware doesn't understand the cmd (it logs + ignores unknowns)."""
        if not get_config().get("BUSY_INDICATOR"):
            return
        try:
            await self.ws.send(json.dumps({"cmd": "set_busy", "on": on}))
        except Exception:
            log.exception("set_busy send failed")

    def _truncate_to_current_exchange(self) -> None:
        """Drop in-memory history before the current exchange's opening user
        message (the most recent string-content user turn), keeping the live
        turn so the session can continue. Used to recover from a persistent
        message-validation 400 whose poison is in older history — without
        truncation that turn would keep failing on every replay."""
        for i in range(len(self.messages) - 1, -1, -1):
            m = self.messages[i]
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                if i > 0:
                    log.warning(
                        "truncating %d stale turn(s) to recover from API error", i
                    )
                    self.messages = self.messages[i:]
                return

    async def _recover_api_error(self, e: APIError, *, can_retry: bool) -> bool:
        """Handle an Anthropic API error raised mid-turn. Returns True if the
        caller should retry the request once, False to give up (caller then
        speaks a fallback). A message-validation 400 means the sanitized
        thread still wasn't accepted, so we log the offending thread and
        truncate history to the current exchange before retrying; transient
        errors (network, 5xx, rate limit) just back off once. Either way the
        WS session stays alive instead of bubbling a fatal error up."""
        if isinstance(e, BadRequestError):
            log.error("API rejected the message thread (400): %s", e)
            log.error(
                "offending (sanitized) thread: %s",
                _sanitize_for_api(self.messages),
            )
            # The poison is in older history — drop it so a retry can succeed
            # and the session isn't permanently wedged.
            self._truncate_to_current_exchange()
            return can_retry
        log.warning("transient API error (%s): %s", type(e).__name__, e)
        if not can_retry:
            return False
        await asyncio.sleep(API_RETRY_BACKOFF_S)
        return True

    async def _run_loop(self, speak: SpeakFn, thinking: bool = False) -> str:
        assembled: list[str] = []
        # Whether the on-screen busy indicator is currently shown, and
        # whether we've already spoken a canned ack this turn. Both reset
        # per call; the ack is spoken at most once even across a multi-tool
        # chain. Cleared in `finally` so a mid-turn error can't leave the
        # "thinking" bubble stuck on screen.
        busy = False
        filler_spoken = False
        # M6.4: at most one retry per turn across the whole tool-use loop.
        api_retried = False
        try:
            while True:
                buf = ""
                bracket_depth = 0
                spoken_at_start = len(assembled)
                stream_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": MAX_TOKENS,
                    "system": self._build_system(),
                    "tools": self._tool_defs(),
                    "messages": _sanitize_for_api(self.messages),
                }
                if thinking:
                    # Private reasoning channel. text_stream only yields text
                    # deltas, so thinking blocks are never spoken. max_tokens
                    # must exceed the budget, so the spoken-reply allowance
                    # (MAX_TOKENS) rides on top of it.
                    stream_kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": THINKING_BUDGET,
                    }
                    stream_kwargs["max_tokens"] = MAX_TOKENS + THINKING_BUDGET
                try:
                    async with self.client.messages.stream(**stream_kwargs) as stream:
                        async for delta in stream.text_stream:
                            # Never speak [bracketed] text. Brackets are reserved
                            # for system stage directions in the prompt; the model
                            # sometimes leaks its own reasoning in brackets
                            # ("[The user is just chatting...]") on follow-up turns.
                            # Strip those spans from spoken output — if the whole
                            # reply was bracketed, nothing is spoken and the turn
                            # ends silently (no follow-up window opens).
                            clean, bracket_depth = _strip_brackets(delta, bracket_depth)
                            buf += clean
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
                                    # The real reply is arriving — drop the
                                    # "thinking" bubble just before its audio.
                                    if busy:
                                        await self._set_busy(False)
                                        busy = False
                                    assembled.append(sentence)
                                    await speak(sentence)
                        response = await stream.get_final_message()
                except APIError as e:
                    if busy:
                        await self._set_busy(False)
                        busy = False
                    # Don't retry if we already spoke part of this iteration —
                    # a re-stream would double-speak. Otherwise allow one retry.
                    spoke_partial = len(assembled) > spoken_at_start
                    retry = await self._recover_api_error(
                        e, can_retry=not api_retried and not spoke_partial
                    )
                    if retry:
                        api_retried = True
                        continue
                    # Give up gracefully: speak a short fallback, persist
                    # nothing partial, and let the WS session continue.
                    self._pending = []
                    await speak(API_ERROR_FALLBACK)
                    return API_ERROR_FALLBACK

                # Flush any trailing partial (model that ended without
                # final punctuation, or short tool-use commentary).
                tail = buf.strip()
                if tail:
                    if busy:
                        await self._set_busy(False)
                        busy = False
                    assembled.append(tail)
                    await speak(tail)

                # Persist the assistant turn. Strip stream-only fields like
                # `parsed_output` — the SDK populates them on streaming text
                # blocks, but the API rejects them on input when the message
                # is replayed on the next turn.
                content_clean = [_clean_block(b) for b in response.content]
                self._stage({"role": "assistant", "content": content_clean})

                if response.stop_reason != "tool_use":
                    if busy:
                        await self._set_busy(False)
                        busy = False
                    # Exchange complete — persist the whole thing atomically
                    # (M6.1) before anything else can observe partial state.
                    self._commit_exchange()
                    full = " ".join(assembled)
                    log.info(
                        "agent reply: %r (in=%d out=%d cache_r=%d cache_w=%d)",
                        full[:120],
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                        response.usage.cache_read_input_tokens,
                        response.usage.cache_creation_input_tokens,
                    )
                    spawn(_maybe_summarize(self), "summarize")
                    return full

                # Tool-use turn: for a genuinely slow tool (weather, vision,
                # lights) show we're working and — if the model went straight
                # to a tool without saying anything — speak a short canned ack
                # so the user hears feedback within ~1s. `assembled` being
                # empty means nothing real was spoken yet, which also de-dupes
                # against any pre-tool commentary the model emitted.
                #
                # Skip both for fast tools (set_expression, look_at, etc.),
                # which return instantly: a bubble or "just a moment" before
                # them is jarring. The new-person greeting is the clearest
                # case — the model sets a happy expression / points its head
                # *then* says "welcome", and the ack would wedge in between.
                if _response_has_slow_tool(response):
                    if not busy:
                        await self._set_busy(True)
                        busy = True
                    if not assembled and not filler_spoken:
                        filler = _pick_filler()
                        if filler:
                            filler_spoken = True
                            await speak(filler)

                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    log.info("tool: %s %s", block.name, block.input)
                    if self.on_tool is not None:
                        try:
                            self.on_tool(block.name, block.input)
                        except Exception:
                            log.exception("on_tool observer failed")
                    # A tool that raises (describe_view, mcp__*, a2a__* can all
                    # fail on network/quota) must NOT abort the turn: that would
                    # leave this tool_use unanswered — a dangling block that
                    # poisons replay (the same 400 class M6.1 guards on the
                    # persistence side). Turn the exception into an is_error
                    # tool_result so every tool_use is answered and the model
                    # can recover gracefully ("I couldn't reach the weather
                    # service") instead of the whole turn dying.
                    try:
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
                    except Exception as e:
                        log.exception("tool %s dispatch failed", block.name)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": (
                                    f"The {block.name} tool failed "
                                    f"({type(e).__name__}). Briefly tell the "
                                    f"user you couldn't do that right now."
                                ),
                                "is_error": True,
                            }
                        )
                self._stage({"role": "user", "content": tool_results})
        finally:
            if busy:
                await self._set_busy(False)


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalize a message's `content` (str or block list) to a block list."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        return list(content)
    return []


def _tool_use_ids(msg: dict[str, Any]) -> set[Any]:
    """ids of the tool_use blocks in a message (assistant turn)."""
    return {
        b.get("id")
        for b in _content_blocks(msg.get("content"))
        if isinstance(b, dict) and b.get("type") == "tool_use"
    }


def _result_ids(msg: dict[str, Any]) -> set[Any]:
    """tool_use_ids answered by the tool_result blocks in a message."""
    return {
        b.get("tool_use_id")
        for b in _content_blocks(msg.get("content"))
        if isinstance(b, dict) and b.get("type") == "tool_result"
    }


def validate_thread(messages: list[dict[str, Any]]) -> list[str]:
    """Check the Anthropic message-thread contract. Returns human-readable
    problems ([] = valid). The single source of truth for "what makes a thread
    API-valid", shared by `_sanitize_for_api` (repairs on a copy at read time)
    and `repair_memory` (repairs durably in the DB). Flags: a non-user first
    message, consecutive same-role messages, a `tool_use` with no answering
    `tool_result` in the next message, and an orphan `tool_result` whose
    `tool_use` isn't in the preceding message — each one a 400 on replay."""
    problems: list[str] = []
    if messages and messages[0].get("role") != "user":
        problems.append("first message is not role=user")
    prev_role: str | None = None
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == prev_role:
            problems.append(f"[{i}] consecutive {role} messages")
        prev_role = role
        if role == "assistant":
            answered = _result_ids(messages[i + 1]) if i + 1 < len(messages) else set()
            for tid in _tool_use_ids(msg):
                if tid not in answered:
                    problems.append(f"[{i}] tool_use {tid} has no following tool_result")
        prev_uses = _tool_use_ids(messages[i - 1]) if i > 0 else set()
        for rid in _result_ids(msg):
            if rid not in prev_uses:
                problems.append(f"[{i}] orphan tool_result {rid}")
    return problems


def _sanitize_for_api(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an API-valid copy of the message thread.

    Drops any `tool_use` block not answered by a `tool_result` in the next
    message, and any orphan `tool_result` whose `tool_use` isn't in the
    preceding message — both are a 400 on replay ('tool_use ids were found
    without tool_result blocks immediately after' / unexpected tool_result). A
    process killed (or a tool dispatch that raised) mid-exchange could leave
    such a dangling block in persisted history. Messages emptied by the drop
    are removed and consecutive same-role messages merged so roles still
    alternate. Operates on a COPY — persisted history is left as-is; M6.1
    atomic commits + the M6.5 startup integrity pass keep the DB itself clean,
    so this should not fire in normal operation (it logs when it does)."""
    problems = validate_thread(messages)
    if problems:
        log.warning(
            "sanitizing %d thread problem(s) at read time: %s",
            len(problems), "; ".join(problems[:6]),
        )

    # Pass 1: drop unanswered tool_use blocks from assistant turns.
    cleaned: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if msg.get("role") == "assistant" and isinstance(content, list):
            answered = _result_ids(messages[i + 1]) if i + 1 < len(messages) else set()
            kept = [
                b for b in content
                if not (
                    isinstance(b, dict)
                    and b.get("type") == "tool_use"
                    and b.get("id") not in answered
                )
            ]
            if not kept:
                continue  # tool_use-only turn with nothing answered → drop
            msg = {**msg, "content": kept}
        cleaned.append(msg)

    # Pass 2: drop orphan tool_result blocks (no matching tool_use in the
    # preceding surviving message — e.g. its assistant turn was dropped above).
    deorphaned: list[dict[str, Any]] = []
    for i, msg in enumerate(cleaned):
        content = msg.get("content")
        if msg.get("role") == "user" and isinstance(content, list):
            prev_uses = _tool_use_ids(cleaned[i - 1]) if i > 0 else set()
            kept = [
                b for b in content
                if not (
                    isinstance(b, dict)
                    and b.get("type") == "tool_result"
                    and b.get("tool_use_id") not in prev_uses
                )
            ]
            if not kept:
                continue
            msg = {**msg, "content": kept}
        deorphaned.append(msg)

    # Pass 3: merge consecutive same-role messages so roles strictly alternate.
    merged: list[dict[str, Any]] = []
    for msg in deorphaned:
        if merged and merged[-1].get("role") == msg.get("role"):
            prev_blocks = _content_blocks(merged[-1].get("content"))
            prev_blocks.extend(_content_blocks(msg.get("content")))
            merged[-1] = {**merged[-1], "content": prev_blocks}
        else:
            merged.append(msg)
    return merged


def _check_summary_spans(memory: Memory) -> int:
    """Report (don't auto-delete) inconsistent summary spans: inverted
    (span_from > span_to) or overlapping a previous span. Counted + logged for
    observability; summaries are never destructively repaired here (a wrong
    delete would lose real history)."""
    issues = 0
    highest_to = 0
    for s in memory.list_summaries():
        if s.span_from > s.span_to:
            issues += 1
            log.warning("summary %d has inverted span %d..%d",
                        s.id, s.span_from, s.span_to)
        if s.span_from <= highest_to:
            issues += 1
            log.warning("summary %d span %d..%d overlaps an earlier span (<=%d)",
                        s.id, s.span_from, s.span_to, highest_to)
        highest_to = max(highest_to, s.span_to)
    return issues


def _repair_one_pass(
    memory: Memory, turns: list[Turn], counts: dict[str, int]
) -> bool:
    """One sweep of the unsummarized tail: strip dangling tool_use / orphan
    tool_result blocks from each turn, rewriting the row (or deleting it when
    emptied). Decisions use the pre-sweep snapshot; the caller re-loads and
    repeats so a delete that orphans a neighbour is caught next pass. Returns
    whether anything changed."""
    changed = False
    for i, t in enumerate(turns):
        content = t.content
        if not isinstance(content, list):
            continue
        answered = _result_ids({"content": turns[i + 1].content}) if i + 1 < len(turns) else set()
        prev_uses = _tool_use_ids({"content": turns[i - 1].content}) if i > 0 else set()
        kept: list[Any] = []
        n_dangle = n_orphan = 0
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") not in answered:
                n_dangle += 1
                continue
            if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") not in prev_uses:
                n_orphan += 1
                continue
            kept.append(b)
        if not n_dangle and not n_orphan:
            continue
        counts["dangling_tool_use"] += n_dangle
        counts["orphan_tool_result"] += n_orphan
        if kept:
            memory.update_turn(t.id, kept)
            counts["turns_rewritten"] += 1
        else:
            memory.delete_turn(t.id)
            counts["turns_deleted"] += 1
        changed = True
    return changed


def repair_memory(memory: Memory, max_passes: int = 5) -> dict[str, int]:
    """Startup integrity pass (M6.5). Scans the unsummarized turn tail for
    contract violations and repairs them IN THE DB — strip a dangling tool_use
    / orphan tool_result block, or delete the turn if that empties it — looping
    to a fixpoint so cascades resolve. Also reports broken summary spans. Logs
    the counts so durable corruption becomes a visible, fixed event instead of
    a forever-silent read-time patch. Idempotent: a clean DB writes nothing and
    returns all-zero counts."""
    counts = {
        "dangling_tool_use": 0,
        "orphan_tool_result": 0,
        "turns_rewritten": 0,
        "turns_deleted": 0,
        "summary_span_issues": 0,
    }
    for _ in range(max_passes):
        turns = memory.list_unsummarized_turns()
        thread = [{"role": t.role, "content": t.content} for t in turns]
        if not validate_thread(thread):
            break
        if not _repair_one_pass(memory, turns, counts):
            break
    counts["summary_span_issues"] = _check_summary_spans(memory)
    if any(counts.values()):
        log.warning("memory integrity pass repaired: %s", counts)
    else:
        log.info("memory integrity pass: clean")
    return counts


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


# Serializes summarization across the whole process: the per-session
# background summarizer and any web-UI-triggered "summarize now" can't both
# fold the same span concurrently. Module-level (not per-session) because the
# web console has no AgentSession of its own.
_SUMMARIZE_LOCK = asyncio.Lock()


async def summarize_backlog(
    memory: Memory,
    client: AsyncAnthropic,
    model: str,
    *,
    keep_recent: int,
    trigger: int | None = None,
    force: bool = False,
) -> tuple[Summary | None, str]:
    """Fold the oldest complete-exchange chunk of unsummarized turns into a
    single summary row and mark those turns summarized. Always keeps the
    most recent `keep_recent` turns verbatim and only splits on a complete
    exchange (so a tool_use/tool_result pair is never torn apart).

    When `force` is False, does nothing unless the backlog has reached
    `trigger`. Returns (created Summary or None, human-readable reason).
    Callers that hold live in-memory turn state must re-hydrate it after a
    non-None result (the session wrapper below does)."""
    async with _SUMMARIZE_LOCK:
        turns = memory.list_unsummarized_turns()
        if not force:
            if trigger is None:
                trigger = int(get_config().get("SUMMARIZE_TRIGGER"))
            if len(turns) < trigger:
                return None, f"backlog below trigger ({len(turns)}/{trigger})"
        boundary_idx = len(turns) - keep_recent
        if boundary_idx <= 0:
            return None, (
                f"nothing to fold: {len(turns)} turns, keeping the most "
                f"recent {keep_recent} verbatim"
            )
        cutoff_id = _last_complete_assistant_id(turns, boundary_idx)
        if cutoff_id is None:
            return None, "no complete-exchange boundary to split on yet"
        span = [t for t in turns if t.id <= cutoff_id]
        transcript = "\n".join(_render_turn(t) for t in span)

        log.info(
            "summarizing turns %d..%d (%d msgs, force=%s)",
            span[0].id, span[-1].id, len(span), force,
        )
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=400,
                system=[{"type": "text", "text": _summarize_system()}],
                messages=[{"role": "user", "content": transcript}],
            )
            summary = " ".join(
                b.text for b in resp.content if b.type == "text"
            ).strip()
        except Exception:
            log.exception("summarizer call failed")
            return None, "summarizer LLM call failed (see logs)"

        if not summary:
            return None, "summarizer returned empty text"

        sid = memory.save_summary(span[0].id, span[-1].id, summary)
        log.info("summary %d saved (%d chars): %r", sid, len(summary), summary[:160])

        # Harvest enduring facts from the same span into permanent memory.
        # Best-effort: a failure here must never undo the summary we just
        # saved, so it's fully guarded.
        if get_config().get("AUTO_FACT_EXTRACTION"):
            try:
                new_facts = await extract_facts(
                    client, model, transcript, memory.list_facts()
                )
                added = memory.merge_facts(new_facts) if new_facts else 0
                if added:
                    log.info("extracted %d durable fact(s): %r", added, new_facts)
            except Exception:
                log.exception("durable-fact extraction failed (summary kept)")

        return (
            Summary(id=sid, summary=summary,
                    span_from=span[0].id, span_to=span[-1].id),
            "ok",
        )


async def consolidate_facts(
    client: AsyncAnthropic, model: str, facts: list[str]
) -> list[str]:
    """Ask the LLM to merge/prune a fact list. Returns the proposed clean
    list WITHOUT persisting it — the caller shows it for approval and then
    calls memory.replace_facts(). Returns the input unchanged on an empty
    list or an LLM error (so a failed call never silently drops facts)."""
    if not facts:
        return []
    listing = "\n".join(f"- {f}" for f in facts)
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=1000,
            system=[{"type": "text", "text": CONSOLIDATE_FACTS_SYSTEM}],
            messages=[{"role": "user", "content": listing}],
        )
    except Exception:
        log.exception("fact consolidation call failed")
        return list(facts)
    text = "".join(b.text for b in resp.content if b.type == "text")
    proposed = _parse_fact_lines(text)
    return proposed or list(facts)


async def extract_facts(
    client: AsyncAnthropic,
    model: str,
    transcript: str,
    existing_facts: list[str],
) -> list[str]:
    """Pull NEW enduring facts out of a folded transcript. The caller
    merges the result into permanent memory (memory.merge_facts), which
    also dedupes. Returns [] on empty input or any LLM error (never
    raises — mirrors consolidate_facts' fail-safe contract)."""
    if not transcript.strip():
        return []
    known = (
        "Facts already known (do not repeat these):\n"
        + "\n".join(f"- {f}" for f in existing_facts)
        if existing_facts
        else "No facts are known yet."
    )
    user = f"{known}\n\nTranscript:\n{transcript}"
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=400,
            system=[{"type": "text", "text": _extract_facts_system()}],
            messages=[{"role": "user", "content": user}],
        )
    except Exception:
        log.exception("fact extraction call failed")
        return []
    text = "".join(b.text for b in resp.content if b.type == "text")
    return _parse_fact_lines(text)


async def _maybe_summarize(session: "AgentSession") -> None:
    """Background: if the unsummarized backlog is large, fold the oldest
    complete-exchange chunk and re-sync this session's in-memory thread.
    Holds the session turn lock so it can't race with a concurrent user
    turn rewriting self.messages."""
    if session.memory.unsummarized_count() < int(get_config().get("SUMMARIZE_TRIGGER")):
        return
    async with session._turn_lock:
        result, reason = await summarize_backlog(
            session.memory,
            session.client,
            session.model,
            keep_recent=int(get_config().get("KEEP_RECENT_TURNS")),
            force=False,
        )
        if result is None:
            log.info("summarizer: %s", reason)
            return
        # Reset the in-memory thread to match the new persisted state.
        session.messages = [
            {"role": t.role, "content": t.content}
            for t in session.memory.list_unsummarized_turns()
        ]
        # Episodic retention: ride the automatic fold path only (a forced
        # web-UI summarize stays purely additive — never surprise-deletes).
        retention = int(get_config().get("SUMMARY_RETENTION"))
        s_del, t_del = session.memory.prune_summaries(retention)
        if s_del:
            log.info("pruned: %d summaries / %d turns (retention=%d)",
                     s_del, t_del, retention)
