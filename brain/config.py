"""Runtime configuration for the brain (Phase 9a).

A single source of truth for the operator-tunable knobs that used to be
scattered as module-level constants. Defaults live here; overrides are
persisted in the `config` table of memory.db and edited from the web UI.

Two classes of knob:
  - hot   : read at the use site every time, so a write takes effect on
            the next turn/tick with no restart.
  - restart : bound when a long-lived object is constructed (the TTS
            voice, the STT model, the Anthropic model name). Editing it
            is persisted but only applies after a process restart; the
            UI surfaces this.

Usage:
    from config import get_config
    cfg = get_config()
    interval = cfg.get("LOOK_AROUND_INTERVAL_S")

`init_config(memory)` must be called once at startup (agent_server.main)
before any `get_config()` call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("brain.config")


@dataclass(frozen=True)
class Spec:
    """Metadata for one tunable knob."""
    default: Any
    type: str          # "float" | "int" | "str" — for UI rendering + coercion
    restart: bool      # True = applies only after a process restart
    group: str         # UI grouping
    help: str
    # Excluded from describe() so it doesn't render in the generic config
    # grid — used for values that need a bespoke panel (e.g. the multi-line
    # system prompt). Still a normal hot knob for get()/set()/reload().
    hidden: bool = False


# The knob catalog. Keys mirror the original module constant names so the
# diff at the use sites is a 1:1 substitution.
SPECS: dict[str, Spec] = {
    # --- Voice / STT / LLM (restart-required: bound at construction) -----
    "PIPER_VOICE": Spec(
        "en_US-libritts_r-medium", "str", True, "voice",
        "Piper TTS voice model name.",
    ),
    "MODEL": Spec(
        "claude-haiku-4-5", "str", True, "voice",
        "Anthropic model for conversational turns.",
    ),
    "STT_DEVICE": Spec(
        "cuda", "str", True, "voice",
        "faster-whisper device (cuda/cpu).",
    ),
    "STT_COMPUTE_TYPE": Spec(
        "float16", "str", True, "voice",
        "faster-whisper compute type (float16/int8/...).",
    ),
    "STT_MODEL": Spec(
        "small.en", "str", True, "voice",
        "faster-whisper model size.",
    ),
    "STT_DEBUG_DUMP": Spec(
        0, "int", False, "voice",
        "Write each captured utterance to ~/.stackchan/captures/*.wav so the "
        "raw STT input can be listened to when transcription is wrong. "
        "1=on, 0=off. Hot — no restart needed.",
    ),

    # --- VAD / capture (hot) --------------------------------------------
    "SPEECH_RMS": Spec(
        150.0, "float", False, "capture",
        "RMS threshold for a frame to count as voiced.",
    ),
    "SILENCE_TAIL_MS": Spec(
        700, "int", False, "capture",
        "Trailing silence that ends an utterance.",
    ),
    "SPEECH_LEAD_MS": Spec(
        200, "int", False, "capture",
        "Minimum voiced audio before silence can end an utterance.",
    ),
    "MAX_UTTERANCE_MS": Spec(
        10000, "int", False, "capture",
        "Hard cap on a single utterance length.",
    ),
    "FOLLOW_UP_WINDOW_S": Spec(
        8.0, "float", False, "capture",
        "How long the mic stays open after a reply without re-saying the wakeword.",
    ),
    "FOLLOW_UP_GUARD_S": Spec(
        0.5, "float", False, "capture",
        "Extra delay after TTS playback finishes before reopening the mic for "
        "the follow-up window, so the robot doesn't hear its own voice tail. "
        "Raise if the robot replies to itself; lower for snappier follow-ups.",
    ),

    # --- Behavior / vision (hot) ----------------------------------------
    "SLEEP_TIMEOUT_S": Spec(
        300.0, "float", False, "behavior",
        "Sleep after this many seconds with no interaction (conversation, "
        "wake word, or head tap). Sleeping turns off the screen and stops "
        "look-around + face detection until a wake word or tap. 0 = never "
        "sleep.",
    ),
    "GREETING_COOLDOWN_S": Spec(
        1800.0, "float", False, "behavior",
        "Minimum time between proactive greetings.",
    ),
    "RECENT_INTERACTION_S": Spec(
        90.0, "float", False, "behavior",
        "Suppress greeting if the user spoke within this window.",
    ),
    "DETECT_INTERVAL_IDLE_S": Spec(
        20.0, "float", False, "behavior",
        "Seconds between face-detection passes when idle.",
    ),
    "LOOK_AROUND_INTERVAL_S": Spec(
        180.0, "float", False, "behavior",
        "Seconds between idle look-around sweeps.",
    ),
    "LOOK_AROUND_POSE_DURATION_S": Spec(
        4.0, "float", False, "behavior",
        "Dwell time at each look-around pose.",
    ),
    "LOOK_AROUND_SPEED": Spec(
        350, "int", False, "behavior",
        "Servo spring speed (0-1000) for look-around sweep poses; higher = "
        "snappier glance. Centering and agent look_at keep the gentler "
        "firmware default.",
    ),
    "CENTERING_COOLDOWN_S": Spec(
        180.0, "float", False, "behavior",
        "Lockout after centering on a face before the next centering.",
    ),
    "CENTERING_GAIN": Spec(
        0.7, "float", False, "behavior",
        "Under-centering factor (FOV is approximate).",
    ),

    # --- MCP / tools (hot) ----------------------------------------------
    "DEFAULT_LOCATION": Spec(
        "Seattle, Washington", "str", False, "tools",
        "Default location for the weather tool when the user doesn't name one.",
    ),
    # Note: HUE_BRIDGE_IP and HUE_TOKEN live in .env, not here — the Hue
    # MCP server reads them from its environment (HUE_BRIDGE_IP passes
    # through the child env; HUE_TOKEN is injected via the server's
    # env_ref). Set once out-of-band; see mcp_servers/README.md.

    # --- Responsiveness during slow tool calls (hot) --------------------
    # int 0/1 are used as booleans (the config layer has no bool type);
    # `get()` returns the int and truthiness works at the use site.
    "ACK_FILLER": Spec(
        1, "int", False, "responsiveness",
        "Speak a short canned acknowledgement (e.g. 'Let me check on "
        "that.') when a tool call runs and the model hasn't already said "
        "something, so there's audio within ~1s even on a slow tool. "
        "1=on, 0=off.",
    ),
    "ACK_FILLER_PHRASES": Spec(
        "Let me check on that.|One sec.|Let me take a look.|"
        "Give me just a moment.",
        "str", False, "responsiveness",
        "Pipe-separated acknowledgement phrases; one is chosen at random "
        "each time. Leave empty to disable the spoken ack regardless of "
        "ACK_FILLER.",
    ),
    "BUSY_INDICATOR": Spec(
        1, "int", False, "responsiveness",
        "Show an on-screen 'thinking' indicator (a '…' speech bubble) "
        "while tool calls run, cleared when the reply begins. 1=on, 0=off. "
        "Requires firmware with the set_busy command.",
    ),

    # --- Memory / summarizer (hot) --------------------------------------
    "SUMMARIZE_TRIGGER": Spec(
        20, "int", False, "memory",
        "Unsummarized-turn backlog that triggers a background fold into a "
        "summary. Lower = summarize more eagerly (smaller prompts, more "
        "LLM calls).",
    ),
    "KEEP_RECENT_TURNS": Spec(
        10, "int", False, "memory",
        "How many of the most recent turns to always keep verbatim (never "
        "fold into a summary).",
    ),
    "SUMMARIZE_SYSTEM": Spec(
        "", "str", False, "memory",
        "Override for the summarizer's system prompt. Empty = built-in "
        "default (claude_agent.DEFAULT_SUMMARIZE_SYSTEM). Read per fold.",
        hidden=True,
    ),

    # --- Claude Buddy (hot) — Option C Milestone 0 ----------------------
    "BUDDY_MODE": Spec(
        1, "int", False, "buddy",
        "Enable Claude Buddy: surface Claude Code tool-permission prompts on "
        "the robot (tap to approve, wake word to deny). 1=on, 0=off (the "
        "/buddy/permission endpoint then returns the fallback decision).",
    ),
    "BUDDY_PERMISSION_TIMEOUT_S": Spec(
        60.0, "float", False, "buddy",
        "How long to wait for a tap/wake-word approval before falling back to "
        "BUDDY_PERMISSION_FALLBACK.",
    ),
    "BUDDY_PERMISSION_FALLBACK": Spec(
        "ask", "str", False, "buddy",
        "Decision returned when no one responds in time, the robot is "
        "disconnected, or Buddy is off: 'ask' (normal Claude Code prompt), "
        "'deny', or 'allow'. 'ask' is safest — never auto-approves and never "
        "hangs the session.",
    ),
    "BUDDY_WAITING_EXPRESSION": Spec(
        "surprised", "str", False, "buddy",
        "Facial expression shown while awaiting an approval (maps via "
        "set_expression; 'surprised' renders as the Doubt face).",
    ),
    "BUDDY_SPEAK_PROMPTS": Spec(
        1, "int", False, "buddy",
        "Speak the pending tool name aloud when an approval is needed "
        "(e.g. 'Approve Bash?'), so it's usable without looking. 1=on, 0=off.",
    ),

    # --- Persona (hot; edited via its own console panel, not the grid) ---
    "SYSTEM_PROMPT": Spec(
        "", "str", False, "persona",
        "Override for the robot's core persona/system prompt. Empty = use "
        "the built-in default (claude_agent.DEFAULT_SYSTEM_PROMPT). Read "
        "per-turn, so an edit applies on the next conversation turn.",
        hidden=True,
    ),
}


def _coerce(spec: Spec, value: Any) -> Any:
    if spec.type == "float":
        return float(value)
    if spec.type == "int":
        return int(value)
    return str(value)


class Config:
    """In-memory view of the knobs, backed by the `config` table.

    Not thread-safe — same single-event-loop contract as Memory."""

    def __init__(self, memory: Any) -> None:
        self._memory = memory
        self._overrides: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        stored = self._memory.get_all_config()
        clean: dict[str, Any] = {}
        for key, raw in stored.items():
            if key not in SPECS:
                log.warning("ignoring unknown config key in db: %r", key)
                continue
            try:
                clean[key] = _coerce(SPECS[key], raw)
            except (TypeError, ValueError):
                log.warning("bad stored value for %s: %r — using default", key, raw)
        self._overrides = clean

    def get(self, key: str) -> Any:
        spec = SPECS.get(key)
        if spec is None:
            raise KeyError(f"unknown config key: {key}")
        return self._overrides.get(key, spec.default)

    def set(self, key: str, value: Any) -> Any:
        spec = SPECS.get(key)
        if spec is None:
            raise KeyError(f"unknown config key: {key}")
        coerced = _coerce(spec, value)
        self._memory.set_config(key, coerced)
        self._overrides[key] = coerced
        log.info("config set %s = %r%s", key, coerced,
                 " (restart required)" if spec.restart else "")
        return coerced

    def describe(self) -> list[dict[str, Any]]:
        """Snapshot for the web UI: every knob with current/default/meta."""
        out = []
        for key, spec in SPECS.items():
            if spec.hidden:
                continue
            out.append({
                "key": key,
                "value": self.get(key),
                "default": spec.default,
                "type": spec.type,
                "restart": spec.restart,
                "group": spec.group,
                "help": spec.help,
                "overridden": key in self._overrides,
            })
        return out


_config: Config | None = None


def init_config(memory: Any) -> Config:
    global _config
    _config = Config(memory)
    return _config


def get_config() -> Config:
    if _config is None:
        raise RuntimeError("config not initialized — call init_config(memory) first")
    return _config
