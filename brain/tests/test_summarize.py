"""The summarizer's LLM calls run outside the session turn lock (a user turn
must never queue behind two summarizer round-trips); only the write — saving
the summary and re-syncing the in-memory thread — takes it, and a backlog that
changed underneath the LLM call is not summarized."""

from __future__ import annotations

from types import SimpleNamespace

import claude_agent
from config import get_config


def _seed_turns(mem, n: int) -> None:
    for i in range(n):
        mem.append_turn("user", f"question {i}")
        mem.append_turn("assistant", f"answer {i}")


class _FakeLLM:
    """chat.completions.create stand-in; `during` runs mid-call, where a
    concurrent turn would."""

    def __init__(self, sess_ref: dict, during=None) -> None:
        self.chat = SimpleNamespace(completions=self)
        self.sess_ref = sess_ref
        self.during = during
        self.locked_during_call: list[bool] = []
        self.calls: list[dict] = []

    async def create(self, **kw):
        self.calls.append(kw)
        self.locked_during_call.append(self.sess_ref["s"]._turn_lock.locked())
        if self.during:
            self.during()
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="they talked"))])


async def test_llm_runs_unlocked_and_commit_resyncs(mem, make_agent):
    get_config().set("SUMMARIZE_TRIGGER", 4)
    get_config().set("KEEP_RECENT_TURNS", 2)
    get_config().set("AUTO_FACT_EXTRACTION", 0)
    _seed_turns(mem, 3)
    sess = make_agent([])
    llm = _FakeLLM({"s": sess})
    sess.client = llm

    await claude_agent.maybe_summarize(sess)

    assert llm.locked_during_call == [False]
    assert len(mem.list_summaries()) == 1
    # The in-memory thread now matches the persisted, unsummarized tail.
    assert sess.messages == [t.message for t in mem.list_unsummarized_turns()]
    assert len(sess.messages) == 2
    # Background jobs use MODEL unless SUMMARY_MODEL overrides it.
    assert llm.calls[0]["model"] == get_config().get("MODEL")
    assert llm.calls[0]["messages"][0]["role"] == "system"


async def test_summary_model_overrides_model(mem, make_agent):
    get_config().set("SUMMARIZE_TRIGGER", 4)
    get_config().set("KEEP_RECENT_TURNS", 2)
    get_config().set("AUTO_FACT_EXTRACTION", 0)
    get_config().set("SUMMARY_MODEL", "vendor/cheap-model")
    _seed_turns(mem, 3)
    sess = make_agent([])
    llm = _FakeLLM({"s": sess})
    sess.client = llm
    await claude_agent.maybe_summarize(sess)
    assert llm.calls[0]["model"] == "vendor/cheap-model"


async def test_single_round_exchange_is_a_fold_boundary(mem, make_agent):
    # A device command that ended on its tool result must be foldable: the
    # summary may end on the tool row, and the verbatim tail starts at the
    # next user message.
    get_config().set("AUTO_FACT_EXTRACTION", 0)
    call = {"id": "c1", "type": "function",
            "function": {"name": "mcp__homeassistant__HassTurnOn", "arguments": "{}"}}
    ids = mem.append_turns([
        {"role": "user", "content": "light on"},
        {"role": "assistant", "content": "Turning on the light.", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "Any time."},
    ])
    sess = make_agent([])
    llm = _FakeLLM({"s": sess})
    summary, reason = await claude_agent.summarize_backlog(
        mem, llm, "m", keep_recent=2, force=True)
    assert reason == "ok"
    assert (summary.span_from, summary.span_to) == (ids[0], ids[2])
    transcript = llm.calls[0]["messages"][1]["content"]
    assert "HassTurnOn" in transcript and claude_agent.UNTRUSTED_OPEN in transcript
    tail = [t.message for t in mem.list_unsummarized_turns()]
    assert [m["role"] for m in tail] == ["user", "assistant"]


async def test_backlog_purged_mid_summary_is_not_summarized(mem, make_agent):
    get_config().set("SUMMARIZE_TRIGGER", 4)
    get_config().set("KEEP_RECENT_TURNS", 2)
    get_config().set("AUTO_FACT_EXTRACTION", 0)
    _seed_turns(mem, 3)
    sess = make_agent([])
    # A live turn's over-length recovery deletes the unsummarized history
    # while the summarizer's LLM call is in flight.
    sess.client = _FakeLLM({"s": sess}, during=mem.delete_unsummarized_turns)

    await claude_agent.maybe_summarize(sess)

    assert mem.list_summaries() == []


async def test_below_trigger_does_nothing(mem, make_agent, monkeypatch):
    get_config().set("SUMMARIZE_TRIGGER", 100)
    _seed_turns(mem, 2)
    sess = make_agent([])

    async def boom(*a, **k):
        raise AssertionError("summarize_backlog should not run below the trigger")

    monkeypatch.setattr(claude_agent, "summarize_backlog", boom)
    await claude_agent.maybe_summarize(sess)
