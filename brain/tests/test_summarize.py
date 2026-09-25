"""maybe_summarize runs its LLM calls outside the session turn lock (a user
turn must never queue behind two summarizer round-trips) and takes the lock
only to re-sync the in-memory thread from the persisted state."""

from __future__ import annotations

import claude_agent
from config import get_config
from memory import Summary


def _seed_turns(mem, n: int) -> None:
    for i in range(n):
        mem.append_turn("user", f"question {i}")
        mem.append_turn("assistant", [{"type": "text", "text": f"answer {i}"}])


async def test_llm_work_runs_without_the_turn_lock(mem, make_agent, monkeypatch):
    get_config().set("SUMMARIZE_TRIGGER", 4)
    _seed_turns(mem, 3)
    sess = make_agent([])
    seen: dict[str, bool] = {}

    async def fake_backlog(memory, client, model, *, keep_recent, force):
        seen["locked_during_llm"] = sess._turn_lock.locked()
        turns = memory.list_unsummarized_turns()
        sid = memory.save_summary(turns[0].id, turns[1].id, "they talked")
        return Summary(id=sid, summary="they talked",
                       span_from=turns[0].id, span_to=turns[1].id), "ok"

    monkeypatch.setattr(claude_agent, "summarize_backlog", fake_backlog)
    await claude_agent.maybe_summarize(sess)

    assert seen == {"locked_during_llm": False}
    # The in-memory thread now matches the persisted, unsummarized turns.
    assert sess.messages == [
        {"role": t.role, "content": t.content} for t in mem.list_unsummarized_turns()
    ]
    assert len(sess.messages) == 4


async def test_below_trigger_does_nothing(mem, make_agent, monkeypatch):
    get_config().set("SUMMARIZE_TRIGGER", 100)
    _seed_turns(mem, 2)
    sess = make_agent([])

    async def boom(*a, **k):
        raise AssertionError("summarize_backlog should not run below the trigger")

    monkeypatch.setattr(claude_agent, "summarize_backlog", boom)
    await claude_agent.maybe_summarize(sess)
