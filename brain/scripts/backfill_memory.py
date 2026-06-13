#!/usr/bin/env python3
"""One-time migration: seed permanent memory from existing summaries, then
prune episodic summaries to the retention window.

Run ONCE per device, with the brain stopped, after deploying the two-tier
memory change. Back up the DB first:

    ssh jetson
    cd ~/code/stackchanagent/brain
    cp ~/.stackchan/memory.db ~/.stackchan/memory.db.bak
    .venv/bin/python scripts/backfill_memory.py            # live DB
    .venv/bin/python scripts/backfill_memory.py /tmp/x.db  # a copy (dry run)

It (1) harvests enduring facts from ALL existing summaries into known_facts
and (2) prunes summaries down to SUMMARY_RETENTION (purging their turns).
Nothing outside the SQLite DB is touched. Requires ANTHROPIC_API_KEY in
brain/.env (same as the agent)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Resolve imports relative to brain/ regardless of CWD.
BRAIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BRAIN))

from dotenv import load_dotenv

load_dotenv(BRAIN.parent / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

import claude_agent  # noqa: E402
from config import get_config, init_config  # noqa: E402
from memory import Memory  # noqa: E402


async def main() -> None:
    db_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    memory = Memory(db_path)
    init_config(memory)
    cfg = get_config()
    model = cfg.get("MODEL")
    retention = int(cfg.get("SUMMARY_RETENTION"))

    summaries = memory.list_summaries()
    facts_before = memory.list_facts()
    print(f"DB: {memory.path}")
    print(f"before: {len(summaries)} summaries, {len(facts_before)} facts")

    if summaries:
        transcript = "\n\n".join(s.summary for s in summaries)
        client = AsyncAnthropic()
        new_facts = await claude_agent.extract_facts(
            client, model, transcript, facts_before
        )
        added = memory.merge_facts(new_facts) if new_facts else 0
        print(f"extracted {len(new_facts)} candidate fact(s); {added} new:")
        for f in new_facts:
            print(f"  - {f}")
    else:
        print("no summaries to harvest")

    s_del, t_del = memory.prune_summaries(retention)
    print(f"pruned {s_del} summaries / {t_del} turns (retention={retention})")
    print(
        f"after: {len(memory.list_summaries())} summaries, "
        f"{len(memory.list_facts())} facts"
    )
    memory.close()


if __name__ == "__main__":
    asyncio.run(main())
