#!/usr/bin/env python3
"""Preflight checks for the agent. Run from the repo root:

    python scripts/preflight.py          # offline checks only
    python scripts/preflight.py --api    # also live xAI smoke tests (uses your key; a few tiny calls)

Offline: config loads, discord.py version, real-archive FTS ingest + recall, schedule parsing + channel
coverage, presence-name cross-check. Online: xAI text completion, tool-calling round trip, vision input.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discord_agent.config import load_config
from discord_agent.messages import Messages
from discord_agent.store import Store


def main() -> int:
    api = "--api" in sys.argv
    cfg = load_config()
    st = Store(cfg)
    guild = cfg.get("bot", "guild_id", default="")

    import discord

    print(f"[discord] version {discord.__version__}  (native Poll needs >= 2.4)")

    # Real-archive ingest + recall (messy live data path).
    db = Path(tempfile.mkdtemp()) / "preflight.db"
    m = Messages(st.guild_log(guild), db)
    n = m.ingest_archive()
    hits = m.search("fortnite", 5)
    print(f"[archive] ingested {n} messages; recall('fortnite') -> {len(hits)} hits")
    if n == 0:
        print("  WARN: nothing ingested - check the guild_*.jsonl path")
    m.close()

    # Schedules: enabled counts, channel coverage.
    sched = json.loads(st.scheduler_file.read_text(encoding="utf-8")).get("schedules", [])
    enabled = [s for s in sched if s.get("enabled", True)]
    chans = set(cfg.get("channels", default={}) or {})
    off_config = sorted({str(s.get("channel_id")) for s in enabled if str(s.get("channel_id")) not in chans})
    print(f"[schedules] {len(enabled)} enabled  by_type={dict(Counter(s.get('type') for s in enabled))}")
    if off_config:
        print(f"  note: enabled schedules target channels not named in config: {off_config}")

    # Presence-name cross-check (proxy for whether greetings will match real members).
    ctx = json.loads(st.member_memory_file.read_text(encoding="utf-8"))
    known = set()
    for u in ctx.get("users", []):
        known.add(u.get("name", "").lower())
        for a in u.get("aliases", []):
            known.add(a.lower())
    greeted = set()
    for s in enabled:
        if s.get("type") == "condition_based":
            for u in s.get("conditions", {}).get("users_online", []):
                greeted.add(u.lower())
    unmatched = sorted(greeted - known)
    print(f"[presence] {len(greeted)} greeted users; not in server_context: {unmatched or 'none'}")

    if api:
        from discord_agent.llm import LLM

        llm = LLM(cfg.secrets.xai_api_key, cfg.get("llm", "model", default="grok-4-fast-reasoning"), 0.5)
        r = llm.complete([{"role": "user", "content": "Reply with the single word: ok"}], None)
        print(f"[xai text]   -> {r['content'][:60]!r}")
        tools = [{"type": "function", "function": {
            "name": "ping", "description": "a test tool", "parameters": {"type": "object", "properties": {}}}}]
        r2 = llm.complete([{"role": "user", "content": "Call the ping tool now."}], tools)
        print(f"[xai tools]  -> tool_calls={[t['name'] for t in r2['tool_calls']]}")
        png = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
               "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        r3 = llm.complete([{"role": "user", "content": [
            {"type": "text", "text": "Reply with one word describing this image."},
            {"type": "image_url", "image_url": {"url": png}}]}], None)
        print(f"[xai vision] -> {r3['content'][:60]!r}  (empty likely means image input was rejected)")

    print("PREFLIGHT DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
