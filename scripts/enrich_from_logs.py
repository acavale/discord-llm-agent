#!/usr/bin/env python3
"""Batch profile enrichment: mine the whole message archive and append grounded per-member facts.

Unlike the live agent (which only reasons in the allowlisted channels), this reads the canonical guild JSONL
across ALL channels -- including the general/game chat -- groups each human's recent messages, and asks the
model to extract only NEW, durable facts about them, written through MemberMemory.note (which de-duplicates
against what's already known). Idempotent by construction: re-running proposes the same facts and the dedupe
drops them.

  # preview without writing
  python scripts/enrich_from_logs.py --dry-run
  # last 14 days, then apply
  python scripts/enrich_from_logs.py --days 14

Runs anywhere the storage root resolves. Intended as a scheduled job
(cron, launchd, or Task Scheduler). The running bot won't see the new
facts until it reloads (`!refresh`) or restarts.

Scope note: evidence is each member's OWN messages, so facts others state about a person aren't captured
here (the live post-chain reflection and manual edits cover those).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discord_agent import reflection
from discord_agent.config import load_config
from discord_agent.llm import LLM
from discord_agent.memory import MemberMemory
from discord_agent.store import Store

BOT_NAMES = {"agent"}  # never profile the bot itself


def _parse_ts(s: str):
    try:
        return datetime.datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _load_user_messages(log_path: Path, cutoff, max_msgs: int) -> dict[str, list[str]]:
    """username -> their most recent messages within the window (bot excluded, empty/image-only dropped)."""
    by_user: dict[str, list[str]] = defaultdict(list)
    if not log_path.exists():
        raise FileNotFoundError(f"archive not found: {log_path}")
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            user = (rec.get("username") or "").strip()
            uid = str(rec.get("user_id") or "")
            if not user or user.lower() in BOT_NAMES or uid in BOT_NAMES:
                continue
            msg = (rec.get("message") or "").strip()
            if not msg or msg == "[image]":
                continue
            ts = _parse_ts(rec.get("timestamp") or "")
            if cutoff is not None and ts is not None and ts < cutoff:
                continue
            by_user[user].append(msg)
    return {u: msgs[-max_msgs:] for u, msgs in by_user.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich member profiles from the chat archive.")
    ap.add_argument("--days", type=int, default=7, help="how far back to mine (0 = all history)")
    ap.add_argument("--max-msgs", type=int, default=60, help="cap on messages per user fed as evidence")
    ap.add_argument("--min-msgs", type=int, default=6, help="skip users with fewer messages in the window")
    ap.add_argument("--guild", default=None, help="guild id (default: bot.guild_id from config)")
    ap.add_argument("--dry-run", action="store_true", help="print proposed facts, write nothing")
    args = ap.parse_args()

    cfg = load_config()
    store = Store(cfg)
    guild_id = args.guild or str(cfg.get("bot", "guild_id", default=""))
    if not guild_id:
        print("no guild id (set bot.guild_id or pass --guild)")
        return 2
    log_path = store.guild_log(guild_id)

    cutoff = None
    if args.days > 0:
        cutoff = datetime.datetime.now() - datetime.timedelta(days=args.days)

    users = _load_user_messages(log_path, cutoff, args.max_msgs)
    users = {u: m for u, m in users.items() if len(m) >= args.min_msgs}
    if not users:
        print(f"no users with >= {args.min_msgs} messages in the window; nothing to do")
        return 0

    provider = str(cfg.get("llm", "provider", default="grok")).lower()
    temp = float(cfg.get("llm", "temperature", default=0.7))
    if provider == "ollama":
        llm = LLM(
            api_key="ollama",
            model=cfg.get("llm", "ollama", "model", default="qwen3:30b-a3b-instruct-2507-q8_0"),
            temperature=temp,
            timeout=int(cfg.get("llm", "ollama", "timeout", default=120)),
            base_url=cfg.get("llm", "ollama", "base_url", default="http://localhost:11434/v1"),
            extra_payload={"keep_alive": cfg.get("llm", "ollama", "keep_alive", default=-1)},
        )
    else:
        llm = LLM(
            api_key=cfg.secrets.xai_api_key,
            model=cfg.get("llm", "model", default="grok-4.20-0309-reasoning"),
            temperature=temp,
        )
    print(f"provider: {provider} (model {llm.model})")
    mm = MemberMemory(store.member_memory_file)

    # Rollback points: preserve the very first original once, and always snapshot the prior state.
    bak = store.member_memory_file.with_suffix(".bak.json")
    prev = store.member_memory_file.with_suffix(".prev.json")
    if not args.dry_run:
        cur = store.member_memory_file.read_text(encoding="utf-8") if store.member_memory_file.exists() else ""
        if cur and not bak.exists():
            bak.write_text(cur, encoding="utf-8")
            print(f"backed up original -> {bak.name}")
        if cur:
            prev.write_text(cur, encoding="utf-8")

    total_added = 0
    window_desc = "all history" if args.days == 0 else f"last {args.days}d"
    print(f"mining {log_path.name} ({window_desc}); {len(users)} active user(s)\n")
    for user in sorted(users):
        msgs = users[user]
        evidence = "\n".join(f"{user}: {m}" for m in msgs)
        known = mm.format_for_prompt(only_names={user})
        facts = reflection.extract_member_facts(llm, evidence, known, roster=[user])
        if not facts:
            print(f"  {user}: (nothing new from {len(msgs)} msgs)")
            continue
        if args.dry_run:
            for f in facts:
                print(f"  [dry] {f['name']}: {f['fact']}")
            continue
        applied = reflection.apply_facts(mm, facts)
        total_added += len(applied)
        if applied:
            for line in applied:
                print(f"  + {line}")
        else:
            print(f"  {user}: {len(facts)} proposed, all already known")

    if args.dry_run:
        print("\n(dry run: nothing written)")
    else:
        print(f"\napplied {total_added} new fact(s) to {store.member_memory_file.name}")
        if total_added:
            print("run !refresh (or restart the bot) to load them live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
